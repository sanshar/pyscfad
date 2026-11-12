#ifndef LIBGMBPT_QCHEM_STC_MP2_DETERMINISTIC_ENERGY_H
#define LIBGMBPT_QCHEM_STC_MP2_DETERMINISTIC_ENERGY_H
#include "stc_mp2_kernel_helpers.h"
#include "stc_mp2_kernel_qchem.h"

namespace libgmbpt { namespace qchem { namespace stc_mp2 {

struct deterministic_energy_result {
    double energy=0, seconds=0, bare_integrals_seconds=0;
    double propagation_seconds=0, contraction_seconds=0, backprop_seconds=0;
    size_t propagation_calls=0, backprop_calls=0, cached_targets=0;
    arma::mat bar_b, bar_occ, bar_virt;
};

// Every selected target shares the point's dressed factors and accumulated
// Gamma. Reverse the propagation once after all target/partner contractions.
// Bound the persistent bare-Gram cache independently of the selected group;
// uncached pairs use worker-local matrices instead of another point reverse.
inline deterministic_energy_result evaluate_deterministic_energy(stc_mp2_kernel& kernel,
        const arma::vec& beta,const arma::vec& weights,
        const std::vector<size_t>& group,int threads,bool gradients=true,
        size_t cache_targets=1) {
    const size_t no=kernel.occupied_fock().n_rows,nv=kernel.virtual_fock().n_rows;
    if (!no || !nv || group.empty() || beta.is_empty() || beta.n_elem!=weights.n_elem || threads<1 || !cache_targets)
        throw std::invalid_argument("Invalid deterministic selected energy input");
    kernel.select_occupied(group);
    kernel.retain_original_integrals(true);
    const arma::mat& bare=kernel.original_integrals();
    const size_t np=bare.n_rows;
    const double start=omp_get_wtime();
    deterministic_energy_result out;
    out.cached_targets=std::min(group.size(),cache_targets);
    std::vector<arma::cube> bare_grams(out.cached_targets);
    for(auto& g:bare_grams) g.set_size(nv,nv,no);
    double tick=omp_get_wtime();
    #pragma omp parallel for num_threads(threads) schedule(static)
    for(size_t task=0;task<out.cached_targets*no;++task) {
        const size_t s=task/no,j=task%no;
        arma::mat bi(const_cast<double*>(bare.colptr(group[s]*nv)),np,nv,false,true);
        arma::mat bj(const_cast<double*>(bare.colptr(j*nv)),np,nv,false,true);
        bare_grams[s].slice(j)=bi.t()*bj;
    }
    out.bare_integrals_seconds=omp_get_wtime()-tick;
    density::propagator dress(kernel.occupied_fock(),kernel.virtual_fock(),bare,beta,threads);
    if (gradients) out.bar_b.zeros(bare.n_rows,bare.n_cols);
    arma::mat c;
    for(size_t l=0;l<beta.n_elem;++l) {
        tick=omp_get_wtime();
        dress.apply(l,c);
        ++out.propagation_calls;
        out.propagation_seconds+=omp_get_wtime()-tick;
        tick=omp_get_wtime();
        arma::mat gamma;
        std::vector<omp_lock_t> locks;
        if (gradients) {
            gamma.zeros(bare.n_rows,bare.n_cols);
            locks.resize(no);
            for(auto& lock:locks) omp_init_lock(&lock);
        }
        double point_energy=0;
        #pragma omp parallel for num_threads(threads) schedule(static) reduction(+:point_energy)
        for(size_t task=0;task<group.size()*no;++task) {
            const size_t s=task/no,j=task%no,i=group[s];
            arma::mat bi(const_cast<double*>(bare.colptr(i*nv)),np,nv,false,true);
            arma::mat bj(const_cast<double*>(bare.colptr(j*nv)),np,nv,false,true);
            arma::mat ci(c.colptr(i*nv),np,nv,false,true);
            arma::mat cj(c.colptr(j*nv),np,nv,false,true);
            arma::mat scratch;
            if (s>=out.cached_targets) scratch=bi.t()*bj;
            const arma::mat& gb=s<out.cached_targets ? bare_grams[s].slice(j) : scratch;
            const arma::mat gc=ci.t()*cj;
            const arma::mat mb=weights(l)*(gc.t()-2*gc);
            point_energy+=arma::accu(gb%mb);
            if (gradients) {
                const arma::mat mc=weights(l)*(gb.t()-2*gb);
                const arma::mat dbi=bj*mb.t(),dbj=bi*mb;
                const arma::mat dci=cj*mc.t(),dcj=ci*mc;
                omp_set_lock(&locks[i]);
                out.bar_b.cols(i*nv,i*nv+nv-1)+=dbi;
                gamma.cols(i*nv,i*nv+nv-1)+=dci;
                omp_unset_lock(&locks[i]);
                omp_set_lock(&locks[j]);
                out.bar_b.cols(j*nv,j*nv+nv-1)+=dbj;
                gamma.cols(j*nv,j*nv+nv-1)+=dcj;
                omp_unset_lock(&locks[j]);
            }
        }
        for(auto& lock:locks) omp_destroy_lock(&lock);
        out.energy+=point_energy;
        out.contraction_seconds+=omp_get_wtime()-tick;
        if (gradients) {
            tick=omp_get_wtime();
            kernel.backpropagate_laplace_point(2*beta(l),gamma,out.bar_occ,out.bar_virt,out.bar_b);
            ++out.backprop_calls;
            out.backprop_seconds+=omp_get_wtime()-tick;
        }
    }
    out.seconds=omp_get_wtime()-start;
    kernel.restore_original_integrals(true);
    return out;
}

}}}
#endif
