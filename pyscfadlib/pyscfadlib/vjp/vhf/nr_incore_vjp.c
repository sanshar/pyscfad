#include <stdlib.h>
#include <math.h>
#include "config.h"
#include "vjp/util/util.h"

void CVHFics4_vj_dms_deriv(double* eri, double* vjp, double* vjk_bar,
                           int nao, int i, int j)
{// i >= j
    int k, l, kl;
    double fac = vjk_bar[i*nao+j];
    if (i != j) {
        fac += vjk_bar[j*nao+i];
    }

    for (k = 0, kl = 0; k < nao; k++) {
        for (l = 0; l < k; l++, kl++) {
            vjp[k*nao+l] += eri[kl] * fac;
            vjp[l*nao+k] += eri[kl] * fac;
        }
        vjp[k*nao+k] += eri[kl] * fac;
        kl++;
    }
}


void CVHFics4_vk_dms_deriv(double *eri, double *vjp, double *vjk_bar,
                           int nao, int i, int j)
{
    int k, l, kl;
    if (i > j) {
        for (k = 0, kl = 0; k < nao; k++) {
            for (l = 0; l < k; l++, kl++) {
                vjp[i*nao+k] += eri[kl] * vjk_bar[j*nao+l];
                vjp[i*nao+l] += eri[kl] * vjk_bar[j*nao+k];
                vjp[j*nao+k] += eri[kl] * vjk_bar[i*nao+l];
                vjp[j*nao+l] += eri[kl] * vjk_bar[i*nao+k];
            }
            vjp[i*nao+k] += eri[kl] * vjk_bar[j*nao+k];
            vjp[j*nao+k] += eri[kl] * vjk_bar[i*nao+k];
            kl++;
        }
    } else if (i == j) {
        for (k = 0, kl = 0; k < nao; k++) {
            for (l = 0; l < k; l++, kl++) {
                vjp[i*nao+k] += eri[kl] * vjk_bar[i*nao+l];
                vjp[i*nao+l] += eri[kl] * vjk_bar[i*nao+k];
            }
            vjp[i*nao+k] += eri[kl] * vjk_bar[i*nao+k];
            kl++;
        }
    }
}


void CVHFics4_vj_eri_deriv(double* vjp, double* dm, double* vjk_bar,
                           int nao, int i, int j)
{// i >= j
    int k, l, kl;
    double fac = vjk_bar[i*nao+j];
    if (i != j) {
        fac += vjk_bar[j*nao+i];
    }

    for (k = 0, kl = 0; k < nao; k++) {
        for (l = 0; l < k; l++, kl++) {
            vjp[kl] += fac * (dm[k*nao+l] + dm[l*nao+k]);
        }
        vjp[kl] += dm[k*nao+k] * fac;
        kl++;
    }
}


void CVHFics4_vk_eri_deriv(double *vjp, double *dm, double *vjk_bar,
                           int nao, int i, int j)
{
    int k, l, kl;
    if (i > j) {
        for (k = 0, kl = 0; k < nao; k++) {
            for (l = 0; l < k; l++, kl++) {
                vjp[kl] += dm[i*nao+k] * vjk_bar[j*nao+l];
                vjp[kl] += dm[i*nao+l] * vjk_bar[j*nao+k];
                vjp[kl] += dm[j*nao+k] * vjk_bar[i*nao+l];
                vjp[kl] += dm[j*nao+l] * vjk_bar[i*nao+k];
            }
            vjp[kl] += dm[i*nao+k] * vjk_bar[j*nao+k];
            vjp[kl] += dm[j*nao+k] * vjk_bar[i*nao+k];
            kl++;
        }
    } else if (i == j) {
        for (k = 0, kl = 0; k < nao; k++) {
            for (l = 0; l < k; l++, kl++) {
                vjp[kl] += dm[i*nao+k] * vjk_bar[i*nao+l];
                vjp[kl] += dm[i*nao+l] * vjk_bar[i*nao+k];
            }
            vjp[kl] += dm[i*nao+k] * vjk_bar[i*nao+k];
            kl++;
        }
    }
}


void CVHFnrs4_incore_dms_vjp(double *vjp, double *eri, double *vjk_bar,
                             int n_dm, int nao, void (*fjk)())
{
    double **vjp_loc_bufs = calloc(omp_get_max_threads_safe(), sizeof(double *));
    #pragma omp parallel default(none) \
        shared(vjp, eri, vjk_bar, n_dm, nao, fjk, vjp_loc_bufs)
    {
        int i, j, ic;
        size_t ij;
        size_t npair = nao*(nao+1)/2;
        size_t nn = nao * nao;
        int thread_id = omp_get_thread_num();
        double *vjp_loc;
        if (thread_id == 0) {
            vjp_loc = vjp;
        } else {
            vjp_loc = calloc(nn*n_dm, sizeof(double));
        }
        vjp_loc_bufs[thread_id] = vjp_loc;
        double *ptr_eri, *ptr_vjp_loc, *ptr_vjk_bar;
        #pragma omp for schedule(static, 4)
        for (ij = 0; ij < npair; ij++) {
            i = (int)(sqrt(2*ij+.25) - .5 + 1e-7);
            j = ij - i*(i+1)/2;
            ptr_eri = eri + ij*npair;
            for (ic = 0; ic < n_dm; ic++) {
                ptr_vjp_loc = vjp_loc + ic*nn;
                ptr_vjk_bar = vjk_bar + ic*nn;
                (*fjk)(ptr_eri, ptr_vjp_loc, ptr_vjk_bar, nao, i, j);
            }
        }
        omp_dsum_reduce_inplace(vjp_loc_bufs, nn*n_dm);
        if (thread_id != 0) {
            free(vjp_loc);
        }
    }
    free(vjp_loc_bufs);
}


void CVHFnrs4_incore_eri_vjp(double *vjp, double *dms, double *vjk_bar,
                             int n_dm, int nao, void (*fjk)())
{
    int ic;
    size_t nn = nao * nao;
    size_t npair = nao*(nao+1)/2;
    for (ic = 0; ic < n_dm; ic++) {
        #pragma omp parallel
        {
            int i, j;
            size_t ij, off;
            #pragma omp for schedule(static, 4)
            for (ij = 0; ij < npair; ij++) {
                i = (int)(sqrt(2*ij+.25) - .5 + 1e-7);
                j = ij - i*(i+1)/2;
                off = ij * npair;
                (*fjk)(vjp+off, dms, vjk_bar, nao, i, j);
            }
        }
        dms += nn;
        vjk_bar += nn;
    }
}
