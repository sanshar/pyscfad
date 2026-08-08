#ifndef HAVE_DEFINED_VJPUTIL_H
#define HAVE_DEFINED_VJPUTIL_H

void omp_dsum_reduce_inplace(double **vec, size_t count);

static inline int omp_get_max_threads_safe(void)
{
#ifdef _OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
}

void pack_tril(int n, double *tril, double *mat);

#endif
