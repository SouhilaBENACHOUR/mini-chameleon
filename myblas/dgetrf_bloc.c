
/**
 * @brief Sequential blocked LU factorization without pivoting.
 */
#include "myblas.h"

int
dgetrf_bloc( CBLAS_LAYOUT layout, int M, int N, double *A, int lda )
{
    int b = myblas_getenv_value_int( "LU_BLOCKSIZE", 32 ); /* Block size */
    int k, m, n;
    int K = (M < N) ? M : N;
    if ( b < 1 ) b = 32; /* garde-fou : sinon boucle infinie si LU_BLOCKSIZE <= 0 */

    if (layout != CblasColMajor) {
        return ALGONUM_NOT_IMPLEMENTED;
    }

    for (k = 0; k < K; k += b) {
        int kb = (K - k < b) ? K - k : b;
        int rows = M - k;
        int cols = N - k;

        /*
         * Factorize the diagonal block.
         * A + lda*k + k points to A(k,k).
         */
        dgetf2_vecto(CblasColMajor, kb, kb, A + lda * k + k, lda);

        /*
         * Compute the block to the right:
         * U12 = L11^{-1} A12
         */
        if (k + kb < N) {
            cblas_dtrsm(
                CblasColMajor, CblasLeft, CblasLower,
                CblasNoTrans, CblasUnit,
                kb, N - k - kb,
                1.0,
                A + lda * k + k, lda,
                A + lda * (k + kb) + k, lda
            );
        }

        /*
         * Compute the block below:
         * L21 = A21 U11^{-1}
         */
        if (k + kb < M) {
            cblas_dtrsm(
                CblasColMajor, CblasRight, CblasUpper,
                CblasNoTrans, CblasNonUnit,
                M - k - kb, kb,
                1.0,
                A + lda * k + k, lda,
                A + lda * k + (k + kb), lda
            );
        }

        /*
         * Update the trailing matrix:
         * A22 = A22 - L21 * U12
         */
        if (k + kb < M && k + kb < N) {
            dgemm_seq(
                CblasColMajor, CblasNoTrans, CblasNoTrans,
                M - k - kb, N - k - kb, kb,
                -1.0,
                A + lda * k + (k + kb), lda,
                A + lda * (k + kb) + k, lda,
                1.0,
                A + lda * (k + kb) + (k + kb), lda
            );
        }

        (void)m;
        (void)n;
        (void)rows;
        (void)cols;
    }

    return ALGONUM_SUCCESS;
}

/* Register the blocked LU implementation */
static dgetrf_fct_t valid_dgetrf_bloc __attribute__((unused)) = dgetrf_bloc;

fct_list_t fct_dgetrf_bloc;

void dgetrf_bloc_init(void) __attribute__((constructor));

void
dgetrf_bloc_init(void)
{
    fct_dgetrf_bloc.mpi    = 0;
    fct_dgetrf_bloc.tiled  = 0;
    fct_dgetrf_bloc.starpu = 0;
    fct_dgetrf_bloc.name   = "bloc";
    fct_dgetrf_bloc.helper = "Sequential blocked LU factorization";
    fct_dgetrf_bloc.fctptr = dgetrf_bloc;
    fct_dgetrf_bloc.next   = NULL;

    register_fct(&fct_dgetrf_bloc, ALGO_GETRF);
}
