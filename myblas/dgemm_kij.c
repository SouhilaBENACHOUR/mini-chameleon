#include "myblas.h"

int dgemm_kij( CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA,
               CBLAS_TRANSPOSE transB, const int M, const int N,
               const int K, const double alpha, const double *A,
               const int lda, const double *B, const int ldb,
               const double beta, double *C, const int ldc )
{
    int i, j, k;

    if (layout != CblasColMajor ||
        transA != CblasNoTrans || transB != CblasNoTrans) {
        return ALGONUM_NOT_IMPLEMENTED;
    }

    for (j = 0; j < N; j++) {
        for (i = 0; i < M; i++) {
            C[ldc * j + i] *= beta;
        }
    }

    for (k = 0; k < K; k++) {
        for (i = 0; i < M; i++) {
            double aik = alpha * A[lda * k + i];
            for (j = 0; j < N; j++) {
                C[ldc * j + i] += aik * B[ldb * j + k];
            }
        }
    }

    return ALGONUM_SUCCESS;
}

static dgemm_fct_t valid_dgemm_kij __attribute__((unused)) = dgemm_kij;
fct_list_t fct_dgemm_kij;

void dgemm_kij_init(void) __attribute__((constructor));
void dgemm_kij_init(void)
{
    fct_dgemm_kij.tiled = 0;
    fct_dgemm_kij.starpu = 0;
    fct_dgemm_kij.name = "kij";
    fct_dgemm_kij.helper = "Sequential GEMM, loop order k-i-j";
    fct_dgemm_kij.fctptr = dgemm_kij;
    fct_dgemm_kij.next = NULL;
    register_fct(&fct_dgemm_kij, ALGO_GEMM);
}
