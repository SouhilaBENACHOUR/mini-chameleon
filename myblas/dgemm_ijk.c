#include "myblas.h"

int dgemm_ijk( CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA,
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

    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            double sum = 0.0;
            for (k = 0; k < K; k++) {
                sum += A[lda * k + i] * B[ldb * j + k];
            }
            C[ldc * j + i] += alpha * sum;
        }
    }

    return ALGONUM_SUCCESS;
}

static dgemm_fct_t valid_dgemm_ijk __attribute__((unused)) = dgemm_ijk;
fct_list_t fct_dgemm_ijk;

void dgemm_ijk_init(void) __attribute__((constructor));
void dgemm_ijk_init(void)
{
    fct_dgemm_ijk.tiled = 0;
    fct_dgemm_ijk.starpu = 0;
    fct_dgemm_ijk.name = "ijk";
    fct_dgemm_ijk.helper = "Sequential GEMM, loop order i-j-k";
    fct_dgemm_ijk.fctptr = dgemm_ijk;
    fct_dgemm_ijk.next = NULL;
    register_fct(&fct_dgemm_ijk, ALGO_GEMM);
}
