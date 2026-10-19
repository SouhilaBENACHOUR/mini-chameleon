#include "myblas.h"

int dgemm_jik( CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA,
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

    for (j = 0; j < N; j++) {
        for (i = 0; i < M; i++) {
            double sum = 0.0;
            for (k = 0; k < K; k++) {
                sum += A[lda * k + i] * B[ldb * j + k];
            }
            C[ldc * j + i] += alpha * sum;
        }
    }

    return ALGONUM_SUCCESS;
}

static dgemm_fct_t valid_dgemm_jik __attribute__((unused)) = dgemm_jik;
fct_list_t fct_dgemm_jik;

void dgemm_jik_init(void) __attribute__((constructor));
void dgemm_jik_init(void)
{
    fct_dgemm_jik.tiled = 0;
    fct_dgemm_jik.starpu = 0;
    fct_dgemm_jik.name = "jik";
    fct_dgemm_jik.helper = "Sequential GEMM, loop order j-i-k";
    fct_dgemm_jik.fctptr = dgemm_jik;
    fct_dgemm_jik.next = NULL;
    register_fct(&fct_dgemm_jik, ALGO_GEMM);
}
