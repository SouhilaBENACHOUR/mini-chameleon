/**
 * @file dgemm_vecto.c
 * @brief DGEMM sequentiel vectorise (AVX2 + FMA) sur la boucle interne m.
 *        Cas transposes : version scalaire (non vectorisee).
 */
#include "myblas.h"
#include <immintrin.h>

/* Cas transposes : recopie des branches de l'ancien dgemm_seq */
static void
dgemm_vecto_trans( CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                   int M, int N, int K, double alpha,
                   const double *A, int lda, const double *B, int ldb,
                   double beta, double *C, int ldc )
{
    int m, n, k;
    for( m=0; m<M; m++ ) {
        for( n=0; n<N; n++ ) {
            if ( beta != 1. ) {
                C[ ldc * n + m ] = beta * C[ ldc * n + m ];
            }
            for( k=0; k<K; k++ ) {
                double a = ( transA == CblasNoTrans ) ? A[ lda * k + m ] : A[ lda * m + k ];
                double b = ( transB == CblasNoTrans ) ? B[ ldb * n + k ] : B[ ldb * k + n ];
                C[ ldc * n + m ] += alpha * a * b;
            }
        }
    }
}

__attribute__((target("avx2,fma")))
int dgemm_vecto( CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA,
                 CBLAS_TRANSPOSE transB, const int M, const int N,
                 const int K, const double alpha, const double *A,
                 const int lda, const double *B, const int ldb,
                 const double beta, double *C, const int ldc )
{
    int m, n, k;
    (void)layout; /* column-major */

    if ( transA != CblasNoTrans || transB != CblasNoTrans ) {
        dgemm_vecto_trans( transA, transB, M, N, K, alpha,
                           A, lda, B, ldb, beta, C, ldc );
        return ALGONUM_SUCCESS;
    }

    for( n=0; n<N; n++ ) {
        double *Cn = C + ldc * n;              /* colonne n de C */

        /* C(:,n) <- beta * C(:,n) */
        if ( beta != 1. ) {
            for( m=0; m<M; m++ ) {
                Cn[m] = beta * Cn[m];
            }
        }

        for( k=0; k<K; k++ ) {
            const double *Ak = A + lda * k;    /* colonne k de A */
            double  bkn = alpha * B[ ldb * n + k ];
            __m256d vb  = _mm256_set1_pd( bkn );   /* (b, b, b, b) */

            /* Partie vectorielle : 4 lignes a la fois */
            for( m=0; m+4<=M; m+=4 ) {
                __m256d va = _mm256_loadu_pd( Ak + m );
                __m256d vc = _mm256_loadu_pd( Cn + m );
                vc = _mm256_fmadd_pd( va, vb, vc );  /* vc = va*vb + vc */
                _mm256_storeu_pd( Cn + m, vc );
            }
            /* Reste : au plus 3 lignes */
            for( ; m<M; m++ ) {
                Cn[m] += Ak[m] * bkn;
            }
        }
    }

    return ALGONUM_SUCCESS;
}

/* To make sure we use the right prototype */
static dgemm_fct_t valid_dgemm_vecto __attribute__ ((unused)) = dgemm_vecto;

fct_list_t fct_dgemm_vecto;

void dgemm_vecto_init( void ) __attribute__( ( constructor ) );
void
dgemm_vecto_init( void )
{
    fct_dgemm_vecto.mpi    = 0;
    fct_dgemm_vecto.tiled  = 0;
    fct_dgemm_vecto.starpu = 0;
    fct_dgemm_vecto.name   = "vecto";
    fct_dgemm_vecto.helper = "Sequential DGEMM vectorized (AVX2/FMA)";
    fct_dgemm_vecto.fctptr = dgemm_vecto;
    fct_dgemm_vecto.next   = NULL;

    register_fct( &fct_dgemm_vecto, ALGO_GEMM );
}
