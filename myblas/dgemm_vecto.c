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

        /* k par paquets de 4 : C lu et ecrit 1 fois pour 4 FMA */
        for( k=0; k+4<=K; k+=4 ) {
            const double *A0 = A + lda * k;    /* colonnes k a k+3 de A */
            const double *A1 = A0 + lda;
            const double *A2 = A1 + lda;
            const double *A3 = A2 + lda;
            double b0 = alpha * B[ ldb * n + k     ];
            double b1 = alpha * B[ ldb * n + k + 1 ];
            double b2 = alpha * B[ ldb * n + k + 2 ];
            double b3 = alpha * B[ ldb * n + k + 3 ];
            __m256d vb0 = _mm256_set1_pd( b0 );
            __m256d vb1 = _mm256_set1_pd( b1 );
            __m256d vb2 = _mm256_set1_pd( b2 );
            __m256d vb3 = _mm256_set1_pd( b3 );

            /* Partie vectorielle : 4 lignes a la fois */
            for( m=0; m+4<=M; m+=4 ) {
                __m256d vc = _mm256_loadu_pd( Cn + m );                 /* 1 lecture  */
                vc = _mm256_fmadd_pd( _mm256_loadu_pd( A0 + m ), vb0, vc );
                vc = _mm256_fmadd_pd( _mm256_loadu_pd( A1 + m ), vb1, vc );
                vc = _mm256_fmadd_pd( _mm256_loadu_pd( A2 + m ), vb2, vc );
                vc = _mm256_fmadd_pd( _mm256_loadu_pd( A3 + m ), vb3, vc );
                _mm256_storeu_pd( Cn + m, vc );                         /* 1 ecriture */
            }
            /* Reste : au plus 3 lignes */
            for( ; m<M; m++ ) {
                Cn[m] += A0[m] * b0 + A1[m] * b1 + A2[m] * b2 + A3[m] * b3;
            }
        }

        /* Reste de k (0 a 3 valeurs) : comme la version 1 */
        for( ; k<K; k++ ) {
            const double *Ak = A + lda * k;
            double  bkn = alpha * B[ ldb * n + k ];
            __m256d vb  = _mm256_set1_pd( bkn );
            for( m=0; m+4<=M; m+=4 ) {
                __m256d va = _mm256_loadu_pd( Ak + m );
                __m256d vc = _mm256_loadu_pd( Cn + m );
                vc = _mm256_fmadd_pd( va, vb, vc );
                _mm256_storeu_pd( Cn + m, vc );
            }
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
