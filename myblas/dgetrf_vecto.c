/**
 * @file dgetrf_vecto.c
 * @brief LU non bloquee vectorisee (AVX2 + FMA).
 *        Meme calcul que dgetf2, mais la boucle interne parcourt les lignes m
 *        (contigues en memoire) et traite 4 lignes a la fois.
 */
#include "myblas.h"
#include <immintrin.h>

__attribute__((target("avx2,fma")))
int
dgetf2_vecto( CBLAS_LAYOUT layout, int M, int N, double *A, int lda )
{
    int m, n, k;
    int K = ( M > N ) ? N : M;
    (void)layout; /* column-major */

    for( k=0; k<K; k++ ) {
        double *Ak   = A + lda * k;              /* colonne k */
        double  piv  = Ak[k];                    /* pivot A(k,k) */
        __m256d vpiv = _mm256_set1_pd( piv );    /* (piv, piv, piv, piv) */

        /* a. colonne k de L : A(k+1:M, k) = A(k+1:M, k) / pivot */
        for( m=k+1; m+4<=M; m+=4 ) {
            __m256d va = _mm256_loadu_pd( Ak + m );
            _mm256_storeu_pd( Ak + m, _mm256_div_pd( va, vpiv ) );
        }
        for( ; m<M; m++ ) {                      /* reste : au plus 3 lignes */
            Ak[m] = Ak[m] / piv;
        }

        /* b. mise a jour : A(k+1:M, n) -= A(k+1:M, k) * A(k, n) */
        for( n=k+1; n<N; n++ ) {
            double *An  = A + lda * n;           /* colonne n */
            double  ukn = An[k];                 /* A(k,n), ligne k de U */
            __m256d vu  = _mm256_set1_pd( ukn ); /* (u, u, u, u) */

            for( m=k+1; m+4<=M; m+=4 ) {
                __m256d va = _mm256_loadu_pd( Ak + m );
                __m256d vc = _mm256_loadu_pd( An + m );
                vc = _mm256_fnmadd_pd( va, vu, vc );   /* vc = vc - va*vu */
                _mm256_storeu_pd( An + m, vc );
            }
            for( ; m<M; m++ ) {                  /* reste : au plus 3 lignes */
                An[m] -= Ak[m] * ukn;
            }
        }
    }

    return ALGONUM_SUCCESS;
}

/* To make sure we use the right prototype */
static dgetrf_fct_t valid_dgetf2_vecto __attribute__ ((unused)) = dgetf2_vecto;

fct_list_t fct_dgetrf_unblocked_vecto;

void dgetrf_unblocked_vecto_init( void ) __attribute__( ( constructor ) );
void
dgetrf_unblocked_vecto_init( void )
{
    fct_dgetrf_unblocked_vecto.mpi    = 0;
    fct_dgetrf_unblocked_vecto.tiled  = 0;
    fct_dgetrf_unblocked_vecto.starpu = 0;
    fct_dgetrf_unblocked_vecto.name   = "unblocked_vecto";
    fct_dgetrf_unblocked_vecto.helper = "Unblocked LU vectorized (AVX2/FMA)";
    fct_dgetrf_unblocked_vecto.fctptr = dgetf2_vecto;
    fct_dgetrf_unblocked_vecto.next   = NULL;

    register_fct( &fct_dgetrf_unblocked_vecto, ALGO_GETRF );
}
