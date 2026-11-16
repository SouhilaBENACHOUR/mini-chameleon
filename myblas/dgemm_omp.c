/**
 * @file dgemm_omp.c
 * @brief DGEMM bloque parallelise avec OpenMP.
 *        Les blocs de C sont repartis entre les threads ;
 *        chaque produit de blocs est fait par dgemm_vecto (sequentiel).
 */
#include "myblas.h"
#include <omp.h>

/* Taille de bloc, lue dans GEMM_BLOCKSIZE (48 par defaut) */
static int dgemm_omp_block_size = 48;

int dgemm_omp( CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA,
               CBLAS_TRANSPOSE transB, const int M, const int N,
               const int K, const double alpha, const double *A,
               const int lda, const double *B, const int ldb,
               const double beta, double *C, const int ldc )
{
    int b = dgemm_omp_block_size;
    if ( b < 1 ) b = 48;                         /* garde-fou */

    if ( layout != CblasColMajor ) {
        return ALGONUM_NOT_IMPLEMENTED;
    }

    int MT = my_iceil( M, b );                   /* nombre de blocs de lignes   */
    int NT = my_iceil( N, b );                   /* nombre de blocs de colonnes */
    int KT = my_iceil( K, b );
    int KTe = ( KT > 0 ) ? KT : 1;               /* K = 0 : C = beta * C seulement */

    /* Les blocs (n, m) de C sont independants : on les repartit entre threads.
     * La boucle sur k reste dans le meme thread (sinon race condition sur C). */
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for( int n = 0; n < NT; n++ ) {
        for( int m = 0; m < MT; m++ ) {
            int mm = ( m == MT-1 ) ? M - m * b : b;
            int nn = ( n == NT-1 ) ? N - n * b : b;
            double *Cb = C + (size_t)ldc * n * b + m * b;

            for( int k = 0; k < KTe; k++ ) {
                int kk = ( K == 0 ) ? 0 : ( ( k == KT-1 ) ? K - k * b : b );
                double lbeta = ( k == 0 ) ? beta : 1.;   /* beta une seule fois */

                const double *Ab = ( transA == CblasNoTrans )
                    ? A + (size_t)lda * k * b + m * b
                    : A + (size_t)lda * m * b + k * b;
                const double *Bb = ( transB == CblasNoTrans )
                    ? B + (size_t)ldb * n * b + k * b
                    : B + (size_t)ldb * k * b + n * b;

                dgemm_vecto( CblasColMajor, transA, transB, mm, nn, kk,
                             alpha, Ab, lda, Bb, ldb, lbeta, Cb, ldc );
            }
        }
    }

    return ALGONUM_SUCCESS;
}

/* To make sure we use the right prototype */
static dgemm_fct_t valid_dgemm_omp __attribute__ ((unused)) = dgemm_omp;

/* Declare the variable that will store the information about this version */
fct_list_t fct_dgemm_omp;

/**
 * @brief Registration function
 */
void dgemm_omp_init( void ) __attribute__( ( constructor ) );
void
dgemm_omp_init( void )
{
    fct_dgemm_omp.mpi    = 0;
    fct_dgemm_omp.tiled  = 0;
    fct_dgemm_omp.starpu = 0;
    fct_dgemm_omp.name   = "omp";
    fct_dgemm_omp.helper = "Blocked OpenMP DGEMM (dgemm_vecto in each block)";
    fct_dgemm_omp.fctptr = dgemm_omp;
    fct_dgemm_omp.next   = NULL;

    register_fct( &fct_dgemm_omp, ALGO_GEMM );

    dgemm_omp_block_size = myblas_getenv_value_int( "GEMM_BLOCKSIZE", 48 );
}
