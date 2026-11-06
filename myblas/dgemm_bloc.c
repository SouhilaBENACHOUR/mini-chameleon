/**
 * @file dgemm_bloc.c
 *
 * @brief Blocked version of the Matrix-Matrix multiply operation.
 *        C is split in blocks nb x nb ; each block product is done by dgemm_scalaire.
 */
#include "myblas.h"

/* Taille de bloc, lue dans la variable d'environnement GEMM_BLOCKSIZE */
static int dgemm_bloc_block_size = 128;

int dgemm_bloc( CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA,
                CBLAS_TRANSPOSE transB, const int M, const int N,
                const int K, const double alpha, const double *A,
                const int lda, const double *B, const int ldb,
                const double beta, double *C, const int ldc )
{
    int nb = dgemm_bloc_block_size;
    int ii, jj, kk;

    if ( nb < 1 ) nb = 32; /* garde-fou */

    for ( jj = 0; jj < N; jj += nb ) {
        int nn = ( N - jj < nb ) ? N - jj : nb;

        for ( ii = 0; ii < M; ii += nb ) {
            int mm = ( M - ii < nb ) ? M - ii : nb;

            for ( kk = 0; kk < K; kk += nb ) {
                int          kb = ( K - kk < nb ) ? K - kk : nb;
                /* beta n'est appliqué qu'une fois, au premier bloc de k */
                double       bt = ( kk == 0 ) ? beta : 1.;

                /* début du bloc de A (mm x kb) et de B (kb x nn) selon la transposition */
                const double *Ab = ( transA == CblasNoTrans ) ? A + lda * kk + ii
                                                              : A + lda * ii + kk;
                const double *Bb = ( transB == CblasNoTrans ) ? B + ldb * jj + kk
                                                              : B + ldb * kk + jj;
                double       *Cb = C + ldc * jj + ii;

                dgemm_vecto( layout, transA, transB, mm, nn, kb,
                                alpha, Ab, lda, Bb, ldb, bt, Cb, ldc );
            }
        }
    }

    /* Cas K = 0 : le résultat est simplement beta * C */
    if ( K == 0 ) {
        int m, n;
        for ( n = 0; n < N; n++ )
            for ( m = 0; m < M; m++ )
                C[ ldc * n + m ] *= beta;
    }

    return ALGONUM_SUCCESS;
}

/* To make sure we use the right prototype */
static dgemm_fct_t valid_dgemm_bloc __attribute__ ((unused)) = dgemm_bloc;

/* Declare the variable that will store the information about this version */
fct_list_t fct_dgemm_bloc;

/**
 * @brief Registration function
 */
void dgemm_bloc_init( void ) __attribute__( ( constructor ) );
void
dgemm_bloc_init( void )
{
    fct_dgemm_bloc.mpi    = 0;
    fct_dgemm_bloc.tiled  = 0;
    fct_dgemm_bloc.starpu = 0;
    fct_dgemm_bloc.name   = "bloc";
    fct_dgemm_bloc.helper = "Blocked version of DGEMM";
    fct_dgemm_bloc.fctptr = dgemm_bloc;
    fct_dgemm_bloc.next   = NULL;

    register_fct( &fct_dgemm_bloc, ALGO_GEMM );

    dgemm_bloc_block_size = myblas_getenv_value_int( "GEMM_BLOCKSIZE", 128 );
}
