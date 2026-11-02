/**
 *
 * @file dgetrf_seq.c
 *
 * @copyright 2019-2021 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief Basic sequential implementation of the dgetrf.
 *
 * @version 0.2.0
 * @author Mathieu Faverge
 * @date 2021-09-21
 *
 */
#include "myblas.h"
#include <assert.h>

int
dgetf2( CBLAS_LAYOUT layout, int M, int N, double *A, int lda )
{
    int m, n, k;
    int K = ( M > N ) ? N : M;
       
     (void)layout; /*column-major*/

    for( k=0; k<K; k++ ) {
        for( m=k+1; m<M; m++ ) {
            A[ lda * k + m ] = A[ lda * k + m ] / A[ lda * k + k ];
            for( n=k+1; n<N; n++ ) {
                A[ lda * n + m ] = A[ lda * n + m ] - A[ lda * k + m ] * A[ lda * n + k ];
            }
        }
    }

    return ALGONUM_SUCCESS; /* Success */
}

int dgetrf_seq(CBLAS_LAYOUT  layout, int M, int N, double *A, int  lda ){
   return dgetrf_bloc( layout, M , N, A, lda);
}
/* To make sure we use the right prototype */
static dgetrf_fct_t valid_dgetrf_seq __attribute__ ((unused)) = dgetrf_seq;

/* Declare the variable that will store the information about this version */
fct_list_t fct_dgetrf_seq;

/**
 * @brief Registration function
 */
void dgetrf_seq_init( void ) __attribute__( ( constructor ) );
void
dgetrf_seq_init( void )
{
    fct_dgetrf_seq.mpi    = 0;
    fct_dgetrf_seq.tiled  = 0;
    fct_dgetrf_seq.starpu = 0;
    fct_dgetrf_seq.name   = "seq";
    fct_dgetrf_seq.helper = "Basic sequential implementation of the dgetrf";
    fct_dgetrf_seq.fctptr = dgetrf_seq;
    fct_dgetrf_seq.next   = NULL;

    register_fct( &fct_dgetrf_seq, ALGO_GETRF );
}

/* Register the unblocked LU (dgetf2) so that it can be measured on its own */
static dgetrf_fct_t valid_dgetf2 __attribute__ ((unused)) = dgetf2;

fct_list_t fct_dgetrf_unblocked;

void dgetrf_unblocked_init( void ) __attribute__( ( constructor ) );
void
dgetrf_unblocked_init( void )
{
    fct_dgetrf_unblocked.mpi    = 0;
    fct_dgetrf_unblocked.tiled  = 0;
    fct_dgetrf_unblocked.starpu = 0;
    fct_dgetrf_unblocked.name   = "unblocked";
    fct_dgetrf_unblocked.helper = "Unblocked sequential LU (dgetf2)";
    fct_dgetrf_unblocked.fctptr = dgetf2;
    fct_dgetrf_unblocked.next   = NULL;

    register_fct( &fct_dgetrf_unblocked, ALGO_GETRF );
}
