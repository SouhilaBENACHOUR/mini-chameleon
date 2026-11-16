#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "../myblas/myblas.h"

int main( int argc, char **argv )
{
    option_t *o = &global_options;
    int rank, M, N, b, i, j, lda, ok = 1;
    double *A = NULL, *R = NULL, **T;

    algonum_init( argc, argv, o, ALGO_GEMM );
    rank = o->mpirank;  M = o->M;  N = o->N;  b = o->b;
    lda = ( M > 1 ) ? M : 1;

    if ( rank == 0 ) {
        A = malloc( (size_t)lda * N * sizeof(double) );
        R = malloc( (size_t)lda * N * sizeof(double) );
        for ( j = 0; j < N; j++ )
            for ( i = 0; i < M; i++ )
                A[ lda * j + i ] = i + 10. * j;
    }

    T = distribute( M, N, b, A, lda );

    if ( rank == 0 ) printf( "--- matrice affichee par affiche_dist ---\n" );
    affiche_dist( M, N, b, (const double **)T, stdout );

    tile2lapack( M, N, b, (const double **)T, R, lda );
    if ( rank == 0 ) {
        ok = ( memcmp( A, R, (size_t)lda * N * sizeof(double) ) == 0 );
        printf( "aller-retour distribution/centralisation : %s\n", ok ? "SUCCESS" : "FAIL" );
        free( A ); free( R );
    }

    tileFree( M, N, b, T );
    algonum_exit( o, ALGO_GEMM );
    return 0;
}
