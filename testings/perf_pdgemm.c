#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "../myblas/myblas.h"
#include "flops.h"

int main( int argc, char **argv )
{
    option_t *options = &global_options;
    int rank, M, N, K, b, it;
    int lda, ldb, ldc;
    double *A = NULL, *B = NULL, *C = NULL, *Cinit = NULL, *Cref = NULL;
    double alpha, beta, flops;

    algonum_init( argc, argv, options, ALGO_GEMM );

    rank = options->mpirank;
    M = options->M;  N = options->N;  K = options->K;  b = options->b;
    lda = ( M > 1 ) ? M : 1;
    ldb = ( K > 1 ) ? K : 1;
    ldc = ( M > 1 ) ? M : 1;
    flops = flops_dgemm( M, N, K );

    CORE_dplrnt( 0, 1, 1, &alpha, 1, 1, 0, 0, 342 );
    CORE_dplrnt( 0, 1, 1, &beta,  1, 1, 0, 0, 651 );

    if ( rank == 0 ) {
        A     = malloc( (size_t)lda * ( K > 1 ? K : 1 ) * sizeof(double) );
        B     = malloc( (size_t)ldb * ( N > 1 ? N : 1 ) * sizeof(double) );
        C     = malloc( (size_t)ldc * ( N > 1 ? N : 1 ) * sizeof(double) );
        Cinit = malloc( (size_t)ldc * ( N > 1 ? N : 1 ) * sizeof(double) );
        Cref  = malloc( (size_t)ldc * ( N > 1 ? N : 1 ) * sizeof(double) );

        CORE_dplrnt( 0, M, K, A,     lda, M, 0, 0, 4972  );
        CORE_dplrnt( 0, K, N, B,     ldb, K, 0, 0, 330   );
        CORE_dplrnt( 0, M, N, Cinit, ldc, M, 0, 0, 93047 );

        printf( "procs=%d (P=%d x Q=%d) M=%d N=%d K=%d b=%d\n",
                options->mpisize, options->P, options->Q, M, N, K, b );
    }

    for ( it = 0; it < options->iter; it++ ) {
        pdgemm_times_t t;
        int rc;

        if ( rank == 0 ) {
            memcpy( C, Cinit, (size_t)ldc * N * sizeof(double) );
        }

        rc = dgemm_dist_compute_centralize( M, N, K, b, alpha, A, lda, B, ldb,
                                            beta, C, ldc, &t );
        if ( rc != ALGONUM_SUCCESS ) {
            if ( rank == 0 ) printf( "Version non implementee pour ces options\n" );
            break;
        }

        if ( rank == 0 ) {
            double total   = t.t_dist + t.t_comp + t.t_cent;
            double g_comp  = ( t.t_comp > 0. ) ? flops / t.t_comp / 1.e9 : 0.;
            double g_total = ( total    > 0. ) ? flops / total    / 1.e9 : 0.;
            const char *verdict = "";

            if ( options->check ) {
                memcpy( Cref, Cinit, (size_t)ldc * N * sizeof(double) );
                rc = check_dgemm( CblasNoTrans, CblasNoTrans, M, N, K,
                                  alpha, A, lda, B, ldb, beta, Cref, C, ldc );
                verdict = ( rc == ALGONUM_SUCCESS ) ? "SUCCESS" : "FAIL";
            }

            printf( "dist %.4f s | calcul %.4f s | recentralisation %.4f s | "
                    "calcul seul %.2f GFlop/s | total %.2f GFlop/s %s\n",
                    t.t_dist, t.t_comp, t.t_cent, g_comp, g_total, verdict );
            printf( "CSV,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.4f,%.4f\n",
                    options->mpisize, options->P, M, N, K, b,
                    t.t_dist, t.t_comp, t.t_cent, g_comp, g_total );
        }
    }

    free( A ); free( B ); free( C ); free( Cinit ); free( Cref );
    algonum_exit( options, ALGO_GEMM );
    return 0;
}
