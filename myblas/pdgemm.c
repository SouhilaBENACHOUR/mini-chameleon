#include "myblas.h"
#include <stdio.h>
#include <stdlib.h>

void affiche( int m, int n, const double *a, int lda, FILE *flux );

void
affiche_dist( int M, int N, int b, const double **Atile, FILE *flux )
{
    double *A   = NULL;
    int     lda = my_imax( M, 1 );

    if ( global_options.mpirank == 0 ) {
        A = malloc( (size_t)lda * my_imax( N, 1 ) * sizeof(double) );
    }

    /* tous les processus participent : les tuiles arrivent sur le 0 */
    tile2lapack( M, N, b, Atile, A, lda );

    if ( global_options.mpirank == 0 ) {
        affiche( M, N, A, lda, flux );
        free( A );
    }
}

double **
distribute( int M, int N, int b, const double *Alapack, int lda )
{
    int MT = my_iceil( M, b );
    int NT = my_iceil( N, b );
    int rank = global_options.mpirank;
    int m, n;

    double **Atile = malloc( (size_t)MT * NT * sizeof(double*) );
    double  *buf   = NULL;

    if ( rank == 0 ) {
        buf = malloc( (size_t)b * b * sizeof(double) );
    }

    for ( n = 0; n < NT; n++ ) {
        for ( m = 0; m < MT; m++ ) {
            int tag   = MT * n + m;
            int owner = get_rank_of( m, n );
            int mm    = ( m == (MT-1) ) ? M - m * b : b;
            int nn    = ( n == (NT-1) ) ? N - n * b : b;

            Atile[tag] = NULL;

            if ( rank == owner ) {
                Atile[tag] = calloc( (size_t)b * b, sizeof(double) );

                if ( rank == 0 ) {
                    LAPACKE_dlacpy_work( LAPACK_COL_MAJOR, 'A', mm, nn,
                                         Alapack + (size_t)lda * b * n + b * m, lda,
                                         Atile[tag], b );
                }
#if defined(ENABLE_MPI)
                else {
                    MPI_Status status;
                    MPI_Recv( Atile[tag], b * b, MPI_DOUBLE,
                              0, tag, MPI_COMM_WORLD, &status );
                }
#endif
            }
#if defined(ENABLE_MPI)
            else if ( rank == 0 ) {
                LAPACKE_dlacpy_work( LAPACK_COL_MAJOR, 'A', mm, nn,
                                     Alapack + (size_t)lda * b * n + b * m, lda,
                                     buf, b );
                MPI_Send( buf, b * b, MPI_DOUBLE, owner, tag, MPI_COMM_WORLD );
            }
#endif
        }
    }

    free( buf );
    return Atile;
}

int
my_pdgemm( CBLAS_LAYOUT layout,
           CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
           int M, int N, int K, int b,
           double alpha, const double **A,
                         const double **B,
           double beta,        double **C )
{
#if !defined(ENABLE_MPI)
    (void)layout; (void)transA; (void)transB; (void)M; (void)N; (void)K;
    (void)b; (void)alpha; (void)A; (void)B; (void)beta; (void)C;
    return ALGONUM_NOT_IMPLEMENTED;
#else
    int P = global_options.P;
    int Q = global_options.Q;
    int rank = global_options.mpirank;
    int p = rank / Q;
    int q = rank % Q;

    int MT = my_iceil( M, b );
    int NT = my_iceil( N, b );
    int KT = my_iceil( K, b );
    int nlm = ( MT > p ) ? ( MT - p + P - 1 ) / P : 0;
    int nln = ( NT > q ) ? ( NT - q + Q - 1 ) / Q : 0;

    int i, j, k, m, n, kk, mm, nn;
    double lbeta;
    MPI_Comm rowcomm, colcomm;
    double  *wsA, *wsB;
    double **Aptr, **Bptr;

    (void)layout;
    if ( (transA != CblasNoTrans) || (transB != CblasNoTrans) ) {
        return ALGONUM_NOT_IMPLEMENTED;
    }

    if ( KT == 0 ) {
        for ( n = q; n < NT; n += Q ) {
            nn = ( n == (NT-1) ) ? N - n * b : b;
            for ( m = p; m < MT; m += P ) {
                mm = ( m == (MT-1) ) ? M - m * b : b;
                for ( j = 0; j < nn; j++ ) {
                    for ( i = 0; i < mm; i++ ) {
                        C[ MT * n + m ][ b * j + i ] *= beta;
                    }
                }
            }
        }
        return ALGONUM_SUCCESS;
    }

    MPI_Comm_split( MPI_COMM_WORLD, p, q, &rowcomm );
    MPI_Comm_split( MPI_COMM_WORLD, q, p, &colcomm );

    wsA  = malloc( (size_t)( nlm + 1 ) * b * b * sizeof(double) );
    wsB  = malloc( (size_t)( nln + 1 ) * b * b * sizeof(double) );
    Aptr = malloc( (size_t)( nlm + 1 ) * sizeof(double*) );
    Bptr = malloc( (size_t)( nln + 1 ) * sizeof(double*) );

    for ( k = 0; k < KT; k++ ) {
        kk    = ( k == (KT-1) ) ? K - k * b : b;
        lbeta = ( k == 0 ) ? beta : 1.;

        for ( i = 0; i < nlm; i++ ) {
            m = p + i * P;
            Aptr[i] = ( q == k % Q ) ? (double*)A[ MT * k + m ] : wsA + (size_t)i * b * b;
            MPI_Bcast( Aptr[i], b * b, MPI_DOUBLE, k % Q, rowcomm );
        }

        for ( j = 0; j < nln; j++ ) {
            n = q + j * Q;
            Bptr[j] = ( p == k % P ) ? (double*)B[ KT * n + k ] : wsB + (size_t)j * b * b;
            MPI_Bcast( Bptr[j], b * b, MPI_DOUBLE, k % P, colcomm );
        }

        for ( j = 0; j < nln; j++ ) {
            n  = q + j * Q;
            nn = ( n == (NT-1) ) ? N - n * b : b;
            for ( i = 0; i < nlm; i++ ) {
                m  = p + i * P;
                mm = ( m == (MT-1) ) ? M - m * b : b;

                dgemm_seq( CblasColMajor, CblasNoTrans, CblasNoTrans,
                           mm, nn, kk,
                           alpha, Aptr[i], b, Bptr[j], b,
                           lbeta, C[ MT * n + m ], b );
            }
        }
    }

    free( wsA ); free( wsB ); free( Aptr ); free( Bptr );
    MPI_Comm_free( &rowcomm );
    MPI_Comm_free( &colcomm );

    return ALGONUM_SUCCESS;
#endif
}

static double
elapsed( const perf_t *begin, const perf_t *end )
{
    perf_t d = *end;
    perf_diff( begin, &d );
    return d.tv_sec + 1.e-6 * d.tv_usec;
}

int
dgemm_dist_compute_centralize( int M, int N, int K, int b,
                               double alpha, const double *A, int lda,
                                             const double *B, int ldb,
                               double beta,        double *C, int ldc,
                               pdgemm_times_t *times )
{
    perf_t t0, t1, t2, t3;
    double **Atile, **Btile, **Ctile;
    int rc;

    perf( &t0 );
    Atile = distribute( M, K, b, A, lda );
    Btile = distribute( K, N, b, B, ldb );
    Ctile = distribute( M, N, b, C, ldc );

    perf( &t1 );
    rc = my_pdgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, b,
                    alpha, (const double **)Atile, (const double **)Btile,
                    beta, Ctile );

    perf( &t2 );
    tile2lapack( M, N, b, (const double **)Ctile, C, ldc );
    perf( &t3 );

    tileFree( M, K, b, Atile );
    tileFree( K, N, b, Btile );
    tileFree( M, N, b, Ctile );

    if ( times != NULL ) {
        times->t_dist = elapsed( &t0, &t1 );
        times->t_comp = elapsed( &t1, &t2 );
        times->t_cent = elapsed( &t2, &t3 );
    }
    return rc;
}
