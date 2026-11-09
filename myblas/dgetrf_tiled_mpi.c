/**
 *
 * @file dgetrf_tiled_mpi.c
 *
 * @copyright 2019-2021 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
 *                      Univ. Bordeaux. All rights reserved.
 *
 * @brief OpenMP tile-based implementation of the dgetrf.
 *
 * @version 0.2.0
 * @author Mathieu Faverge
 * @date 2021-09-21
 *
 */
#include "myblas.h"
#include <stdio.h>
#include <stdlib.h>

int
dgetrf_tiled_mpi( CBLAS_LAYOUT layout,
                  int M, int N, int b, double **A )
{
#if !defined(ENABLE_MPI)
    return ALGONUM_NOT_IMPLEMENTED;
#else
    MPI_Status status;
    double Akk[b*b], Amk[b*b], Akn[b*b];
    double *Akkptr, *Amkptr, *Aknptr;

    /* Chronometres (secondes) : une case par operation */
    double t0;
    double t_getrf = 0., t_bcast_kk = 0., t_trsm = 0.;
    double t_bcast_m = 0., t_recv = 0., t_send = 0., t_gemm = 0.;

    /* Let's compute the total number of tiles with a *ceil* */
    int MT = my_iceil( M, b );
    int NT = my_iceil( N, b );
    int KT = my_imin( MT, NT );
    int m, n, k;

    for( k=0; k<KT; k++) {
        int mk = k == (MT-1) ? M - k * b : b;
        int nk = k == (NT-1) ? N - k * b : b;
        int kk = my_imin( mk, nk );

        if ( get_rank_of( k, k ) == global_options.mpirank ) {
            Akkptr = A[ MT * k + k ];
            t0 = MPI_Wtime();
            dgetrf_seq( LAPACK_COL_MAJOR, mk, nk, Akkptr, b );
            t_getrf += MPI_Wtime() - t0;
        }
        else {
            Akkptr = Akk;
        }

        t0 = MPI_Wtime();
        MPI_Bcast( Akkptr, b*b, MPI_DOUBLE, get_rank_of( k, k ), MPI_COMM_WORLD );
        t_bcast_kk += MPI_Wtime() - t0;

        for( n=k+1; n<NT; n++) {
            int nn = n == (NT-1) ? N - n * b : b;

            if ( get_rank_of( k, n ) == global_options.mpirank ) {
                t0 = MPI_Wtime();
                cblas_dtrsm( CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                             kk, nn,
                             1., Akkptr,          b,
                                 A[ MT * n + k ], b );
                t_trsm += MPI_Wtime() - t0;
            }
        }
        for( m=k+1; m<MT; m++) {
            int mm = m == (MT-1) ? M - m * b : b;

            if ( get_rank_of( m, k ) == global_options.mpirank ) {
                Amkptr = A[ MT * k + m ];
                t0 = MPI_Wtime();
                cblas_dtrsm( CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
                             mm, kk,
                             1., Akkptr, b,
                                 Amkptr, b );
                t_trsm += MPI_Wtime() - t0;
            }
            else {
                Amkptr = Amk;
            }
            t0 = MPI_Wtime();
            MPI_Bcast( Amkptr, b*b, MPI_DOUBLE, get_rank_of( m, k ), MPI_COMM_WORLD );
            t_bcast_m += MPI_Wtime() - t0;

            for( n=k+1; n<NT; n++) {
                int nn = n == (NT-1) ? N - n * b : b;

                Aknptr = A[ MT * n + k ];
                if ( get_rank_of( m, n ) == global_options.mpirank ) {

                    if ( get_rank_of( k, n ) != global_options.mpirank ) {
                        Aknptr = Akn;
                        t0 = MPI_Wtime();
                        MPI_Recv( Aknptr, b*b, MPI_DOUBLE,
                                  get_rank_of( k, n ), MT * n + k, MPI_COMM_WORLD, &status );
                        t_recv += MPI_Wtime() - t0;
                    }

                    t0 = MPI_Wtime();
                    dgemm_seq( CblasColMajor, CblasNoTrans, CblasNoTrans,
                               mm, nn, kk,
                               -1., Amkptr, b,
                                    Aknptr, b,
                               1.,  A[ MT * n + m ], b );
                    t_gemm += MPI_Wtime() - t0;
                }
                else {
                    if ( get_rank_of( k, n ) == global_options.mpirank ) {
                        t0 = MPI_Wtime();
                        MPI_Send( Aknptr, b*b, MPI_DOUBLE,
                                  get_rank_of( m, n ), MT * n + k, MPI_COMM_WORLD );
                        t_send += MPI_Wtime() - t0;
                    }
                }
            }
        }
    }

    /* Affichage du profil si la variable d'environnement MPI_PROFILE est definie */
    if ( getenv( "MPI_PROFILE" ) != NULL ) {
        double loc[7] = { t_getrf, t_bcast_kk, t_trsm, t_bcast_m, t_recv, t_send, t_gemm };
        double mx[7], sm[7];
        MPI_Reduce( loc, mx, 7, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD );
        MPI_Reduce( loc, sm, 7, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );
        if ( global_options.mpirank == 0 ) {
            double sz = (double)global_options.mpisize;
            printf( "[profil] max sur les rangs (s) : getrf=%.3f bcast_kk=%.3f trsm=%.3f "
                    "bcast_m=%.3f recv=%.3f send=%.3f gemm=%.3f\n",
                    mx[0], mx[1], mx[2], mx[3], mx[4], mx[5], mx[6] );
            printf( "[profil] moyenne sur les rangs (s) : getrf=%.3f bcast_kk=%.3f trsm=%.3f "
                    "bcast_m=%.3f recv=%.3f send=%.3f gemm=%.3f\n",
                    sm[0]/sz, sm[1]/sz, sm[2]/sz, sm[3]/sz, sm[4]/sz, sm[5]/sz, sm[6]/sz );
        }
    }

    return ALGONUM_SUCCESS; /* Success */
#endif
}

/* To make sure we use the right prototype */
static dgetrf_tiled_fct_t valid_dgetrf_tiled_mpi __attribute__ ((unused)) = dgetrf_tiled_mpi;

/* Declare the variable that will store the information about this version */
fct_list_t fct_dgetrf_tiled_mpi;

/**
 * @brief Registration function
 */
void dgetrf_tiled_mpi_init( void ) __attribute__( ( constructor ) );
void
dgetrf_tiled_mpi_init( void )
{
    fct_dgetrf_tiled_mpi.mpi    = 1;
    fct_dgetrf_tiled_mpi.tiled  = 1;
    fct_dgetrf_tiled_mpi.starpu = 0;
    fct_dgetrf_tiled_mpi.name   = "mpi";
    fct_dgetrf_tiled_mpi.helper = "MPI tile-based implementation of the dgetrf";
    fct_dgetrf_tiled_mpi.fctptr = dgetrf_tiled_mpi;
    fct_dgetrf_tiled_mpi.next   = NULL;

    register_fct( &fct_dgetrf_tiled_mpi, ALGO_GETRF );
}
