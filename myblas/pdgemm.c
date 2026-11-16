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
