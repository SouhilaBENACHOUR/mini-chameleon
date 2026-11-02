#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Function to measure (defined in myblas/ddot_seq.c) */
double ddot_seq( int N, const double *X, int incX,
                 const double *Y, int incY );

static double now( void )
{
    struct timespec t;
    clock_gettime( CLOCK_MONOTONIC, &t );
    return t.tv_sec + 1.e-9 * t.tv_nsec;
}

int main( void )
{
    int    N;
    double sink = 0.;   /* printed at the end so the calls are not optimised away */

    for ( N = 50; N <= 1000000; N = N * 125 / 100 ) {
        double *X = malloc( N * sizeof(double) );
        double *Y = malloc( N * sizeof(double) );
        long    R, r;
        int     i;

        for ( i = 0; i < N; i++ ) {
            X[i] = 1. + (i % 7) * 0.01;
            Y[i] = 1. - (i % 5) * 0.01;
        }

        /* about 1e8 elements processed per point */
        R = 100000000L / N;
        if ( R < 1 ) R = 1;

        sink += ddot_seq( N, X, 1, Y, 1 );   /* warm-up, not timed */

        double t0 = now();
        for ( r = 0; r < R; r++ ) {
            sink += ddot_seq( N, X, 1, Y, 1 );
        }
        double t1 = now();

        double gflops = (2.0 * N * R) / (t1 - t0) / 1e9;
        printf( "%d %e\n", N, gflops );

        free( X );
        free( Y );
    }
    fprintf( stderr, "sink = %g\n", sink );
    return 0;
}
