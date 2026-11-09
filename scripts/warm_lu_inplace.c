#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* CBLAS_LAYOUT is an int-sized enum; CblasColMajor == 102 */
int dgetrf_seq( int layout, int M, int N, double *A, int lda );

static double now( void )
{
    struct timespec t;
    clock_gettime( CLOCK_MONOTONIC, &t );
    return t.tv_sec + 1.e-9 * t.tv_nsec;
}

int main( int argc, char **argv )
{
    int     n = (argc > 1) ? atoi( argv[1] ) : 500;
    int     it, i;
    double *A = malloc( (size_t)n * n * sizeof(double) );
    double  flops = 2. / 3. * n * n * (double)n;
    double  total = 0.;

    /* diagonally dominant random matrix: no pivoting needed */
    srand( 1 );
    for ( i = 0; i < n * n; i++ ) A[i] = rand() / (double)RAND_MAX;
    for ( i = 0; i < n; i++ )     A[i + (size_t)i * n] += n;

    for ( it = 0; it < 240; it++ ) {
        double t0 = now();
        dgetrf_seq( 102, n, n, A, n );
        double t1 = now();
        total += t1 - t0;

        if ( it < 2 || (it >= 100 && it <= 125) || it == 239 ) {
            long nbad = 0, nsub = 0, nzero = 0;
            for ( i = 0; i < n * n; i++ ) {
                int c = fpclassify( A[i] );
                if ( c == FP_NAN || c == FP_INFINITE ) nbad++;
                else if ( c == FP_SUBNORMAL )          nsub++;
                else if ( c == FP_ZERO )               nzero++;
            }
            printf( "it=%3d  %7.3f GFlop/s  nan/inf=%ld  subnormal=%ld  zero=%ld\n",
                    it, flops / (t1 - t0) / 1e9, nbad, nsub, nzero );
        }
    }
    printf( "average over 240 iterations: %.3f GFlop/s\n", 240 * flops / total / 1e9 );
    return 0;
}
