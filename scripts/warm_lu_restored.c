#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* CBLAS_LAYOUT is an int-sized enum; CblasColMajor == 102 */
int dgetrf_seq( int layout, int M, int N, double *A, int lda );

static double now( void )
{
    struct timespec t;
    clock_gettime( CLOCK_MONOTONIC, &t );
    return t.tv_sec + 1.e-9 * t.tv_nsec;
}

static int cmp( const void *a, const void *b )
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main( int argc, char **argv )
{
    int     n    = (argc > 1) ? atoi( argv[1] ) : 500;
    int     reps = (argc > 2) ? atoi( argv[2] ) : 21;
    size_t  sz   = (size_t)n * n * sizeof(double);
    double *A0   = malloc( sz );
    double *A    = malloc( sz );
    double *t    = malloc( reps * sizeof(double) );
    double  flops = 2. / 3. * n * n * (double)n;
    int     i, r;

    /* diagonally dominant random matrix: no pivoting needed */
    srand( 1 );
    for ( i = 0; i < n * n; i++ ) A0[i] = rand() / (double)RAND_MAX;
    for ( i = 0; i < n; i++ )     A0[i + (size_t)i * n] += n;

    /* untimed warm-up call: caches, pages, branch predictors */
    memcpy( A, A0, sz );
    dgetrf_seq( 102, n, n, A, n );

    for ( r = 0; r < reps; r++ ) {
        memcpy( A, A0, sz );           /* restore the matrix (not timed) */
        double t0 = now();
        dgetrf_seq( 102, n, n, A, n );
        t[r] = now() - t0;
    }
    qsort( t, reps, sizeof(double), cmp );
    printf( "N=%d warm median of %d: %.3f GFlop/s\n",
            n, reps, flops / t[reps / 2] / 1e9 );
    return 0;
}
