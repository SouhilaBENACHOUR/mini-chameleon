/* usage: test_ld M N K lda ldb ldc  -> median GFlop/s of dgemm_seq (column-major, NoTrans) */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* CblasColMajor == 102, CblasNoTrans == 111 */
int dgemm_seq( int layout, int ta, int tb, int M, int N, int K, double alpha,
               const double *A, int lda, const double *B, int ldb,
               double beta, double *C, int ldc );

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

static double *alloc64( size_t n )
{
    void *p = NULL;
    if ( posix_memalign( &p, 64, n * sizeof(double) ) ) exit( 1 );
    return p;
}

int main( int argc, char **argv )
{
    if ( argc < 7 ) { fprintf( stderr, "usage: %s M N K lda ldb ldc\n", argv[0] ); return 1; }
    int M = atoi( argv[1] ), N = atoi( argv[2] ), K = atoi( argv[3] );
    int lda = atoi( argv[4] ), ldb = atoi( argv[5] ), ldc = atoi( argv[6] );
    double *A = alloc64( (size_t)lda * K );
    double *B = alloc64( (size_t)ldb * N );
    double *C = alloc64( (size_t)ldc * N );
    int reps = 11, i, r;
    double t[11];
    double flops = 2. * M * N * (double)K;

    for ( i = 0; i < lda * K; i++ ) A[i] = 1. + (i % 7) * 0.01;
    for ( i = 0; i < ldb * N; i++ ) B[i] = 1. - (i % 5) * 0.01;
    for ( i = 0; i < ldc * N; i++ ) C[i] = 0.;

    dgemm_seq( 102, 111, 111, M, N, K, 0.5 / K, A, lda, B, ldb, 1., C, ldc ); /* warm-up */
    for ( r = 0; r < reps; r++ ) {
        double t0 = now();
        dgemm_seq( 102, 111, 111, M, N, K, 0.5 / K, A, lda, B, ldb, 1., C, ldc );
        t[r] = now() - t0;
    }
    qsort( t, reps, sizeof(double), cmp );
    printf( "M=%d N=%d K=%d lda=%d ldb=%d ldc=%d : %.2f GFlop/s\n",
            M, N, K, lda, ldb, ldc, flops / t[reps / 2] / 1e9 );
    return 0;
}
