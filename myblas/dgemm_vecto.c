/**
 * @file dgemm_vecto.c
 * @brief DGEMM sequentiel vectorise (AVX2 + FMA) - tache 2B
 *        v2 : micro-noyau 8x4, les 8 sommes partielles de C restent
 *        dans des registres pendant toute la boucle sur k.
 *        Bords et cas transposes : versions scalaires simples.
 */
#include "myblas.h"
#include <immintrin.h>

/* ---------- Cas transposes : version scalaire ---------- */
static void
dgemm_vecto_trans( CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                   int M, int N, int K, double alpha,
                   const double *A, int lda, const double *B, int ldb,
                   double beta, double *C, int ldc )
{
    for ( int n = 0; n < N; n++ ) {
        for ( int m = 0; m < M; m++ ) {
            double sum = 0.;
            for ( int k = 0; k < K; k++ ) {
                double a = ( transA == CblasNoTrans ) ? A[ (size_t)lda * k + m ] : A[ (size_t)lda * m + k ];
                double b = ( transB == CblasNoTrans ) ? B[ (size_t)ldb * n + k ] : B[ (size_t)ldb * k + n ];
                sum += a * b;
            }
            C[ (size_t)ldc * n + m ] = beta * C[ (size_t)ldc * n + m ] + alpha * sum;
        }
    }
}

/* ---------- Bords (NoTrans/NoTrans) : version scalaire ----------
 * Traite les lignes [m_deb, m_fin) et les colonnes [n_deb, n_fin) de C. */
static void
dgemm_vecto_bord( int m_deb, int m_fin, int n_deb, int n_fin, int K,
                  double alpha, const double *A, int lda,
                  const double *B, int ldb,
                  double beta, double *C, int ldc )
{
    for ( int n = n_deb; n < n_fin; n++ ) {
        for ( int m = m_deb; m < m_fin; m++ ) {
            double sum = 0.;
            for ( int k = 0; k < K; k++ ) {
                sum += A[ (size_t)lda * k + m ] * B[ (size_t)ldb * n + k ];
            }
            C[ (size_t)ldc * n + m ] = beta * C[ (size_t)ldc * n + m ] + alpha * sum;
        }
    }
}

/* ---------- Ecriture d'une colonne de la tuile : C = beta*C + alpha*c ---------- */
__attribute__((target("avx2,fma")))
static inline void
ecrire_colonne( double *Cj, __m256d c_haut, __m256d c_bas,
                __m256d valpha, __m256d vbeta )
{
    __m256d r0 = _mm256_mul_pd( vbeta, _mm256_loadu_pd( Cj     ) );
    __m256d r1 = _mm256_mul_pd( vbeta, _mm256_loadu_pd( Cj + 4 ) );
    r0 = _mm256_fmadd_pd( valpha, c_haut, r0 );
    r1 = _mm256_fmadd_pd( valpha, c_bas,  r1 );
    _mm256_storeu_pd( Cj,     r0 );
    _mm256_storeu_pd( Cj + 4, r1 );
}

/* ---------- Micro-noyau 8x4 ----------
 * A pointe sur A(m0, 0), B sur B(0, n0), C sur C(m0, n0).
 * Calcule C(m0..m0+7, n0..n0+3) = beta*C + alpha * A * B. */
__attribute__((target("avx2,fma")))
static void
micro_noyau_8x4( int K, double alpha, const double *A, int lda,
                 const double *B, int ldb, double beta, double *C, int ldc )
{
    /* 8 sommes partielles : ch = lignes 0-3, cb = lignes 4-7, chiffre = colonne */
    __m256d ch0 = _mm256_setzero_pd(), cb0 = _mm256_setzero_pd();
    __m256d ch1 = _mm256_setzero_pd(), cb1 = _mm256_setzero_pd();
    __m256d ch2 = _mm256_setzero_pd(), cb2 = _mm256_setzero_pd();
    __m256d ch3 = _mm256_setzero_pd(), cb3 = _mm256_setzero_pd();

    const double *B0 = B;
    const double *B1 = B +     (size_t)ldb;
    const double *B2 = B + 2 * (size_t)ldb;
    const double *B3 = B + 3 * (size_t)ldb;

    for ( int k = 0; k < K; k++ ) {
        const double *Ak = A + (size_t)lda * k;
        __m256d ah = _mm256_loadu_pd( Ak     );   /* A(m0..m0+3, k)   */
        __m256d ab = _mm256_loadu_pd( Ak + 4 );   /* A(m0+4..m0+7, k) */
        __m256d b;

        b   = _mm256_broadcast_sd( B0 + k );      /* B(k, n0)   x4 */
        ch0 = _mm256_fmadd_pd( ah, b, ch0 );
        cb0 = _mm256_fmadd_pd( ab, b, cb0 );

        b   = _mm256_broadcast_sd( B1 + k );      /* B(k, n0+1) x4 */
        ch1 = _mm256_fmadd_pd( ah, b, ch1 );
        cb1 = _mm256_fmadd_pd( ab, b, cb1 );

        b   = _mm256_broadcast_sd( B2 + k );      /* B(k, n0+2) x4 */
        ch2 = _mm256_fmadd_pd( ah, b, ch2 );
        cb2 = _mm256_fmadd_pd( ab, b, cb2 );

        b   = _mm256_broadcast_sd( B3 + k );      /* B(k, n0+3) x4 */
        ch3 = _mm256_fmadd_pd( ah, b, ch3 );
        cb3 = _mm256_fmadd_pd( ab, b, cb3 );
    }

    /* Une seule ecriture de C, avec alpha et beta */
    __m256d valpha = _mm256_set1_pd( alpha );
    __m256d vbeta  = _mm256_set1_pd( beta );
    ecrire_colonne( C,                     ch0, cb0, valpha, vbeta );
    ecrire_colonne( C +     (size_t)ldc,   ch1, cb1, valpha, vbeta );
    ecrire_colonne( C + 2 * (size_t)ldc,   ch2, cb2, valpha, vbeta );
    ecrire_colonne( C + 3 * (size_t)ldc,   ch3, cb3, valpha, vbeta );
}

/* ---------- Fonction principale (interface cblas) ---------- */
__attribute__((target("avx2,fma")))
int dgemm_vecto( CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA,
                 CBLAS_TRANSPOSE transB, const int M, const int N,
                 const int K, const double alpha, const double *A,
                 const int lda, const double *B, const int ldb,
                 const double beta, double *C, const int ldc )
{
    if ( layout != CblasColMajor ) {
        return ALGONUM_NOT_IMPLEMENTED;
    }

    if ( transA != CblasNoTrans || transB != CblasNoTrans ) {
        dgemm_vecto_trans( transA, transB, M, N, K, alpha,
                           A, lda, B, ldb, beta, C, ldc );
        return ALGONUM_SUCCESS;
    }

    int M8 = M - M % 8;   /* lignes couvertes par des tuiles completes   */
    int N4 = N - N % 4;   /* colonnes couvertes par des tuiles completes */

    /* Tuiles 8x4 : micro-noyau */
    for ( int n0 = 0; n0 < N4; n0 += 4 ) {
        for ( int m0 = 0; m0 < M8; m0 += 8 ) {
            micro_noyau_8x4( K, alpha,
                             A + m0, lda,
                             B + (size_t)ldb * n0, ldb,
                             beta,
                             C + m0 + (size_t)ldc * n0, ldc );
        }
    }

    /* Bords : lignes restantes (pour les colonnes deja traitees),
     * puis colonnes restantes (pour toutes les lignes) */
    if ( M8 < M ) {
        dgemm_vecto_bord( M8, M, 0, N4, K, alpha, A, lda, B, ldb, beta, C, ldc );
    }
    if ( N4 < N ) {
        dgemm_vecto_bord( 0, M, N4, N, K, alpha, A, lda, B, ldb, beta, C, ldc );
    }

    return ALGONUM_SUCCESS;
}

/* Verifie que le prototype est le bon */
static dgemm_fct_t valid_dgemm_vecto __attribute__ ((unused)) = dgemm_vecto;

fct_list_t fct_dgemm_vecto;

void dgemm_vecto_init( void ) __attribute__( ( constructor ) );
void
dgemm_vecto_init( void )
{
    fct_dgemm_vecto.mpi    = 0;
    fct_dgemm_vecto.tiled  = 0;
    fct_dgemm_vecto.starpu = 0;
    fct_dgemm_vecto.name   = "vecto";
    fct_dgemm_vecto.helper = "Sequential GEMM, AVX2/FMA 8x4 micro-kernel";
    fct_dgemm_vecto.fctptr = dgemm_vecto;
    fct_dgemm_vecto.next   = NULL;

    register_fct( &fct_dgemm_vecto, ALGO_GEMM );
}
