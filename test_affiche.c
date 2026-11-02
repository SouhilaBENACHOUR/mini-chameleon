#include <stdio.h>

void affiche( int m, int n, const double *a, int lda, FILE *flux );

int main( void )
{
    double a[8] = {1, 2, 3, 0, 4, 5, 6, 0};   /* lda = 4, m = 3, n = 2 */
    affiche( 3, 2, a, 4, stdout );
    return 0;
}
