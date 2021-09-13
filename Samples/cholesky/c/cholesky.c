#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N 8192

void cholesky(double *A, double *L, int n) {
    for (int j = 0; j < n; j++) {
        double s1 = 0.0;
        for (int k = 0; k < j; k++) {
            s1 += L[j * n + k] * L[j * n + k];
        }
        L[j * n + j] = sqrt(A[j * n + j] - s1);

        #pragma omp parallel for
        for (int i = j + 1; i < n; i++) {
            double s2 = 0;
            for (int k = 0; k < j; k++) {
                s2 += L[i * n + k] * L[j * n + k];
            }
            L[i * n + j] = (1.0 / L[j * n + j] * (A[i * n + j] - s2));
        }
    }
}


void show_matrix(double *A, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            printf("%2.5f ", A[i * n + j]);
        printf("\n");
    }
}

void init_array(double * A, int n) {
    double *b = (double *)malloc(n * n * sizeof(double));

    #pragma omp parallel for
    for(int i = 0; i < n * n; i++)
        A[i] = 0;

    srand(42);
    for(int i = 0; i < n * n; i++)
        b[i] = rand() % 100;

    #pragma omp parallel for
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            for(int k = 0; k < n; k++)
                A[i * n + j] += b[i * n + k] * b[j * n + k];

    free(b);
}

int main() {
    double* A = (double *) malloc(N * N * sizeof(double));
    double* L = (double *) calloc(N * N, sizeof(double));
    if (A == NULL || L == NULL) {
        printf("Error allocating memory\n");
        return -1;
    }

    init_array(A, N);
    // show_matrix(A, N);
    // printf("\n");

    double start = omp_get_wtime();
    cholesky(A, L, N);
    double time = omp_get_wtime() - start;

    // show_matrix(L, N);
    // printf("\n");
    printf("Computation took: %f\n", time);
    free(A);
    free(L);

    return 0;
}