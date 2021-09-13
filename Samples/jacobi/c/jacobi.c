#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 8192
#define ITER 50

void iterate(double* A, double* b1, double* b2, double* x_, double* x, double* y_, double* y)
{
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        double d = b1[i];
        double e = b2[i];

        for (int j = 0; j < N; j++)
        {
            if (j != i)
            {
                d = d - A[i * N + j] * x_[j];
            }
        }

        for (int j = 0; j < N; j++)
        {
            if (j != i)
            {
                e = e - A[i * N + j] * y_[j];
            }
        }

        x[i] = d / x_[i];
        y[i] = e / y_[i];
    }
}

int main(int argc, char** argv)
{
    double* A = (double *) malloc(N * N * sizeof(double));
    double* b1 = (double *) malloc(N * sizeof(double));
    double* b2 = (double *) malloc(N * sizeof(double));
    double* x = (double *) malloc(N * sizeof(double));
    double* x_ = (double *) malloc(N * sizeof(double));
    double* y = (double *) malloc(N * sizeof(double));
    double* y_ = (double *) malloc(N * sizeof(double));
    if (A == NULL || b1 == NULL || b2 == NULL || x == NULL || x_ == NULL || y == NULL || y_ == NULL)
    {
        printf("Error allocating memory\n");
        return -1;
    }

    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            if (i == j)
            {
                A[i * N + j] = 1.0;
            } else {
                A[i * N + j] = 0.0;
            }
        }
    }

    for (size_t i = 0; i < N; i++) {
        b1[i] = i + 1;
        b2[i] = i + 2;

        x_[i] = 1.0;
        y_[i] = 1.0;
    }

    double start = omp_get_wtime();

    for (int k = 0; k < ITER; k++)
    {
        if (k % 2 == 0) {
            iterate(A, b1, b2, x_, x, y_, y);

        } else {
            iterate(A, b1, b2, x, x_, y, y_);
        }
    }

    double time = omp_get_wtime() - start;

    for (int i = 0; i < N; i++)
    {
        double b = 0.0;
        double c = 0.0;
        double d = 0.0;
        for (int j = 0; j < N; j++)
        {
            b += A[i * N + j] * x[j];
            c += A[i * N + j] * y[j];
        }
        if (b != b1[i] || c != b2[i])
        {
            printf("Wrong result!\n");
            return -1;
        }
    }

    printf("Computation took: %f\n", time);

    free(A);
    free(b1);
    free(b2);
    free(x);
    free(x_);
    free(y);
    free(y_);

    return 0;
}
