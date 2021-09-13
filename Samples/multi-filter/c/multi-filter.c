#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#define N 8194
#define REPETITIONS 10


int main(int argc, char** argv) {
    double* image = (double *) malloc(N * N * sizeof(double));
    double* upper_half = (double *) malloc((N - 2) * (N - 2) * sizeof(double));
    double* lower_half = (double *) malloc((N - 2) * (N - 2) * sizeof(double));
    double* full = (double *) malloc((N - 2) * (N - 2) * sizeof(double));
    if (image == NULL || upper_half == NULL || lower_half == NULL || upper_half == NULL) {
        printf("Malloc error.\n");
        return -1;
    }

    omp_set_dynamic(1);
    omp_set_nested(1);

    int seed = 42;
    double sum = 0.0;
    for (int i = 0; i < REPETITIONS; i++)
    {
        seed++;
        srand(seed);
        for (int i = 0; i < N * N; i++) {
            image[i] = (double) rand() / (double) RAND_MAX;
        }

        double time = omp_get_wtime();

        #pragma omp parallel sections num_threads(2)
        {
            #pragma omp section
            #pragma omp parallel for num_threads(24)
            for (int i = 0; i < (N - 2) / 2; i++) {
                for (int j = 0; j < (N - 2); j++)
                {
                    double right = 2 * image[(i + 1) * N + (j + 2)] + image[i * N + (j + 2)] + image[(i + 2) * N + (j + 2)];
                    double left = 2 * image[(i + 1) * N + j] + image[i * N + j] + image[(i + 1) * N + j];
                    upper_half[i * (N - 2) + j] += right - left;

                    double outer = image[(i + 1) * N + j] + image[(i + 1) * N + (j + 2)] + image[i * N + (j + 1)] + image[(i + 2) * N + (j + 1)];
                    full[i * (N - 2) + j] += outer - 4 * image[(i + 1) * N + (j + 1)]; 
		        }
            }
         
	    #pragma omp section
            #pragma omp parallel for num_threads(24)
            for (int i = (N - 2) / 2; i < (N - 2); i++)
            {
                for (int j = 0; j < (N - 2); j++)
                {
                    double right = image[(i + 1) * N + (j + 2)] + image[i * N + (j + 2)] + image[(i + 2) * N + (j + 2)];
                    double left = image[(i + 1) * N + j] + image[i * N + j] + image[(i + 2) * N + j];
                    lower_half[i * (N - 2) + j] += right - left;

                    double outer = image[(i + 1) * N + j] + image[(i + 1) * N + (j + 2)] + image[i * N + (j + 1)] + image[(i + 2) * N + (j + 1)];
                    full[i * (N - 2) + j] += outer - 4 * image[(i + 1) * N + (j + 1)];
                }
            }
     
        }

        time = omp_get_wtime() - time;
        sum += time;
    }
    double mean = sum / REPETITIONS;

    printf("Computation took %f (total: %f) with results: %f, %f, %f\n", mean, sum, lower_half[(N - 2) * (N - 2) - 10], upper_half[10], full[10]);

    free(image);
    free(upper_half);
    free(lower_half);
    free(full);

    return 0;
}
