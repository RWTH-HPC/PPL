#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define N (size_t) 8 * 65536
#define FEATURE_LENGTH 4096
#define CLASSIFIER 4096

void normalize(double* data, double* normalized, double mu, double variance)
{
    for (size_t i = 0; i < FEATURE_LENGTH; i++)
    {
        normalized[i] = (data[i] - mu) / variance;
    }
}

double extract(double* normalized)
{
    double sum = 0.0;
    for (int i = 0; i < FEATURE_LENGTH; i++)
    {
        if (normalized[i] < 0.0) {
            sum = sum - normalized[i];
        } else {
            sum = sum + normalized[i];
        }
    }
    sum /= FEATURE_LENGTH;
    return sum;
}

int classify(double feature, double* classifier_thresh, double* classifier_weights)
{
    double vote = 0.0;
    for (int i = 0; i < CLASSIFIER; i++)
    {
        if (feature > classifier_thresh[i])
        {
            vote += classifier_weights[i];
        } else {
            vote -= classifier_weights[i];
        }
    }
    return vote >= 0.0 ? 1 : -1;
}

int main(int argc, char** argv)
{
    // 1. Data allocation
    double* data = (double *) malloc(N * FEATURE_LENGTH * sizeof(double));
    double* normalized = (double *) malloc(N * FEATURE_LENGTH * sizeof(double)); 
    double* features = (double *) malloc(N * sizeof(double));
    int* classes = (int *) malloc(N * sizeof(int));

    double* classifier_weights = (double *) malloc(CLASSIFIER * sizeof(double));
    double* classifier_thresh = (double *) malloc(CLASSIFIER * sizeof(double));
    if (data == NULL || classifier_weights == NULL || classifier_thresh == NULL || classes == NULL || normalized == NULL || features == NULL)
    {
        printf("Error allocating memory\n");
        return -1;
    }

    // 2. Random initialization
    srand(42);
    for (size_t i = 0; i < N * FEATURE_LENGTH; i++) {
        data[i] = ((double) rand() / (double) RAND_MAX) * 2.0 - 1.0;
    }

    srand(43);
    for (int i = 0; i < CLASSIFIER; i++) {
        classifier_weights[i] = ((double) rand() / (double) RAND_MAX);
        classifier_thresh[i] = ((double) rand() / (double) RAND_MAX) * 0.2 + 1.8;
    }

    double start = omp_get_wtime();
    
    #pragma omp parallel
    {
        #pragma omp for
        for (size_t i = 0; i < N; i++)
        {
            double* d = data + (N - i - 1) * FEATURE_LENGTH;
            normalize(d, normalized + (N - i - 1) * FEATURE_LENGTH, 0.4, 0.3);
        }

        #pragma omp for
        for (size_t i = 0; i < N; i++)
        {
            double* n = normalized + i * FEATURE_LENGTH;
            features[i] = extract(n);
        }

        #pragma omp for
        for (size_t i = 0; i < N; i++)
        {
            classes[N - i - 1] = classify(features[N - i - 1], classifier_thresh, classifier_weights);
        }
    }

    double time = omp_get_wtime() - start;

    size_t class_A = 0;
    size_t class_B = 0;
    for (size_t i = 0; i < N; i++)
    {
        if (classes[i] == -1)
        {
            class_A += 1;
        } else {
            class_B += 1;
        }
    }

    printf("Class A: %d, Class B: %d, Time: %f\n", class_A, class_B, time);

    free(data);
    free(normalized);
    free(features);
    free(classes);
    free(classifier_weights);
    free(classifier_thresh);

    return 0;
}
