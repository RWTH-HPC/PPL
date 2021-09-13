#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#define BATCH_SIZE 262144
#define FEATURE_LENGTH 64

#define WEIGHTS 64
#define CLASSES 10

float activation(float* weights, float* features, int length) {
    float z = 0;
    for (int k = 0; k < length; k++) {
        z += weights[k] * features[k];
    }
    return tanhf(z);
}

void layer(float* weights, float* features, float* activations, int rows, int columns) {
    for (int k = 0; k < rows; k++) {
        float* row = weights + k * columns;
        activations[k] = activation(row, features, columns);
    }
}

void forward(float* batch, float* weights, float* result, int activations_length, int features_length) {
    #pragma omp parallel for
    for (int i = 0; i < BATCH_SIZE; i++) {
        float* features = batch + i * features_length;
        float* activations = result + i * activations_length;

        layer(weights, features, activations, activations_length, features_length);
    }
}

int main(int argc, char** argv)
{
    #pragma omp parallel
    #pragma omp single
    printf("number of threads: %d \n", omp_get_num_threads());    

    // 1. Memory allocation
    float* batch = (float*) malloc(BATCH_SIZE * FEATURE_LENGTH * sizeof(float));
    float* weights_1 = (float*) malloc(WEIGHTS * FEATURE_LENGTH * sizeof(float));
    float* weights_2 = (float*) malloc(WEIGHTS * WEIGHTS * sizeof(float));
    float* weights_3 = (float*) malloc(WEIGHTS * WEIGHTS * sizeof(float));
    float* weights_4 = (float*) malloc(WEIGHTS * WEIGHTS * sizeof(float));
    float* weights_5 = (float*) malloc(WEIGHTS * WEIGHTS * sizeof(float));
    float* weights_6 = (float*) malloc(WEIGHTS * WEIGHTS * sizeof(float));
    float* weights_7 = (float*) malloc(WEIGHTS * WEIGHTS * sizeof(float));
    float* weights_8 = (float*) malloc(CLASSES * WEIGHTS * sizeof(float));

    float* activations_1 = (float*) malloc(BATCH_SIZE * WEIGHTS * sizeof(float));
    float* activations_2 = (float*) malloc(BATCH_SIZE * WEIGHTS * sizeof(float));
    float* activations_3 = (float*) malloc(BATCH_SIZE * WEIGHTS * sizeof(float));
    float* activations_4 = (float*) malloc(BATCH_SIZE * WEIGHTS * sizeof(float));
    float* activations_5 = (float*) malloc(BATCH_SIZE * WEIGHTS * sizeof(float));
    float* activations_6 = (float*) malloc(BATCH_SIZE * WEIGHTS * sizeof(float));
    float* activations_7 = (float*) malloc(BATCH_SIZE * WEIGHTS * sizeof(float));
    float* results = (float*) malloc(BATCH_SIZE * CLASSES * sizeof(float));

    if (batch == NULL || results == NULL
        || weights_1 == NULL || weights_2 == NULL || weights_3 == NULL || weights_4 == NULL
        || weights_5 == NULL || weights_6 == NULL || weights_7 == NULL || weights_8 == NULL
        || activations_1 == NULL || activations_2 == NULL || activations_3 == NULL || activations_4 == NULL
        || activations_5 == NULL || activations_6 == NULL || activations_7 == NULL) {
            printf("Error allocating memory\n");
            return -1;
        }
    printf("Allocation done\n");

    // 2. Random initialization
    int seed = 42;
    srand(seed);
    for (int i = 0; i < BATCH_SIZE * FEATURE_LENGTH; i++) {
        batch[i] = ((float) rand() / (float) RAND_MAX) * 2.0 - 1.0;
    }
    for (int i = 0; i < WEIGHTS * FEATURE_LENGTH; i++) {
        weights_1[i] = ((float) rand() / (float) RAND_MAX) * 2.0 - 1.0;
    }
    for (int i = 0; i < WEIGHTS * WEIGHTS; i++) {
        weights_2[i] = ((float) rand() / (float) RAND_MAX) * 2.0 - 1.0;
        weights_3[i] = ((float) rand() / (float) RAND_MAX) * 2.0 - 1.0;
        weights_4[i] = ((float) rand() / (float) RAND_MAX) * 2.0 - 1.0;
        weights_5[i] = ((float) rand() / (float) RAND_MAX) * 2.0 - 1.0;
        weights_6[i] = ((float) rand() / (float) RAND_MAX) * 2.0 - 1.0;
        weights_7[i] = ((float) rand() / (float) RAND_MAX) * 2.0 - 1.0;
    }
    for (int i = 0; i < CLASSES * WEIGHTS; i++) {
        weights_8[i] = ((float) rand() / (float) RAND_MAX) * 2.0 - 1.0;
    }
    printf("Random init done\n");

    printf("Starting clock\n");
    // 3. Start clock
    double time = omp_get_wtime();
    
    // 4. Neural network
    forward(batch, weights_1, activations_1, WEIGHTS, FEATURE_LENGTH);
    forward(activations_1, weights_2, activations_2, WEIGHTS, WEIGHTS);
    forward(activations_2, weights_3, activations_3, WEIGHTS, WEIGHTS);
    forward(activations_3, weights_4, activations_4, WEIGHTS, WEIGHTS);
    forward(activations_4, weights_5, activations_5, WEIGHTS, WEIGHTS);
    forward(activations_5, weights_6, activations_6, WEIGHTS, WEIGHTS);
    forward(activations_6, weights_7, activations_7, WEIGHTS, WEIGHTS);
    forward(activations_7, weights_8, results, CLASSES, WEIGHTS);

    // 4. Stop time
    time = omp_get_wtime() - time;
    printf("Stopping clock\n");
    
    // 5. Computation for non-trivialization of code
    float avg = 0.0f;
    for (int i = 0; i < BATCH_SIZE * CLASSES; i++) {
        avg += results[i];
    }
    avg /= BATCH_SIZE * CLASSES;

    printf("Computation took: %f with result: %f\n", time, avg);
    
    free(batch);

    free(weights_1);
    free(weights_2);
    free(weights_3);
    free(weights_4);
    free(weights_5);
    free(weights_6);
    free(weights_7);
    free(weights_8);

    free(activations_1);
    free(activations_2);
    free(activations_3);
    free(activations_4);
    free(activations_5);
    free(activations_6);
    free(activations_7);

    free(results);

    return 0;
}
