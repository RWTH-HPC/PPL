#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <curand.h>

#define BATCH_SIZE 262144
#define FEATURE_LENGTH 64

#define WEIGHTS 64
#define CLASSES 10

#define SPLIT_SIZE 8192
#define BLOCK_SIZE 512

void fill_rand(float *weights, int rows, int cols)
{
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);

    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

    curandGenerateUniform(prng, weights, rows * cols);
}

__device__
float activation(float* weights, float* features, int length) {
    float z = 0;
    for (int k = 0; k < length; k++) {
        z += weights[k] * features[k];
    }
    return tanhf(z);
}

__device__
void layer(float* weights, float* features, float* activations, int rows, int columns) {
    for (int k = 0; k < rows; k++) {
        float* row = weights + k * columns;
        activations[k] = activation(row, features, columns);
    }
}

__global__
void forward(float* batch, float* weights, float* result, int activations_length, int features_length) {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int i = threadIdx.x + blockId * (blockDim.x * blockDim.y);    

    float* features = batch + i * features_length;
    float* activations = result + i * activations_length;
    layer(weights, features, activations, activations_length, features_length);
}

int main(int argc, char** argv)
{
    //omp_set_dynamic(0);
    //omp_set_nested(1);

    // 1. Memory allocation
    float* batch_h = (float *) malloc(BATCH_SIZE * FEATURE_LENGTH * sizeof(float));
    float* batch;
    cudaMalloc(&batch, BATCH_SIZE * FEATURE_LENGTH * sizeof(float));

    float* weights_1;
    cudaMalloc(&weights_1, WEIGHTS * FEATURE_LENGTH * sizeof(float));
    float* weights_2;
    cudaMalloc(&weights_2, WEIGHTS * WEIGHTS * sizeof(float));
    float* weights_3;
    cudaMalloc(&weights_3, WEIGHTS * WEIGHTS * sizeof(float));
    float* weights_4;
    cudaMalloc(&weights_4, WEIGHTS * WEIGHTS * sizeof(float));
    float* weights_5;
    cudaMalloc(&weights_5, WEIGHTS * WEIGHTS * sizeof(float));
    float* weights_6;
    cudaMalloc(&weights_6, WEIGHTS * WEIGHTS * sizeof(float));
    float* weights_7;
    cudaMalloc(&weights_7, WEIGHTS * WEIGHTS * sizeof(float));
    float* weights_8;
    cudaMalloc(&weights_8, CLASSES * WEIGHTS * sizeof(float));

    float* activations_1;
    cudaMalloc(&activations_1, BATCH_SIZE * WEIGHTS * sizeof(float));
    float* activations_2;
    cudaMalloc(&activations_2, BATCH_SIZE * WEIGHTS * sizeof(float));
    float* activations_3;
    cudaMalloc(&activations_3, BATCH_SIZE * WEIGHTS * sizeof(float));
    float* activations_4;
    cudaMalloc(&activations_4, BATCH_SIZE * WEIGHTS * sizeof(float));
    float* activations_5;
    cudaMalloc(&activations_5, BATCH_SIZE * WEIGHTS * sizeof(float));
    float* activations_6;
    cudaMalloc(&activations_6, BATCH_SIZE * WEIGHTS * sizeof(float));
    float* activations_7;
    cudaMalloc(&activations_7, BATCH_SIZE * WEIGHTS * sizeof(float));

    float* results_h = (float *) malloc(BATCH_SIZE * CLASSES * sizeof(float));
    float* results;
    cudaMalloc(&results, BATCH_SIZE * CLASSES * sizeof(float));

    printf("Allocation done\n");

    // 2. Random initialization
    int seed = 42;
    srand(seed);
    for (int i = 0; i < BATCH_SIZE * FEATURE_LENGTH; i++) {
        batch_h[i] = ((float) rand() / (float) RAND_MAX) * 2.0 - 1.0;
    }

    fill_rand(weights_1, WEIGHTS, FEATURE_LENGTH);
    fill_rand(weights_2, WEIGHTS, WEIGHTS);
    fill_rand(weights_3, WEIGHTS, WEIGHTS);
    fill_rand(weights_4, WEIGHTS, WEIGHTS);
    fill_rand(weights_5, WEIGHTS, WEIGHTS);
    fill_rand(weights_6, WEIGHTS, WEIGHTS);
    fill_rand(weights_7, WEIGHTS, WEIGHTS);
    fill_rand(weights_8, CLASSES, WEIGHTS);
    printf("Random init done\n");

    dim3 grid(BATCH_SIZE / SPLIT_SIZE, (SPLIT_SIZE / BLOCK_SIZE));
    printf("Starting clock\n");
    // 3. Start clock
    double time = omp_get_wtime();
    
    cudaMemcpy(batch, batch_h, BATCH_SIZE * FEATURE_LENGTH * sizeof(float), cudaMemcpyHostToDevice);

    // 4. Neural network
    forward<<<grid, BLOCK_SIZE>>>(batch, weights_1, activations_1, WEIGHTS, FEATURE_LENGTH);
    forward<<<grid, BLOCK_SIZE>>>(activations_1, weights_2, activations_2, WEIGHTS, WEIGHTS);
    forward<<<grid, BLOCK_SIZE>>>(activations_2, weights_3, activations_3, WEIGHTS, WEIGHTS);
    forward<<<grid, BLOCK_SIZE>>>(activations_3, weights_4, activations_4, WEIGHTS, WEIGHTS);
    forward<<<grid, BLOCK_SIZE>>>(activations_4, weights_5, activations_5, WEIGHTS, WEIGHTS);
    forward<<<grid, BLOCK_SIZE>>>(activations_5, weights_6, activations_6, WEIGHTS, WEIGHTS);
    forward<<<grid, BLOCK_SIZE>>>(activations_6, weights_7, activations_7, WEIGHTS, WEIGHTS);
    forward<<<grid, BLOCK_SIZE>>>(activations_7, weights_8, results, CLASSES, WEIGHTS);

    cudaMemcpy(results_h, results, BATCH_SIZE * CLASSES * sizeof(float), cudaMemcpyDeviceToHost);

    // 4. Stop time
    time = omp_get_wtime() - time;
    printf("Stopping clock\n");
    
    // 5. Computation for non-trivialization of code
    float avg = 0.0;
    for (int i = 0; i < BATCH_SIZE * CLASSES; i++) {
        avg += results_h[i];
    }
    avg /= BATCH_SIZE * CLASSES;

    printf("Computation took: %f with result: %f\n", time, avg);
      
    cudaFree(batch);
    cudaFree(weights_1);
    cudaFree(weights_2);
    cudaFree(weights_3);
    cudaFree(weights_4);
    cudaFree(weights_5);
    cudaFree(weights_6);
    cudaFree(weights_7);
    cudaFree(weights_8);
    cudaFree(activations_1);
    cudaFree(activations_2);
    cudaFree(activations_3);
    cudaFree(activations_4);
    cudaFree(activations_5);
    cudaFree(activations_6);
    cudaFree(activations_7);
    cudaFree(results);
    
    return 0;
}
