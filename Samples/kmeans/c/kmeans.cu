#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define K 100
#define N 100000
#define NITERS 100
#define THREADSPERBLOCK 1024

struct point_t {
    double x;
    double y;
};

__global__ void k_means(int niters, point_t *points, point_t *centroids, int *assignment) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int iter = 0; iter < niters; ++iter) {
        if (tid < N) {
            double optimal_dist = DBL_MAX;
            // Calculate Euclidean distance to each centroid and determine the closest
            // mean
            for (int j = 0; j < K; ++j) {
                double dist = (points[tid].x - centroids[j].x) * (points[tid].x - centroids[j].x) + (points[tid].y - centroids[j].y) * (points[tid].y - centroids[j].y);
                if (dist < optimal_dist) {
                optimal_dist = dist;
                assignment[tid] = j;
                }
            }
        }
        // Calculate new positions of centroids
        int count[K];
        double sum_x[K];
        double sum_y[K];
        if (tid < K) {
            count[tid] = 0;
            sum_x[tid] = 0.0;
            sum_y[tid] = 0.0;
        }
        if (tid < N) {
            sum_x[assignment[tid]] += points[tid].x;
            sum_y[assignment[tid]] += points[tid].y;
            count[assignment[tid]]++;
        }
        if (tid < K) {
            if (count[tid] != 0.0) {
                centroids[tid].x = sum_x[tid] / count[tid];
                centroids[tid].y = sum_y[tid] / count[tid];
            }
        }
    }
}

int main(int argc, const char *argv[]) {
    srand(1234);
    struct point_t *points = (struct point_t *)malloc(N * sizeof(struct point_t));
    struct point_t *centroids =
        (struct point_t *)malloc(K * sizeof(struct point_t));
    int *assignment = (int *)malloc(N * sizeof(int));
    struct point_t *memory =
        (struct point_t *)malloc((NITERS + 1) * K * sizeof(struct point_t));

    for (int i = 0; i < N; ++i) {
        points[i].x = rand() % 100;
        points[i].y = rand() % 100;
    }
    for (int i = 0; i < K; ++i) {
        centroids[i].x = rand() % 100;
        centroids[i].y = rand() % 100;
    }
    for (int j = 0; j < K; ++j) {
        memory[j].x = centroids[j].x;
        memory[j].y = centroids[j].y;
    }

    double start = omp_get_wtime();

    // Allocate memory for GPU
    struct point_t *d_points = 0;
    struct point_t *d_centroids = 0;
    int *d_assignment = 0;
    cudaMalloc((void **)&d_points, N * sizeof(struct point_t));
    cudaMalloc((void **)&d_centroids, K * sizeof(struct point_t));
    cudaMalloc((void **)&d_assignment, N * sizeof(int));

    // Copy data to GPU
    cudaMemcpy(d_points, points, N * sizeof(struct point_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, centroids, K * sizeof(struct point_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_assignment, assignment, N * sizeof(int), cudaMemcpyHostToDevice);

    k_means<<<(N + THREADSPERBLOCK - 1) / THREADSPERBLOCK, THREADSPERBLOCK>>>(NITERS, d_points, d_centroids, d_assignment);
    cudaDeviceSynchronize();

    // Copy assignments and centroids back to host
    cudaMemcpy(centroids, d_centroids, N * sizeof(struct point_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(assignment, d_assignment, N * sizeof(int), cudaMemcpyDeviceToHost);

    double time = omp_get_wtime() - start;

    printf("Computation took: %f\n", time);

    for (int i = 0; i < K; ++i) {
        printf("%2.5f %2.5f\n", centroids[i].x, centroids[i].y);
    }

    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_assignment);
    free(assignment);
    free(centroids);
    free(points);
    free(memory);

    return 0;
  }