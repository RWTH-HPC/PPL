#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <omp.h>

#define K 100
#define N 100000
#define NITERS 100

struct point_t {
    double x;
    double y;
};

void k_means(int niters, struct point_t *points, struct point_t *centroids, int *assignment, int n, int k) {
    for (int iter = 0; iter < niters; ++iter) {
        // determine nearest centroids
        for (int i = 0; i < n; ++i) {
            double optimal_dist = DBL_MAX;
            for (int j = 0; j < k; ++j) {
                double dist = (points[i].x - centroids[j].x) * (points[i].x - centroids[j].x) + (points[i].y - centroids[j].y) * (points[i].y - centroids[j].y);
                if (dist < optimal_dist) {
                    optimal_dist = dist;
                    assignment[i] = j;
                }
            }
        }

        // update centroid positions
        for (int j = 0; j < k; ++j) {
            int count = 0;
            double sum_x = 0.;
            double sum_y = 0.;
            for (int i = 0; i < n; ++i) {
                if (assignment[i] == j) {
                    count++;
                    sum_x += points[i].x;
                    sum_y += points[i].y;
                }
            }
            if (count != 0) {
                centroids[j].x = sum_x / count;
                centroids[j].y = sum_y / count;
            }
        }
    }
}

int main(int argc, const char* argv[]) {
    srand(1234);
    struct point_t * points = (struct point_t*) malloc(N * sizeof(struct point_t));
    struct point_t * centroids = (struct point_t*) malloc(K * sizeof(struct point_t));
    int * assignment = (int*) malloc(N * sizeof(int));

    for (int i = 0; i < N; ++i) {
        points[i].x = rand() % 100;
        points[i].y = rand() % 100;
    }
    for (int i = 0; i < K; ++i) {
        centroids[i].x = rand() % 100;
        centroids[i].y = rand() % 100;
    }

    double start = omp_get_wtime();
    k_means(NITERS, points, centroids, assignment, N, K);
    double time = omp_get_wtime() - start;

    printf("Computation took: %f\n", time);

    for (int i = 0; i < K; ++i) {
        printf("%2.5f %2.5f\n", centroids[i].x, centroids[i].y);
    }

    free(assignment);
    free(centroids);
    free(points);

    return 0;
}
