#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>

#define N 1000000000

long lehmer_random_number_generator(long x) {
    return (16807 * x) % 2147483647;
}

double uniform(long x) {
    return x / 2147483647.0;
}

double monte_carl(int rank) {
    long sx = rank * 2133;
    long sy = rank * 33;

    long inside = 0;
    long outside = 0;
    for (long i = 0; i < N; i++) {
        sx = lehmer_random_number_generator(sx);
        sy = lehmer_random_number_generator(sy);
        double x = uniform(sx);
        double y = uniform(sy);

        double k = x * x + y * y;
        if (k <= 1.0) {
            inside += 1;
        } else {
            outside += 1;
        }
    }

    return (4.0 * inside) / (inside + outside);
}

int main(int argc, char** argv) {
    int rank, nprocs;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double time = omp_get_wtime();

    double pi_local = 0;
    for (int i = 0; i < 96 / nprocs; i++) {
        pi_local += monte_carl(rank + 1);
    }
    pi_local = pi_local / (96 / nprocs);

    double pi;    
    MPI_Reduce(&pi_local, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        pi = pi / nprocs;
        time = omp_get_wtime() - time;
        printf("\n\nComputation took %f with results: %f\n\n",time, pi);
    }    

    MPI_Finalize();

    return 0;
}
