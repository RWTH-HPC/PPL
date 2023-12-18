
#ifndef CUDA_LIB_particle_g_1_CUH
#define CUDA_LIB_particle_g_1_CUH



__device__ uint randomizerGPU(int n = 42) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(n, tid, 0, &state);
    return curand(&state);
}

__global__ 
void kernel_cuda_wrapper_u_init_ByR3Bpa2y2(double*  u1_bu8N2svs8Q_0lHZcF0TIO, int32_t*  Nparticles_ftpCX5UbUk, double* u_75B8MOd4dN );

__global__ 
void kernel_cuda_wrapper_u_init_7qezuspv9u(double*  u1_bu8N2svs8Q_0lHZcF0TIO, int32_t*  Nparticles_ftpCX5UbUk, double* u_75B8MOd4dN );



#endif
