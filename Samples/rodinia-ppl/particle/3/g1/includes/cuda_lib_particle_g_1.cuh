
#ifndef CUDA_LIB_particle_g_1_CUH
#define CUDA_LIB_particle_g_1_CUH



__device__ uint randomizerGPU(int n = 42) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(n, tid, 0, &state);
    return curand(&state);
}

__global__ 
void kernel_cuda_wrapper_u_init_2c04MpmCvE(double*  u1_eTgP6Ccpm7_8DIbDc4tdI, int32_t*  Nparticles_aBFhrb8G1G, double* u_s8LjziQsQp );

__global__ 
void kernel_cuda_wrapper_u_init_7bIvnL5XRN(double*  u1_eTgP6Ccpm7_8DIbDc4tdI, int32_t*  Nparticles_aBFhrb8G1G, double* u_s8LjziQsQp );



#endif
