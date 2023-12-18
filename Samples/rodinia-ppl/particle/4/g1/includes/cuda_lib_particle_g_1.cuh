
#ifndef CUDA_LIB_particle_g_1_CUH
#define CUDA_LIB_particle_g_1_CUH



__device__ uint randomizerGPU(int n = 42) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(n, tid, 0, &state);
    return curand(&state);
}

__global__ 
void kernel_cuda_wrapper_u_init_zj9EbHdPzb(double*  u1_ZZHMvQb2i6_s96P5Y86vI, int32_t*  Nparticles_2f7BRjYtYI, double* u_x64gIli0uq );

__global__ 
void kernel_cuda_wrapper_u_init_U4raAdIpaG(double*  u1_ZZHMvQb2i6_s96P5Y86vI, int32_t*  Nparticles_2f7BRjYtYI, double* u_x64gIli0uq );



#endif
