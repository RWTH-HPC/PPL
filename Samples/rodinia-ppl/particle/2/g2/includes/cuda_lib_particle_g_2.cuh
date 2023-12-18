
#ifndef CUDA_LIB_particle_g_2_CUH
#define CUDA_LIB_particle_g_2_CUH



__device__ uint randomizerGPU(int n = 42) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(n, tid, 0, &state);
    return curand(&state);
}

__global__ 
void kernel_cuda_wrapper_u_init_zEljNqeMNI(int32_t*  Nparticles_AflVLOUd4r, double*  u1_JjqTLgAmUm_alOS831NNS, double* u_usdhqA1DH6 );

__global__ 
void kernel_cuda_wrapper_u_init_LDt82V3Q4y(double*  u1_JjqTLgAmUm_alOS831NNS, int32_t*  Nparticles_AflVLOUd4r, double* u_usdhqA1DH6 );

__global__ 
void kernel_cuda_wrapper_u_init_mcL9deInrq(double*  u1_JjqTLgAmUm_alOS831NNS, int32_t*  Nparticles_AflVLOUd4r, double* u_usdhqA1DH6 );

__global__ 
void kernel_cuda_wrapper_u_init_8u9rMtQzit(int32_t*  Nparticles_AflVLOUd4r, double*  u1_JjqTLgAmUm_alOS831NNS, double* u_usdhqA1DH6 );



#endif
