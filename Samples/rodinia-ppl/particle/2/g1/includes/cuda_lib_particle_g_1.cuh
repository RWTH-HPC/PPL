
#ifndef CUDA_LIB_particle_g_1_CUH
#define CUDA_LIB_particle_g_1_CUH



__device__ uint randomizerGPU(int n = 42) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(n, tid, 0, &state);
    return curand(&state);
}

__global__ 
void kernel_cuda_wrapper_u_init_eC5NGPiAvv(double*  u1_RHKVSgjzlz_N4IMHoZdrq, int32_t*  Nparticles_RMBeajvFXr, double* u_NAIj3hYPTU );

__global__ 
void kernel_cuda_wrapper_u_init_FFnUAftOOH(double*  u1_RHKVSgjzlz_N4IMHoZdrq, int32_t*  Nparticles_RMBeajvFXr, double* u_NAIj3hYPTU );



#endif
