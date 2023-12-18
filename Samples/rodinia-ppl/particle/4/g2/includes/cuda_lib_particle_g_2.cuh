
#ifndef CUDA_LIB_particle_g_2_CUH
#define CUDA_LIB_particle_g_2_CUH



__device__ uint randomizerGPU(int n = 42) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(n, tid, 0, &state);
    return curand(&state);
}

__global__ 
void kernel_cuda_wrapper_u_init_EuFs5tKDLa(int32_t*  Nparticles_GEZW4pzeoW, double*  u1_cRx97fdDWf_s2dfuBCP2s, double* u_bceK7yP7v7 );

__global__ 
void kernel_cuda_wrapper_u_init_gkZtexcP2p(double*  u1_cRx97fdDWf_s2dfuBCP2s, int32_t*  Nparticles_GEZW4pzeoW, double* u_bceK7yP7v7 );

__global__ 
void kernel_cuda_wrapper_u_init_3SBzj8X8zZ(double*  u1_cRx97fdDWf_s2dfuBCP2s, int32_t*  Nparticles_GEZW4pzeoW, double* u_bceK7yP7v7 );

__global__ 
void kernel_cuda_wrapper_u_init_zYHeikJnbS(int32_t*  Nparticles_GEZW4pzeoW, double*  u1_cRx97fdDWf_s2dfuBCP2s, double* u_bceK7yP7v7 );



#endif
