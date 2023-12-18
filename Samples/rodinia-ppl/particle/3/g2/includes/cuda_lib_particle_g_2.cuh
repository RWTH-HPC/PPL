
#ifndef CUDA_LIB_particle_g_2_CUH
#define CUDA_LIB_particle_g_2_CUH



__device__ uint randomizerGPU(int n = 42) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(n, tid, 0, &state);
    return curand(&state);
}

__global__ 
void kernel_cuda_wrapper_u_init_jbSoE3vhKa(int32_t*  Nparticles_CcjxmD75Zp, double*  u1_iSCTjLSB7b_36NhHedM04, double* u_CYHoLeWYXx );

__global__ 
void kernel_cuda_wrapper_u_init_vg8QPBCDrI(double*  u1_iSCTjLSB7b_36NhHedM04, int32_t*  Nparticles_CcjxmD75Zp, double* u_CYHoLeWYXx );

__global__ 
void kernel_cuda_wrapper_u_init_EPUPPspMoQ(double*  u1_iSCTjLSB7b_36NhHedM04, int32_t*  Nparticles_CcjxmD75Zp, double* u_CYHoLeWYXx );

__global__ 
void kernel_cuda_wrapper_u_init_ioGLUgAge2(int32_t*  Nparticles_CcjxmD75Zp, double*  u1_iSCTjLSB7b_36NhHedM04, double* u_CYHoLeWYXx );



#endif
