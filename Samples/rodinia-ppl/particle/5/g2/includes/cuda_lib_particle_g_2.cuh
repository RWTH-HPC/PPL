
#ifndef CUDA_LIB_particle_g_2_CUH
#define CUDA_LIB_particle_g_2_CUH



__device__ uint randomizerGPU(int n = 42) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(n, tid, 0, &state);
    return curand(&state);
}

__global__ 
void kernel_cuda_wrapper_u_init_r5ejBanMyg(int32_t*  Nparticles_A3WboS1Ear, double*  u1_z1OL8FprVl_IX1WUErsW5, double* u_PPkYFxALrb );

__global__ 
void kernel_cuda_wrapper_u_init_AcGUvJkq7y(double*  u1_z1OL8FprVl_IX1WUErsW5, int32_t*  Nparticles_A3WboS1Ear, double* u_PPkYFxALrb );

__global__ 
void kernel_cuda_wrapper_u_init_gCRghlciun(double*  u1_z1OL8FprVl_IX1WUErsW5, int32_t*  Nparticles_A3WboS1Ear, double* u_PPkYFxALrb );

__global__ 
void kernel_cuda_wrapper_u_init_g6JsLN2Lqm(int32_t*  Nparticles_A3WboS1Ear, double*  u1_z1OL8FprVl_IX1WUErsW5, double* u_PPkYFxALrb );



#endif
