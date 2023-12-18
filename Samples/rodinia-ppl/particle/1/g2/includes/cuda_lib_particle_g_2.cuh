
#ifndef CUDA_LIB_particle_g_2_CUH
#define CUDA_LIB_particle_g_2_CUH



__device__ uint randomizerGPU(int n = 42) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(n, tid, 0, &state);
    return curand(&state);
}

__global__ 
void kernel_cuda_wrapper_u_init_8bIU6Vh2Lc(int32_t*  Nparticles_9WpkAzeGZz, double*  u1_iWRHIHKWBp_RNut6JFnfX, double* u_4nVTHiRif8 );

__global__ 
void kernel_cuda_wrapper_u_init_0qFIgBT9bz(double*  u1_iWRHIHKWBp_RNut6JFnfX, int32_t*  Nparticles_9WpkAzeGZz, double* u_4nVTHiRif8 );

__global__ 
void kernel_cuda_wrapper_u_init_4N21VvnF41(double*  u1_iWRHIHKWBp_RNut6JFnfX, int32_t*  Nparticles_9WpkAzeGZz, double* u_4nVTHiRif8 );

__global__ 
void kernel_cuda_wrapper_u_init_ZjrW763TVm(int32_t*  Nparticles_9WpkAzeGZz, double*  u1_iWRHIHKWBp_RNut6JFnfX, double* u_4nVTHiRif8 );



#endif
