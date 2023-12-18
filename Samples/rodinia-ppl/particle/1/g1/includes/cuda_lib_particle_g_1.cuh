
#ifndef CUDA_LIB_particle_g_1_CUH
#define CUDA_LIB_particle_g_1_CUH



__device__ uint randomizerGPU(int n = 42) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(n, tid, 0, &state);
    return curand(&state);
}

__global__ 
void kernel_cuda_wrapper_u_init_zK3zsoJUg7(double*  u1_UstTxnfY90_J45bILqR5e, int32_t*  Nparticles_Xtno2sLxiz, double* u_uVrNRP6SnA );

__global__ 
void kernel_cuda_wrapper_u_init_vlOhsxzeZk(double*  u1_UstTxnfY90_J45bILqR5e, int32_t*  Nparticles_Xtno2sLxiz, double* u_uVrNRP6SnA );



#endif
