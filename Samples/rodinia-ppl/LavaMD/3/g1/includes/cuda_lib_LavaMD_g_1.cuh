
#ifndef CUDA_LIB_LavaMD_g_1_CUH
#define CUDA_LIB_LavaMD_g_1_CUH



__device__ uint randomizerGPU(int n = 42) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(n, tid, 0, &state);
    return curand(&state);
}



#endif
