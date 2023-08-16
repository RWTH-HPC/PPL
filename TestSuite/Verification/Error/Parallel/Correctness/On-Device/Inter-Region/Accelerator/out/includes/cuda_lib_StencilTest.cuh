
#ifndef CUDA_LIB_StencilTest_CUH
#define CUDA_LIB_StencilTest_CUH



__device__ uint randomizerGPU(int n = 42) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(n, tid, 0, &state);
    return curand(&state);
}

__global__ 
void kernel_cuda_wrapper_sum_rCQOyszoeH(int32_t* initial_UANbCb8XQf, int32_t* result_zQ0Xj2P8Lz );

__global__ 
void kernel_cuda_wrapper_sum_bYUinBGbIN(int32_t* initial_UANbCb8XQf, int32_t* result_zQ0Xj2P8Lz );



#endif
