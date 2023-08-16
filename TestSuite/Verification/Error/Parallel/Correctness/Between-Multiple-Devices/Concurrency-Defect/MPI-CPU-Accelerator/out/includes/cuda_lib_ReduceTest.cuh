
#ifndef CUDA_LIB_ReduceTest_CUH
#define CUDA_LIB_ReduceTest_CUH



__device__ uint randomizerGPU(int n = 42) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(n, tid, 0, &state);
    return curand(&state);
}

__global__ 
void kernel_cuda_wrapper_sum_EZu4NSyRhq(int32_t* initial_p5Bg5zAdt9, int32_t* result_PCTcyANvaq );

__global__ 
void kernel_cuda_wrapper_sum_COqcq4aIJy(int32_t* initial_p5Bg5zAdt9, int32_t* result_PCTcyANvaq );



#endif
