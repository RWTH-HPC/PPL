
#ifndef CUDA_LIB_StencilTest_CUH
#define CUDA_LIB_StencilTest_CUH



__device__ uint randomizerGPU(int n = 42) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(n, tid, 0, &state);
    return curand(&state);
}

__global__ 
void kernel_cuda_wrapper_sum_lLqlI3kORr(int32_t* initial_ZhAABHPNda, int32_t* result_7uQyeL2j9T );

__global__ 
void kernel_cuda_wrapper_sum_2mLxD8zRet(int32_t* initial_ZhAABHPNda, int32_t* result_7uQyeL2j9T );



#endif
