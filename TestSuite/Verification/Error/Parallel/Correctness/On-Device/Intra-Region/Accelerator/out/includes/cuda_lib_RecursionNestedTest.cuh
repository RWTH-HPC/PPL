
#ifndef CUDA_LIB_RecursionNestedTest_CUH
#define CUDA_LIB_RecursionNestedTest_CUH



__device__ uint randomizerGPU(int n = 42) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(n, tid, 0, &state);
    return curand(&state);
}

__global__ 
void kernel_cuda_wrapper_increment_5S36YZbPJO(int32_t*  initial_EbYenl3BrQ, int32_t* result_gJ1dDQWGX2 );



#endif
