
#ifndef CUDA_LIB_ReduceTest_CUH
#define CUDA_LIB_ReduceTest_CUH



__device__ uint randomizerGPU(int n = 42) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(n, tid, 0, &state);
    return curand(&state);
}

__global__ 
void kernel_cuda_wrapper_sum_4sNg7C3iRz(int32_t* initial_BVKSvzi7ir, int32_t* result_5DL2WgWjPX );



#endif
