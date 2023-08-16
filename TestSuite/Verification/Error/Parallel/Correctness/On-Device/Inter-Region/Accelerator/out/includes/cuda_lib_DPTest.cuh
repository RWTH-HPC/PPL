
#ifndef CUDA_LIB_DPTest_CUH
#define CUDA_LIB_DPTest_CUH



__device__ uint randomizerGPU(int n = 42) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(n, tid, 0, &state);
    return curand(&state);
}

__global__ 
void kernel_cuda_wrapper_incr_Djzg7MmFNO(int32_t* initial_a4JNRUnUuW, int32_t* result_ELCh4n0sJc, int INDEX0_XfYU6AM8ic);

__global__ 
void kernel_cuda_wrapper_incr_HpoRVRHOpX(int32_t* initial_a4JNRUnUuW, int32_t* result_ELCh4n0sJc, int INDEX0_XfYU6AM8ic);



#endif
