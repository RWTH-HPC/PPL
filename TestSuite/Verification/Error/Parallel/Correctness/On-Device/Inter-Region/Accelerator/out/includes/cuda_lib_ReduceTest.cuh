
#ifndef CUDA_LIB_ReduceTest_CUH
#define CUDA_LIB_ReduceTest_CUH



__device__ uint randomizerGPU(int n = 42) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(n, tid, 0, &state);
    return curand(&state);
}

__global__ 
void kernel_cuda_wrapper_sum_Ji7XGfHaKX(int32_t* initial_USRQYvaGss, int32_t* result_CFeZRcqAml );

__global__ 
void kernel_cuda_wrapper_sum_qF8o5kc6LL(int32_t* initial_USRQYvaGss, int32_t* result_CFeZRcqAml );



#endif
