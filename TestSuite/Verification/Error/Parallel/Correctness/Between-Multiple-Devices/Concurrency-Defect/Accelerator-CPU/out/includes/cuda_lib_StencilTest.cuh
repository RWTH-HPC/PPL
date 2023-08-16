
#ifndef CUDA_LIB_StencilTest_CUH
#define CUDA_LIB_StencilTest_CUH



__device__ uint randomizerGPU(int n = 42) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(n, tid, 0, &state);
    return curand(&state);
}

__global__ 
void kernel_cuda_wrapper_sum_Puq2QrGy3A(int32_t* initial_qXGiQrK3bj, int32_t* result_76cpeTeYqv );



#endif
