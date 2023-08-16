
#ifndef CUDA_LIB_DPNestedTest_CUH
#define CUDA_LIB_DPNestedTest_CUH



__device__ uint randomizerGPU(int n = 42) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(n, tid, 0, &state);
    return curand(&state);
}

__global__ 
void kernel_cuda_wrapper_matrixIncrement_eqctmEcrHD(int32_t*  initial_O7UVdy8jFf, int32_t* result_D7HcKyZsyA );



#endif
