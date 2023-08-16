
#ifndef CUDA_LIB_StencilNestedTest_CUH
#define CUDA_LIB_StencilNestedTest_CUH



__device__ uint randomizerGPU(int n = 42) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(n, tid, 0, &state);
    return curand(&state);
}

__global__ 
void kernel_cuda_wrapper_partialStencil_pLNuzDlBWy(int32_t*  initial_PiJdo0ggGx, int32_t* result_qN04jyEyY1 );



#endif
