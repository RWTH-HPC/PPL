
#ifndef CUDA_LIB_MapNestedTest_CUH
#define CUDA_LIB_MapNestedTest_CUH



__device__ uint randomizerGPU(int n = 42) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(n, tid, 0, &state);
    return curand(&state);
}

__global__ 
void kernel_cuda_wrapper_matrixIncrement_FU1WgUdkbG(int32_t*  initial_4p4ktWJZxo, int32_t* result_NcgNP2jlMG );



#endif
