
#ifndef CUDA_LIB_ReduceNestedTest_CUH
#define CUDA_LIB_ReduceNestedTest_CUH



__device__ uint randomizerGPU(int n = 42) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(n, tid, 0, &state);
    return curand(&state);
}

__global__ 
void kernel_cuda_wrapper_rowSum_JTkoi9qzzS(int32_t*  initial_Cro7ZoaMZD, int32_t* result_mrCUu80Ufm );



#endif
