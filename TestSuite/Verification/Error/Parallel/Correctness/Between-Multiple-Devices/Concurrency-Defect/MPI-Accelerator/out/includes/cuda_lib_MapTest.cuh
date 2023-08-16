
#ifndef CUDA_LIB_MapTest_CUH
#define CUDA_LIB_MapTest_CUH



__device__ uint randomizerGPU(int n = 42) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(n, tid, 0, &state);
    return curand(&state);
}

__global__ 
void kernel_cuda_wrapper_increment_a0V8c3egsK(int32_t*  initial_q9zXNRkOQa, int32_t* result_lePGdKBCEb );

__global__ 
void kernel_cuda_wrapper_increment_NcekzbsFgR(int32_t*  initial_q9zXNRkOQa, int32_t* result_lePGdKBCEb );



#endif
