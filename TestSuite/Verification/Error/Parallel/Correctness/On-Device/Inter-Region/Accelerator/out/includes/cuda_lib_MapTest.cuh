
#ifndef CUDA_LIB_MapTest_CUH
#define CUDA_LIB_MapTest_CUH



__device__ uint randomizerGPU(int n = 42) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(n, tid, 0, &state);
    return curand(&state);
}

__global__ 
void kernel_cuda_wrapper_increment_N0ebKTuSmh(int32_t*  initial_s6xJ8pOmvy, int32_t* result_UcMz61H54z );

__global__ 
void kernel_cuda_wrapper_increment_KZd7TM9SPp(int32_t*  initial_s6xJ8pOmvy, int32_t* result_UcMz61H54z );



#endif
