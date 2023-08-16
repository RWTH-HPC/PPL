
#ifndef CUDA_LIB_DPTest_CUH
#define CUDA_LIB_DPTest_CUH



__device__ uint randomizerGPU(int n = 42) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(n, tid, 0, &state);
    return curand(&state);
}

__global__ 
void kernel_cuda_wrapper_incr_ektUFQFpO1(int32_t* initial_4q9liNyhoj, int32_t* result_O5snY1gmaG, int INDEX0_307FHaZTU5);



#endif
