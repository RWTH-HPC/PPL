
#ifndef CUDA_LIB_DPTest_CUH
#define CUDA_LIB_DPTest_CUH



__device__ uint randomizerGPU(int n = 42) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(n, tid, 0, &state);
    return curand(&state);
}

__global__ 
void kernel_cuda_wrapper_incr_Um8n4qBVfX(int32_t* initial_QML7k7jnuO, int32_t* result_W9UGyjNjLb, int INDEX0_oGtVqnLsp9);



#endif
