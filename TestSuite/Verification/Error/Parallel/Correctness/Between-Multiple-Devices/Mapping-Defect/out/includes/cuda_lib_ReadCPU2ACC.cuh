
#ifndef CUDA_LIB_ReadCPU2ACC_CUH
#define CUDA_LIB_ReadCPU2ACC_CUH



__device__ uint randomizerGPU(int n = 42) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(n, tid, 0, &state);
    return curand(&state);
}

__global__ 
void kernel_cuda_wrapper_increment_pqMP2YoX9q(int32_t*  initial_Assa9vjb42, int32_t* result_8D0IqABlbo );



#endif
