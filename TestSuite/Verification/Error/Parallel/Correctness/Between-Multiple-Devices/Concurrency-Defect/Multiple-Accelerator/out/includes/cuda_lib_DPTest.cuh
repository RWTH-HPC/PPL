
#ifndef CUDA_LIB_DPTest_CUH
#define CUDA_LIB_DPTest_CUH



__device__ uint randomizerGPU(int n = 42) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(n, tid, 0, &state);
    return curand(&state);
}

__global__ 
void kernel_cuda_wrapper_incr_jCr8Yz9sSg(int32_t* initial_jxAAR4OqHM, int32_t* result_25ndUhg8ZL, int INDEX0_habLccSPNH);

__global__ 
void kernel_cuda_wrapper_incr_hVGVGWenmY(int32_t* initial_jxAAR4OqHM, int32_t* result_25ndUhg8ZL, int INDEX0_habLccSPNH);



#endif
