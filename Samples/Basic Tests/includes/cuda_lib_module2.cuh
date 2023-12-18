
#ifndef CUDA_LIB_module2_CUH
#define CUDA_LIB_module2_CUH



__device__ uint randomizerGPU(int n = 42) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(n, tid, 0, &state);
    return curand(&state);
}

__global__ 
void kernel_cuda_wrapper_increment_gBgKyNg8IY(int32_t*  initial_e438q0bUxc, int32_t* result_EGQknbj5uv );



#endif
