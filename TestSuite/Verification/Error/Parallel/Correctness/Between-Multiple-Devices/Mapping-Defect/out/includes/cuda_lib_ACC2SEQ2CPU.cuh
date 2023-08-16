
#ifndef CUDA_LIB_ACC2SEQ2CPU_CUH
#define CUDA_LIB_ACC2SEQ2CPU_CUH



__device__ uint randomizerGPU(int n = 42) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(n, tid, 0, &state);
    return curand(&state);
}

__global__ 
void kernel_cuda_wrapper_increment_l4eRRHe9xh(int32_t*  initial_KsBnEc0LWd, int32_t* intermediate_R0hHU019IF );



#endif
