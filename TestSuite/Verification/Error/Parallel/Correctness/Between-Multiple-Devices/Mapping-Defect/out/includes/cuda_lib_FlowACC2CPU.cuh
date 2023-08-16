
#ifndef CUDA_LIB_FlowACC2CPU_CUH
#define CUDA_LIB_FlowACC2CPU_CUH



__device__ uint randomizerGPU(int n = 42) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(n, tid, 0, &state);
    return curand(&state);
}

__global__ 
void kernel_cuda_wrapper_increment_dFN25kCeJ7(int32_t*  initial_p2nXoGUtlC, int32_t* intermediate_j8keT4oZLx );



#endif
