
#ifndef CUDA_LIB_FlowCPU2ACC_CUH
#define CUDA_LIB_FlowCPU2ACC_CUH



__device__ uint randomizerGPU(int n = 42) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(n, tid, 0, &state);
    return curand(&state);
}

__global__ 
void kernel_cuda_wrapper_increment_A1ROJRS0uZ(int32_t*  intermediate_cigTaotlR0, int32_t* result_JLXTjYpslU );



#endif
