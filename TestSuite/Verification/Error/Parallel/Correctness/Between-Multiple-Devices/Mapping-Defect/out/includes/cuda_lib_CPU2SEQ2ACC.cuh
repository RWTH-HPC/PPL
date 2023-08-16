
#ifndef CUDA_LIB_CPU2SEQ2ACC_CUH
#define CUDA_LIB_CPU2SEQ2ACC_CUH



__device__ uint randomizerGPU(int n = 42) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(n, tid, 0, &state);
    return curand(&state);
}

__global__ 
void kernel_cuda_wrapper_increment_88tU14gDpz(int32_t*  intermediate_PCMJVl88HJ, int32_t* result_lavXINJeC1 );



#endif
