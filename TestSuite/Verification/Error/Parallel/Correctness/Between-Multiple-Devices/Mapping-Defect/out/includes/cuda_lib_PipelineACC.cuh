
#ifndef CUDA_LIB_PipelineACC_CUH
#define CUDA_LIB_PipelineACC_CUH



__device__ uint randomizerGPU(int n = 42) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(n, tid, 0, &state);
    return curand(&state);
}

__global__ 
void kernel_cuda_wrapper_increment_2XlMsn6B8K(int32_t*  initial_Pfe9Oio93E, int32_t* intermediate_ZFsDXzWPC8 );

__global__ 
void kernel_cuda_wrapper_increment_xxGyRu4JSm(int32_t*  intermediate_ZFsDXzWPC8, int32_t* result_ZMrKKHa9pD );



#endif
