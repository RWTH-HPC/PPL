
#ifndef CUDA_LIB_DPTest_CUH
#define CUDA_LIB_DPTest_CUH



__device__ uint randomizerGPU(int n = 42) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(n, tid, 0, &state);
    return curand(&state);
}

__global__ 
void kernel_cuda_wrapper_incr_tGTQDtaA7M(int32_t* initial_nUIfy29ORQ, int32_t* result_8jvxK8OViz, int INDEX0_t2MUMiZdzW);

__global__ 
void kernel_cuda_wrapper_incr_pfxq6fvbpp(int32_t* initial_nUIfy29ORQ, int32_t* result_8jvxK8OViz, int INDEX0_t2MUMiZdzW);



#endif
