
#ifndef CUDA_LIB_MapTest_CUH
#define CUDA_LIB_MapTest_CUH



__device__ uint randomizerGPU(int n = 42) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(n, tid, 0, &state);
    return curand(&state);
}

__global__ 
void kernel_cuda_wrapper_increment_PJDc73oaB9(int32_t*  initial_VRUERJiOcV, int32_t* result_jvmvgdiKj9 );

__global__ 
void kernel_cuda_wrapper_increment_nq9cxVBaee(int32_t*  initial_VRUERJiOcV, int32_t* result_jvmvgdiKj9 );



#endif
