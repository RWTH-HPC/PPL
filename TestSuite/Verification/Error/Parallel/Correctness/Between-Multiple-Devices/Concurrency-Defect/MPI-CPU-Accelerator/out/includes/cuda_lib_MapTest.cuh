
#ifndef CUDA_LIB_MapTest_CUH
#define CUDA_LIB_MapTest_CUH



__device__ uint randomizerGPU(int n = 42) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(n, tid, 0, &state);
    return curand(&state);
}

__global__ 
void kernel_cuda_wrapper_increment_6NO2GWP6jk(int32_t*  initial_NHnfBBzrpN, int32_t* result_uvFP8uZgGK );

__global__ 
void kernel_cuda_wrapper_increment_4KvgCV6d9e(int32_t*  initial_NHnfBBzrpN, int32_t* result_uvFP8uZgGK );



#endif
