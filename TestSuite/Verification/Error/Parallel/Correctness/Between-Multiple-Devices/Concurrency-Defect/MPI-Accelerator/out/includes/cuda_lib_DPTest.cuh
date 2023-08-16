
#ifndef CUDA_LIB_DPTest_CUH
#define CUDA_LIB_DPTest_CUH



__device__ uint randomizerGPU(int n = 42) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(n, tid, 0, &state);
    return curand(&state);
}

__global__ 
void kernel_cuda_wrapper_incr_3Gej3oHMpD(int32_t* initial_jyCGvVcxP3, int32_t* result_nEJrXOWu06, int INDEX0_dK2NhSbjni);

__global__ 
void kernel_cuda_wrapper_incr_gRtBZdqZMy(int32_t* initial_jyCGvVcxP3, int32_t* result_nEJrXOWu06, int INDEX0_dK2NhSbjni);



#endif
