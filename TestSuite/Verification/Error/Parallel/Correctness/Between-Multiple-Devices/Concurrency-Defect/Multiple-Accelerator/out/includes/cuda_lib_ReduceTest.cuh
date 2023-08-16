
#ifndef CUDA_LIB_ReduceTest_CUH
#define CUDA_LIB_ReduceTest_CUH



__device__ uint randomizerGPU(int n = 42) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(n, tid, 0, &state);
    return curand(&state);
}

__global__ 
void kernel_cuda_wrapper_sum_R1Bgbztek8(int32_t* initial_cXL2qc3HGa, int32_t* result_LiIf2JBnYd );

__global__ 
void kernel_cuda_wrapper_sum_oarWzvni2b(int32_t* initial_cXL2qc3HGa, int32_t* result_LiIf2JBnYd );



#endif
