
#ifndef CUDA_LIB_ACC2SEQ2ACC_CUH
#define CUDA_LIB_ACC2SEQ2ACC_CUH



__device__ uint randomizerGPU(int n = 42) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(n, tid, 0, &state);
    return curand(&state);
}

__global__ 
void kernel_cuda_wrapper_increment_W2KH8AD4Ma(int32_t*  initial_8GPdo0d1W0, int32_t* intermediate_rKNPOUzEou );

__global__ 
void kernel_cuda_wrapper_increment_2COEAgXFgU(int32_t*  intermediate_rKNPOUzEou, int32_t* result_VJcCRyR3D8 );



#endif
