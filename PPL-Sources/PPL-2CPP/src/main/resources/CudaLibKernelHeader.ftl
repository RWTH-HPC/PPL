<#assign gh=glex.getGlobalVar("pfHelper")> <#-- GeneratorHelper -->

#ifndef CUDA_LIB_${gh.getFilename()}_CUH
#define CUDA_LIB_${gh.getFilename()}_CUH



__device__ uint randomizerGPU(int n = 42) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(n, tid, 0, &state);
    return curand(&state);
}

${gh.printCudaKernelHeader()}

#endif
