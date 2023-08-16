/**


This file implements the header for the Thread Pool and barrier implementation with PThreads.


*/
#include <stdio.h>
#include <stdlib.h>
#include "cuda_lib_DPTest.hxx"
#include "cuda_lib_DPTest.cuh"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include "Patternlib.hxx"




template<typename T>
__global__
void cuda_reduce_sum(T* input, T* output, int n) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    sdata[tid] = input[tid];

    __syncthreads();

    if (n >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (n >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (n >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }

    if (tid < 32) {
        if (n >= 64) sdata[tid] += sdata[tid + 32];
        if (n >= 32) sdata[tid] += sdata[tid + 16];
        if (n >= 16) sdata[tid] += sdata[tid + 8];
        if (n >= 8) sdata[tid] += sdata[tid + 4];
        if (n >= 4) sdata[tid] += sdata[tid + 2];
        if (n >= 2) sdata[tid] += sdata[tid + 1];
    }

    if (tid == 0) output[0] = sdata[0];

}

template<typename T>
__global__
void cuda_reduce_times(T* input, T* output, int n) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    sdata[tid] = input[tid];

    __syncthreads();

    if (n >= 512) { if (tid < 256) { sdata[tid] *= sdata[tid + 256]; } __syncthreads(); }
    if (n >= 256) { if (tid < 128) { sdata[tid] *= sdata[tid + 128]; } __syncthreads(); }
    if (n >= 128) { if (tid < 64) { sdata[tid] *= sdata[tid + 64]; } __syncthreads(); }

    if (tid < 32) {
        if (n >= 64) sdata[tid] *= sdata[tid + 32];
        if (n >= 32) sdata[tid] *= sdata[tid + 16];
        if (n >= 16) sdata[tid] *= sdata[tid + 8];
        if (n >= 8) sdata[tid] *= sdata[tid + 4];
        if (n >= 4) sdata[tid] *= sdata[tid + 2];
        if (n >= 2) sdata[tid] *= sdata[tid + 1];
    }

    if (tid == 0) output[0] = sdata[0];

}

template<typename T>
__global__
void cuda_reduce_min(T* input, T* output, int n) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    sdata[tid] = input[tid];

    __syncthreads();

    if (n >= 512) { if (tid < 256) { sdata[tid] = min(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
    if (n >= 256) { if (tid < 128) { sdata[tid] = min(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
    if (n >= 128) { if (tid < 64) { sdata[tid] = min(sdata[tid], sdata[tid + 64]); } __syncthreads(); }

    if (tid < 32) {
        if (n >= 64) sdata[tid] = min(sdata[tid], sdata[tid + 32]);
        if (n >= 32) sdata[tid] = min(sdata[tid], sdata[tid + 16]);
        if (n >= 16) sdata[tid] = min(sdata[tid], sdata[tid + 8]);
        if (n >= 8) sdata[tid] = min(sdata[tid], sdata[tid + 4]);
        if (n >= 4) sdata[tid] = min(sdata[tid], sdata[tid + 2]);
        if (n >= 2) sdata[tid] = min(sdata[tid], sdata[tid + 1]);
    }

    if (tid == 0) output[0] = sdata[0];

}

template<typename T>
__global__
void cuda_reduce_max(T* input, T* output, int n) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    sdata[tid] = input[tid];

    __syncthreads();

    if (n >= 512) { if (tid < 256) { sdata[tid] = max(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
    if (n >= 256) { if (tid < 128) { sdata[tid] = max(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
    if (n >= 128) { if (tid < 64) { sdata[tid] = max(sdata[tid], sdata[tid + 64]); } __syncthreads(); }

    if (tid < 32) {
        if (n >= 64) sdata[tid] = max(sdata[tid], sdata[tid + 32]);
        if (n >= 32) sdata[tid] = max(sdata[tid], sdata[tid + 16]);
        if (n >= 16) sdata[tid] = max(sdata[tid], sdata[tid + 8]);
        if (n >= 8) sdata[tid] = max(sdata[tid], sdata[tid + 4]);
        if (n >= 4) sdata[tid] = max(sdata[tid], sdata[tid + 2]);
        if (n >= 2) sdata[tid] = max(sdata[tid], sdata[tid + 1]);
    }

    if (tid == 0) output[0] = sdata[0];

}

void cuda_wrapper_incr_jCr8Yz9sSg(int32_t* initial_jxAAR4OqHM, int32_t* result_25ndUhg8ZL, int INDEX0_habLccSPNH) {

	kernel_cuda_wrapper_incr_jCr8Yz9sSg<<<1, 168>>> (initial_jxAAR4OqHM, result_25ndUhg8ZL, INDEX0_habLccSPNH);
}

__global__ 
void kernel_cuda_wrapper_incr_jCr8Yz9sSg(int32_t* initial_jxAAR4OqHM, int32_t* result_25ndUhg8ZL, int INDEX0_habLccSPNH) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 0;
	if (tid < 100) {
		exec_range++;
	}
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX1_habLccSPNH = tid * exec_range + range_iterator + 100;
		if (tid < 100) {
			INDEX1_habLccSPNH -= 100;
		}
		result_25ndUhg8ZL[(INDEX1_habLccSPNH)] = initial_jxAAR4OqHM[(INDEX1_habLccSPNH)] + 1;
	}
}

void cuda_wrapper_incr_hVGVGWenmY(int32_t* initial_jxAAR4OqHM, int32_t* result_25ndUhg8ZL, int INDEX0_habLccSPNH) {

	kernel_cuda_wrapper_incr_hVGVGWenmY<<<1, 168>>> (initial_jxAAR4OqHM, result_25ndUhg8ZL, INDEX0_habLccSPNH);
}

__global__ 
void kernel_cuda_wrapper_incr_hVGVGWenmY(int32_t* initial_jxAAR4OqHM, int32_t* result_25ndUhg8ZL, int INDEX0_habLccSPNH) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 0;
	if (tid < 100) {
		exec_range++;
	}
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX1_habLccSPNH = tid * exec_range + range_iterator + 100;
		if (tid < 100) {
			INDEX1_habLccSPNH -= 100;
		}
		result_25ndUhg8ZL[(INDEX1_habLccSPNH)] = initial_jxAAR4OqHM[(INDEX1_habLccSPNH)] + 1;
	}
}


