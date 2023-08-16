/**


This file implements the header for the Thread Pool and barrier implementation with PThreads.


*/
#include <stdio.h>
#include <stdlib.h>
#include "cuda_lib_ReduceTest.hxx"
#include "cuda_lib_ReduceTest.cuh"
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

void cuda_wrapper_sum_5X7iW6ACAt(int32_t* initial_1ROPqFElvp, int32_t* result_Bd5XrVmTLu ) {

	kernel_cuda_wrapper_sum_5X7iW6ACAt<<<1, 168>>> (initial_1ROPqFElvp, result_Bd5XrVmTLu );
	}

__global__ 
void kernel_cuda_wrapper_sum_5X7iW6ACAt(int32_t* initial_1ROPqFElvp, int32_t* result_Bd5XrVmTLu ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int id = threadIdx.x;
	__shared__ int32_t result_Bd5XrVmTLu_s [168];
	result_Bd5XrVmTLu_s[id] = 0;

	int exec_range = 1;
	if (tid < 32) {
		exec_range++;
	}
	for ( size_t range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_5X7iW6ACAt = tid * exec_range + range_iterator + 32;
		if (tid < 32) {
			INDEX_5X7iW6ACAt -= 32;
		}
		result_Bd5XrVmTLu_s[id] += initial_1ROPqFElvp[(INDEX_5X7iW6ACAt)];
	}
	__syncthreads();

	if (168 >= 512) { if (id < 256) { result_Bd5XrVmTLu_s[id] += result_Bd5XrVmTLu_s[id + 256]; } __syncthreads(); }
	if (168 >= 256) { if (id < 128) { result_Bd5XrVmTLu_s[id] += result_Bd5XrVmTLu_s[id + 128]; } __syncthreads(); }
	if (168 >= 128) { if (id < 64) { result_Bd5XrVmTLu_s[id] += result_Bd5XrVmTLu_s[id + 64]; } __syncthreads(); }

	if ( id < 32) {
		if (168 >= 64) {result_Bd5XrVmTLu_s[id] += result_Bd5XrVmTLu_s[id + 32]; }
		__syncwarp();
		if (168 >= 32) {result_Bd5XrVmTLu_s[id] += result_Bd5XrVmTLu_s[id + 16]; }
		__syncwarp();
		if (168 >= 16) {result_Bd5XrVmTLu_s[id] += result_Bd5XrVmTLu_s[id + 8]; }
		__syncwarp();
		if (168 >= 8) {result_Bd5XrVmTLu_s[id] += result_Bd5XrVmTLu_s[id + 4]; }
		__syncwarp();
		if (168 >= 4) {result_Bd5XrVmTLu_s[id] += result_Bd5XrVmTLu_s[id + 2]; }
		__syncwarp();
		if (168 >= 2) {result_Bd5XrVmTLu_s[id] += result_Bd5XrVmTLu_s[id + 1]; }
	}
	__syncwarp();
	if ( id == 0) {
		if ( tid == 0) {
			for (size_t i = 128; i < 168; i++) {
				result_Bd5XrVmTLu_s[0] += result_Bd5XrVmTLu_s[i];
			}
		}
		result_Bd5XrVmTLu[blockIdx.x] = result_Bd5XrVmTLu_s[0];
	}
}


