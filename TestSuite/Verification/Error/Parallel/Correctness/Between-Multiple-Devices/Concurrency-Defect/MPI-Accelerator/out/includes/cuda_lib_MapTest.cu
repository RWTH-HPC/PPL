/**


This file implements the header for the Thread Pool and barrier implementation with PThreads.


*/
#include <stdio.h>
#include <stdlib.h>
#include "cuda_lib_MapTest.hxx"
#include "cuda_lib_MapTest.cuh"
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

void cuda_wrapper_increment_a0V8c3egsK(int32_t* initial_q9zXNRkOQa, int32_t* result_lePGdKBCEb ) {

	kernel_cuda_wrapper_increment_a0V8c3egsK<<<1, 168>>> (initial_q9zXNRkOQa, result_lePGdKBCEb );
}

__global__ 
void kernel_cuda_wrapper_increment_a0V8c3egsK(int32_t*  initial_q9zXNRkOQa, int32_t* result_lePGdKBCEb ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 0;
	if (tid < 100) {
		exec_range++;
	}
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_a0V8c3egsK = tid * exec_range + range_iterator + 100;
		if (tid < 100) {
			INDEX_a0V8c3egsK -= 100;
		}
		result_lePGdKBCEb[(INDEX_a0V8c3egsK)] = initial_q9zXNRkOQa[(INDEX_a0V8c3egsK)] + 1;
	}
}

void cuda_wrapper_increment_NcekzbsFgR(int32_t* initial_q9zXNRkOQa, int32_t* result_lePGdKBCEb ) {

	kernel_cuda_wrapper_increment_NcekzbsFgR<<<1, 168>>> (initial_q9zXNRkOQa, result_lePGdKBCEb );
}

__global__ 
void kernel_cuda_wrapper_increment_NcekzbsFgR(int32_t*  initial_q9zXNRkOQa, int32_t* result_lePGdKBCEb ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 0;
	if (tid < 100) {
		exec_range++;
	}
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_NcekzbsFgR = tid * exec_range + range_iterator + 100;
		if (tid < 100) {
			INDEX_NcekzbsFgR -= 100;
		}
		result_lePGdKBCEb[(INDEX_NcekzbsFgR)] = initial_q9zXNRkOQa[(INDEX_NcekzbsFgR)] + 1;
	}
}


