/**


This file implements the header for the Thread Pool and barrier implementation with PThreads.


*/
#include <stdio.h>
#include <stdlib.h>
#include "cuda_lib_StencilTest.hxx"
#include "cuda_lib_StencilTest.cuh"
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

void cuda_wrapper_sum_rCQOyszoeH(int32_t* initial_UANbCb8XQf, int32_t* result_zQ0Xj2P8Lz ) {

	kernel_cuda_wrapper_sum_rCQOyszoeH<<<1, 168>>> (initial_UANbCb8XQf, result_zQ0Xj2P8Lz );
}

__global__ 
void kernel_cuda_wrapper_sum_rCQOyszoeH(int32_t* initial_UANbCb8XQf, int32_t* result_zQ0Xj2P8Lz ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 0;
	if (tid < 99) {
		exec_range++;
	}
	for ( size_t range_iterator = 1; range_iterator < exec_range + 1; range_iterator++) {
		int INDEX0_rCQOyszoeH = tid * exec_range + range_iterator + 99;
		if (tid < 99) {
			INDEX0_rCQOyszoeH -= 99;
		}
		for (size_t INDEX1_rCQOyszoeH = 1; INDEX1_rCQOyszoeH < 1 + 98; ++INDEX1_rCQOyszoeH) {
			for (size_t INDEX2_rCQOyszoeH = 1; INDEX2_rCQOyszoeH < 1 + 98; ++INDEX2_rCQOyszoeH) {
				result_zQ0Xj2P8Lz[100LL * 100LL * (INDEX0_rCQOyszoeH) + 100LL * (INDEX1_rCQOyszoeH) + (INDEX2_rCQOyszoeH)] = initial_UANbCb8XQf[100LL * 100LL * (INDEX0_rCQOyszoeH + 1) + 100LL * (INDEX1_rCQOyszoeH) + (INDEX2_rCQOyszoeH)] + initial_UANbCb8XQf[100LL * 100LL * (INDEX0_rCQOyszoeH) + 100LL * (INDEX1_rCQOyszoeH + 1) + (INDEX2_rCQOyszoeH)] + initial_UANbCb8XQf[100LL * 100LL * (INDEX0_rCQOyszoeH) + 100LL * (INDEX1_rCQOyszoeH) + (INDEX2_rCQOyszoeH + 1)] + initial_UANbCb8XQf[100LL * 100LL * (INDEX0_rCQOyszoeH - 1) + 100LL * (INDEX1_rCQOyszoeH) + (INDEX2_rCQOyszoeH)] + initial_UANbCb8XQf[100LL * 100LL * (INDEX0_rCQOyszoeH) + 100LL * (INDEX1_rCQOyszoeH - 1) + (INDEX2_rCQOyszoeH)] + initial_UANbCb8XQf[100LL * 100LL * (INDEX0_rCQOyszoeH) + 100LL * (INDEX1_rCQOyszoeH) + (INDEX2_rCQOyszoeH - 1)] + initial_UANbCb8XQf[100LL * 100LL * (INDEX0_rCQOyszoeH) + 100LL * (INDEX1_rCQOyszoeH) + (INDEX2_rCQOyszoeH)];
			}
		}
	}
}

void cuda_wrapper_sum_bYUinBGbIN(int32_t* initial_UANbCb8XQf, int32_t* result_zQ0Xj2P8Lz ) {

	kernel_cuda_wrapper_sum_bYUinBGbIN<<<1, 168>>> (initial_UANbCb8XQf, result_zQ0Xj2P8Lz );
}

__global__ 
void kernel_cuda_wrapper_sum_bYUinBGbIN(int32_t* initial_UANbCb8XQf, int32_t* result_zQ0Xj2P8Lz ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 0;
	if (tid < 99) {
		exec_range++;
	}
	for ( size_t range_iterator = 1; range_iterator < exec_range + 1; range_iterator++) {
		int INDEX0_bYUinBGbIN = tid * exec_range + range_iterator + 99;
		if (tid < 99) {
			INDEX0_bYUinBGbIN -= 99;
		}
		for (size_t INDEX1_bYUinBGbIN = 1; INDEX1_bYUinBGbIN < 1 + 98; ++INDEX1_bYUinBGbIN) {
			for (size_t INDEX2_bYUinBGbIN = 1; INDEX2_bYUinBGbIN < 1 + 98; ++INDEX2_bYUinBGbIN) {
				result_zQ0Xj2P8Lz[100LL * 100LL * (INDEX0_bYUinBGbIN) + 100LL * (INDEX1_bYUinBGbIN) + (INDEX2_bYUinBGbIN)] = initial_UANbCb8XQf[100LL * 100LL * (INDEX0_bYUinBGbIN + 1) + 100LL * (INDEX1_bYUinBGbIN) + (INDEX2_bYUinBGbIN)] + initial_UANbCb8XQf[100LL * 100LL * (INDEX0_bYUinBGbIN) + 100LL * (INDEX1_bYUinBGbIN + 1) + (INDEX2_bYUinBGbIN)] + initial_UANbCb8XQf[100LL * 100LL * (INDEX0_bYUinBGbIN) + 100LL * (INDEX1_bYUinBGbIN) + (INDEX2_bYUinBGbIN + 1)] + initial_UANbCb8XQf[100LL * 100LL * (INDEX0_bYUinBGbIN - 1) + 100LL * (INDEX1_bYUinBGbIN) + (INDEX2_bYUinBGbIN)] + initial_UANbCb8XQf[100LL * 100LL * (INDEX0_bYUinBGbIN) + 100LL * (INDEX1_bYUinBGbIN - 1) + (INDEX2_bYUinBGbIN)] + initial_UANbCb8XQf[100LL * 100LL * (INDEX0_bYUinBGbIN) + 100LL * (INDEX1_bYUinBGbIN) + (INDEX2_bYUinBGbIN - 1)] + initial_UANbCb8XQf[100LL * 100LL * (INDEX0_bYUinBGbIN) + 100LL * (INDEX1_bYUinBGbIN) + (INDEX2_bYUinBGbIN)];
			}
		}
	}
}


