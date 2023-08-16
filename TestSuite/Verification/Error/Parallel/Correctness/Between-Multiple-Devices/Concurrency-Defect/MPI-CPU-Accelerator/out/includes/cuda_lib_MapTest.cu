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

void cuda_wrapper_increment_6NO2GWP6jk(int32_t* initial_NHnfBBzrpN, int32_t* result_uvFP8uZgGK ) {

	kernel_cuda_wrapper_increment_6NO2GWP6jk<<<1, 168>>> (initial_NHnfBBzrpN, result_uvFP8uZgGK );
}

__global__ 
void kernel_cuda_wrapper_increment_6NO2GWP6jk(int32_t*  initial_NHnfBBzrpN, int32_t* result_uvFP8uZgGK ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 0;
	if (tid < 50) {
		exec_range++;
	}
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_6NO2GWP6jk = tid * exec_range + range_iterator + 50;
		if (tid < 50) {
			INDEX_6NO2GWP6jk -= 50;
		}
		result_uvFP8uZgGK[(INDEX_6NO2GWP6jk)] = initial_NHnfBBzrpN[(INDEX_6NO2GWP6jk)] + 1;
	}
}

void cuda_wrapper_increment_4KvgCV6d9e(int32_t* initial_NHnfBBzrpN, int32_t* result_uvFP8uZgGK ) {

	kernel_cuda_wrapper_increment_4KvgCV6d9e<<<1, 168>>> (initial_NHnfBBzrpN, result_uvFP8uZgGK );
}

__global__ 
void kernel_cuda_wrapper_increment_4KvgCV6d9e(int32_t*  initial_NHnfBBzrpN, int32_t* result_uvFP8uZgGK ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 0;
	if (tid < 50) {
		exec_range++;
	}
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_4KvgCV6d9e = tid * exec_range + range_iterator + 50;
		if (tid < 50) {
			INDEX_4KvgCV6d9e -= 50;
		}
		result_uvFP8uZgGK[(INDEX_4KvgCV6d9e)] = initial_NHnfBBzrpN[(INDEX_4KvgCV6d9e)] + 1;
	}
}


