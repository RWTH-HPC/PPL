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

void cuda_wrapper_sum_G67QyBpe5H(int32_t* initial_pb7pYYGUGI, int32_t* temp_data_Ie7GuQ5GJh, pthread_mutex_t reduction_lock_Ie7GuQ5GJh) {
	int32_t* temp_res;
	int32_t* d_temp_res;
	temp_res = Init_List(temp_res, 1);
	cudaMalloc(&d_temp_res, 1 * sizeof(int32_t));

	kernel_cuda_wrapper_sum_G67QyBpe5H<<<1, 168>>> (initial_pb7pYYGUGI, d_temp_res );
	cudaMemcpy(&temp_res[0], &d_temp_res[0], sizeof(int32_t) * 1, cudaMemcpyDeviceToHost);
	cudaFree(d_temp_res);
	pthread_mutex_lock(&reduction_lock_Ie7GuQ5GJh);
	temp_data_Ie7GuQ5GJh[0] = reduction_sum(temp_data_Ie7GuQ5GJh[0], temp_res, 1, 0);
	pthread_mutex_unlock(&reduction_lock_Ie7GuQ5GJh);
	std::free(temp_res);
}

__global__ 
void kernel_cuda_wrapper_sum_G67QyBpe5H(int32_t* initial_pb7pYYGUGI, int32_t* result_YbFznPajHi ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int id = threadIdx.x;
	__shared__ int32_t result_YbFznPajHi_s [168];
	result_YbFznPajHi_s[id] = 0;

	int exec_range = 0;
	if (tid < 100) {
		exec_range++;
	}
	for ( size_t range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_G67QyBpe5H = tid * exec_range + range_iterator + 100;
		if (tid < 100) {
			INDEX_G67QyBpe5H -= 100;
		}
		result_YbFznPajHi_s[id] += initial_pb7pYYGUGI[(INDEX_G67QyBpe5H)];
	}
	__syncthreads();

	if (168 >= 512) { if (id < 256) { result_YbFznPajHi_s[id] += result_YbFznPajHi_s[id + 256]; } __syncthreads(); }
	if (168 >= 256) { if (id < 128) { result_YbFznPajHi_s[id] += result_YbFznPajHi_s[id + 128]; } __syncthreads(); }
	if (168 >= 128) { if (id < 64) { result_YbFznPajHi_s[id] += result_YbFznPajHi_s[id + 64]; } __syncthreads(); }

	if ( id < 32) {
		if (168 >= 64) {result_YbFznPajHi_s[id] += result_YbFznPajHi_s[id + 32]; }
		__syncwarp();
		if (168 >= 32) {result_YbFznPajHi_s[id] += result_YbFznPajHi_s[id + 16]; }
		__syncwarp();
		if (168 >= 16) {result_YbFznPajHi_s[id] += result_YbFznPajHi_s[id + 8]; }
		__syncwarp();
		if (168 >= 8) {result_YbFznPajHi_s[id] += result_YbFznPajHi_s[id + 4]; }
		__syncwarp();
		if (168 >= 4) {result_YbFznPajHi_s[id] += result_YbFznPajHi_s[id + 2]; }
		__syncwarp();
		if (168 >= 2) {result_YbFznPajHi_s[id] += result_YbFznPajHi_s[id + 1]; }
	}
	__syncwarp();
	if ( id == 0) {
		if ( tid == 0) {
			for (size_t i = 128; i < 168; i++) {
				result_YbFznPajHi_s[0] += result_YbFznPajHi_s[i];
			}
		}
		result_YbFznPajHi[blockIdx.x] = result_YbFznPajHi_s[0];
	}
}

void cuda_wrapper_sum_ZeOzFvempS(int32_t* initial_pb7pYYGUGI, int32_t* temp_data_Ie7GuQ5GJh, pthread_mutex_t reduction_lock_Ie7GuQ5GJh) {
	int32_t* temp_res;
	int32_t* d_temp_res;
	temp_res = Init_List(temp_res, 1);
	cudaMalloc(&d_temp_res, 1 * sizeof(int32_t));

	kernel_cuda_wrapper_sum_ZeOzFvempS<<<1, 168>>> (initial_pb7pYYGUGI, d_temp_res );
	cudaMemcpy(&temp_res[0], &d_temp_res[0], sizeof(int32_t) * 1, cudaMemcpyDeviceToHost);
	cudaFree(d_temp_res);
	pthread_mutex_lock(&reduction_lock_Ie7GuQ5GJh);
	temp_data_Ie7GuQ5GJh[0] = reduction_sum(temp_data_Ie7GuQ5GJh[0], temp_res, 1, 0);
	pthread_mutex_unlock(&reduction_lock_Ie7GuQ5GJh);
	std::free(temp_res);
}

__global__ 
void kernel_cuda_wrapper_sum_ZeOzFvempS(int32_t* initial_pb7pYYGUGI, int32_t* result_YbFznPajHi ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int id = threadIdx.x;
	__shared__ int32_t result_YbFznPajHi_s [168];
	result_YbFznPajHi_s[id] = 0;

	int exec_range = 0;
	if (tid < 100) {
		exec_range++;
	}
	for ( size_t range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_ZeOzFvempS = tid * exec_range + range_iterator + 100;
		if (tid < 100) {
			INDEX_ZeOzFvempS -= 100;
		}
		result_YbFznPajHi_s[id] += initial_pb7pYYGUGI[(INDEX_ZeOzFvempS)];
	}
	__syncthreads();

	if (168 >= 512) { if (id < 256) { result_YbFznPajHi_s[id] += result_YbFznPajHi_s[id + 256]; } __syncthreads(); }
	if (168 >= 256) { if (id < 128) { result_YbFznPajHi_s[id] += result_YbFznPajHi_s[id + 128]; } __syncthreads(); }
	if (168 >= 128) { if (id < 64) { result_YbFznPajHi_s[id] += result_YbFznPajHi_s[id + 64]; } __syncthreads(); }

	if ( id < 32) {
		if (168 >= 64) {result_YbFznPajHi_s[id] += result_YbFznPajHi_s[id + 32]; }
		__syncwarp();
		if (168 >= 32) {result_YbFznPajHi_s[id] += result_YbFznPajHi_s[id + 16]; }
		__syncwarp();
		if (168 >= 16) {result_YbFznPajHi_s[id] += result_YbFznPajHi_s[id + 8]; }
		__syncwarp();
		if (168 >= 8) {result_YbFznPajHi_s[id] += result_YbFznPajHi_s[id + 4]; }
		__syncwarp();
		if (168 >= 4) {result_YbFznPajHi_s[id] += result_YbFznPajHi_s[id + 2]; }
		__syncwarp();
		if (168 >= 2) {result_YbFznPajHi_s[id] += result_YbFznPajHi_s[id + 1]; }
	}
	__syncwarp();
	if ( id == 0) {
		if ( tid == 0) {
			for (size_t i = 128; i < 168; i++) {
				result_YbFznPajHi_s[0] += result_YbFznPajHi_s[i];
			}
		}
		result_YbFznPajHi[blockIdx.x] = result_YbFznPajHi_s[0];
	}
}


