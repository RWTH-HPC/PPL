/**


This file implements the header for the Thread Pool and barrier implementation with PThreads.


*/
#include <stdio.h>
#include <stdlib.h>
#include "cuda_lib_particle_g_1.hxx"
#include "cuda_lib_particle_g_1.cuh"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include "Patternlib.hxx"


float powi(float x, int32_t n) {
	float res_PuUbIfNc8X;
	res_PuUbIfNc8X = 1;
	for ( int32_t i = 0; i < n; i++ ) {
		res_PuUbIfNc8X *= x;
	}
		return res_PuUbIfNc8X;
}
int32_t roundDouble(double value) {
	int32_t newValue_F3e5Dfu58E;
	newValue_F3e5Dfu58E = Cast2Int(value);
	if ((value - newValue_F3e5Dfu58E < 0.5)) {
				return newValue_F3e5Dfu58E;
	} else {
				return newValue_F3e5Dfu58E + 1;
	}
}


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

void cuda_wrapper_u_init_zj9EbHdPzb(double* u1_ZZHMvQb2i6_s96P5Y86vI, int32_t* Nparticles_2f7BRjYtYI, double* u_x64gIli0uq ) {

	kernel_cuda_wrapper_u_init_zj9EbHdPzb<<<10, 64>>> (u1_ZZHMvQb2i6_s96P5Y86vI, Nparticles_2f7BRjYtYI, u_x64gIli0uq );
}

__global__ 
void kernel_cuda_wrapper_u_init_zj9EbHdPzb(double*  u1_ZZHMvQb2i6_s96P5Y86vI, int32_t*  Nparticles_2f7BRjYtYI, double* u_x64gIli0uq ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 15;
	if (tid < 400) {
		exec_range++;
	}
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_zj9EbHdPzb = tid * exec_range + range_iterator + 400;
		if (tid < 400) {
			INDEX_zj9EbHdPzb -= 400;
		}
		u_x64gIli0uq[(INDEX_zj9EbHdPzb)] = u1_ZZHMvQb2i6_s96P5Y86vI[0] + INDEX_zj9EbHdPzb / Nparticles_2f7BRjYtYI[0];
	}
}

void cuda_wrapper_u_init_U4raAdIpaG(double* u1_ZZHMvQb2i6_s96P5Y86vI, int32_t* Nparticles_2f7BRjYtYI, double* u_x64gIli0uq ) {

	kernel_cuda_wrapper_u_init_U4raAdIpaG<<<10, 64>>> (u1_ZZHMvQb2i6_s96P5Y86vI, Nparticles_2f7BRjYtYI, u_x64gIli0uq );
}

__global__ 
void kernel_cuda_wrapper_u_init_U4raAdIpaG(double*  u1_ZZHMvQb2i6_s96P5Y86vI, int32_t*  Nparticles_2f7BRjYtYI, double* u_x64gIli0uq ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 15;
	if (tid < 400) {
		exec_range++;
	}
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_U4raAdIpaG = tid * exec_range + range_iterator + 400;
		if (tid < 400) {
			INDEX_U4raAdIpaG -= 400;
		}
		u_x64gIli0uq[(INDEX_U4raAdIpaG)] = u1_ZZHMvQb2i6_s96P5Y86vI[0] + INDEX_U4raAdIpaG / Nparticles_2f7BRjYtYI[0];
	}
}


