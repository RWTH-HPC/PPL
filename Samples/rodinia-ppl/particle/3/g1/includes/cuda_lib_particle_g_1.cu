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
	float res_r9Q0GkosI0;
	res_r9Q0GkosI0 = 1;
	for ( int32_t i = 0; i < n; i++ ) {
		res_r9Q0GkosI0 *= x;
	}
		return res_r9Q0GkosI0;
}
int32_t roundDouble(double value) {
	int32_t newValue_OoXNCjaqMz;
	newValue_OoXNCjaqMz = Cast2Int(value);
	if ((value - newValue_OoXNCjaqMz < 0.5)) {
				return newValue_OoXNCjaqMz;
	} else {
				return newValue_OoXNCjaqMz + 1;
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

void cuda_wrapper_u_init_2c04MpmCvE(double* u1_eTgP6Ccpm7_8DIbDc4tdI, int32_t* Nparticles_aBFhrb8G1G, double* u_s8LjziQsQp ) {

	kernel_cuda_wrapper_u_init_2c04MpmCvE<<<10, 64>>> (u1_eTgP6Ccpm7_8DIbDc4tdI, Nparticles_aBFhrb8G1G, u_s8LjziQsQp );
}

__global__ 
void kernel_cuda_wrapper_u_init_2c04MpmCvE(double*  u1_eTgP6Ccpm7_8DIbDc4tdI, int32_t*  Nparticles_aBFhrb8G1G, double* u_s8LjziQsQp ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 15;
	if (tid < 400) {
		exec_range++;
	}
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_2c04MpmCvE = tid * exec_range + range_iterator + 400;
		if (tid < 400) {
			INDEX_2c04MpmCvE -= 400;
		}
		u_s8LjziQsQp[(INDEX_2c04MpmCvE)] = u1_eTgP6Ccpm7_8DIbDc4tdI[0] + INDEX_2c04MpmCvE / Nparticles_aBFhrb8G1G[0];
	}
}

void cuda_wrapper_u_init_7bIvnL5XRN(double* u1_eTgP6Ccpm7_8DIbDc4tdI, int32_t* Nparticles_aBFhrb8G1G, double* u_s8LjziQsQp ) {

	kernel_cuda_wrapper_u_init_7bIvnL5XRN<<<10, 64>>> (u1_eTgP6Ccpm7_8DIbDc4tdI, Nparticles_aBFhrb8G1G, u_s8LjziQsQp );
}

__global__ 
void kernel_cuda_wrapper_u_init_7bIvnL5XRN(double*  u1_eTgP6Ccpm7_8DIbDc4tdI, int32_t*  Nparticles_aBFhrb8G1G, double* u_s8LjziQsQp ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 15;
	if (tid < 400) {
		exec_range++;
	}
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_7bIvnL5XRN = tid * exec_range + range_iterator + 400;
		if (tid < 400) {
			INDEX_7bIvnL5XRN -= 400;
		}
		u_s8LjziQsQp[(INDEX_7bIvnL5XRN)] = u1_eTgP6Ccpm7_8DIbDc4tdI[0] + INDEX_7bIvnL5XRN / Nparticles_aBFhrb8G1G[0];
	}
}


