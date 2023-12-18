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
	float res_X3c53UA175;
	res_X3c53UA175 = 1;
	for ( int32_t i = 0; i < n; i++ ) {
		res_X3c53UA175 *= x;
	}
		return res_X3c53UA175;
}
int32_t roundDouble(double value) {
	int32_t newValue_VAtY8WEy1c;
	newValue_VAtY8WEy1c = Cast2Int(value);
	if ((value - newValue_VAtY8WEy1c < 0.5)) {
				return newValue_VAtY8WEy1c;
	} else {
				return newValue_VAtY8WEy1c + 1;
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

void cuda_wrapper_u_init_ByR3Bpa2y2(double* u1_bu8N2svs8Q_0lHZcF0TIO, int32_t* Nparticles_ftpCX5UbUk, double* u_75B8MOd4dN ) {

	kernel_cuda_wrapper_u_init_ByR3Bpa2y2<<<10, 64>>> (u1_bu8N2svs8Q_0lHZcF0TIO, Nparticles_ftpCX5UbUk, u_75B8MOd4dN );
}

__global__ 
void kernel_cuda_wrapper_u_init_ByR3Bpa2y2(double*  u1_bu8N2svs8Q_0lHZcF0TIO, int32_t*  Nparticles_ftpCX5UbUk, double* u_75B8MOd4dN ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 15;
	if (tid < 400) {
		exec_range++;
	}
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_ByR3Bpa2y2 = tid * exec_range + range_iterator + 400;
		if (tid < 400) {
			INDEX_ByR3Bpa2y2 -= 400;
		}
		u_75B8MOd4dN[(INDEX_ByR3Bpa2y2)] = u1_bu8N2svs8Q_0lHZcF0TIO[0] + INDEX_ByR3Bpa2y2 / Nparticles_ftpCX5UbUk[0];
	}
}

void cuda_wrapper_u_init_7qezuspv9u(double* u1_bu8N2svs8Q_0lHZcF0TIO, int32_t* Nparticles_ftpCX5UbUk, double* u_75B8MOd4dN ) {

	kernel_cuda_wrapper_u_init_7qezuspv9u<<<10, 64>>> (u1_bu8N2svs8Q_0lHZcF0TIO, Nparticles_ftpCX5UbUk, u_75B8MOd4dN );
}

__global__ 
void kernel_cuda_wrapper_u_init_7qezuspv9u(double*  u1_bu8N2svs8Q_0lHZcF0TIO, int32_t*  Nparticles_ftpCX5UbUk, double* u_75B8MOd4dN ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 15;
	if (tid < 400) {
		exec_range++;
	}
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_7qezuspv9u = tid * exec_range + range_iterator + 400;
		if (tid < 400) {
			INDEX_7qezuspv9u -= 400;
		}
		u_75B8MOd4dN[(INDEX_7qezuspv9u)] = u1_bu8N2svs8Q_0lHZcF0TIO[0] + INDEX_7qezuspv9u / Nparticles_ftpCX5UbUk[0];
	}
}


