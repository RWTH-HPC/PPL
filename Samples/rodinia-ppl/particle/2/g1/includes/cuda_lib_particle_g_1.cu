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
	float res_Sb9mH6tSYb;
	res_Sb9mH6tSYb = 1;
	for ( int32_t i = 0; i < n; i++ ) {
		res_Sb9mH6tSYb *= x;
	}
		return res_Sb9mH6tSYb;
}
int32_t roundDouble(double value) {
	int32_t newValue_eXdfc8dqHD;
	newValue_eXdfc8dqHD = Cast2Int(value);
	if ((value - newValue_eXdfc8dqHD < 0.5)) {
				return newValue_eXdfc8dqHD;
	} else {
				return newValue_eXdfc8dqHD + 1;
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

void cuda_wrapper_u_init_eC5NGPiAvv(double* u1_RHKVSgjzlz_N4IMHoZdrq, int32_t* Nparticles_RMBeajvFXr, double* u_NAIj3hYPTU ) {

	kernel_cuda_wrapper_u_init_eC5NGPiAvv<<<10, 64>>> (u1_RHKVSgjzlz_N4IMHoZdrq, Nparticles_RMBeajvFXr, u_NAIj3hYPTU );
}

__global__ 
void kernel_cuda_wrapper_u_init_eC5NGPiAvv(double*  u1_RHKVSgjzlz_N4IMHoZdrq, int32_t*  Nparticles_RMBeajvFXr, double* u_NAIj3hYPTU ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 15;
	if (tid < 400) {
		exec_range++;
	}
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_eC5NGPiAvv = tid * exec_range + range_iterator + 400;
		if (tid < 400) {
			INDEX_eC5NGPiAvv -= 400;
		}
		u_NAIj3hYPTU[(INDEX_eC5NGPiAvv)] = u1_RHKVSgjzlz_N4IMHoZdrq[0] + INDEX_eC5NGPiAvv / Nparticles_RMBeajvFXr[0];
	}
}

void cuda_wrapper_u_init_FFnUAftOOH(double* u1_RHKVSgjzlz_N4IMHoZdrq, int32_t* Nparticles_RMBeajvFXr, double* u_NAIj3hYPTU ) {

	kernel_cuda_wrapper_u_init_FFnUAftOOH<<<10, 64>>> (u1_RHKVSgjzlz_N4IMHoZdrq, Nparticles_RMBeajvFXr, u_NAIj3hYPTU );
}

__global__ 
void kernel_cuda_wrapper_u_init_FFnUAftOOH(double*  u1_RHKVSgjzlz_N4IMHoZdrq, int32_t*  Nparticles_RMBeajvFXr, double* u_NAIj3hYPTU ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 15;
	if (tid < 400) {
		exec_range++;
	}
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_FFnUAftOOH = tid * exec_range + range_iterator + 400;
		if (tid < 400) {
			INDEX_FFnUAftOOH -= 400;
		}
		u_NAIj3hYPTU[(INDEX_FFnUAftOOH)] = u1_RHKVSgjzlz_N4IMHoZdrq[0] + INDEX_FFnUAftOOH / Nparticles_RMBeajvFXr[0];
	}
}


