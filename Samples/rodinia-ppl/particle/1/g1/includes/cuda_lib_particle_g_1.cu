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
	float res_5g3ylNY1J1;
	res_5g3ylNY1J1 = 1;
	for ( int32_t i = 0; i < n; i++ ) {
		res_5g3ylNY1J1 *= x;
	}
		return res_5g3ylNY1J1;
}
int32_t roundDouble(double value) {
	int32_t newValue_xULy1ztDPc;
	newValue_xULy1ztDPc = Cast2Int(value);
	if ((value - newValue_xULy1ztDPc < 0.5)) {
				return newValue_xULy1ztDPc;
	} else {
				return newValue_xULy1ztDPc + 1;
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

void cuda_wrapper_u_init_zK3zsoJUg7(double* u1_UstTxnfY90_J45bILqR5e, int32_t* Nparticles_Xtno2sLxiz, double* u_uVrNRP6SnA ) {

	kernel_cuda_wrapper_u_init_zK3zsoJUg7<<<10, 64>>> (u1_UstTxnfY90_J45bILqR5e, Nparticles_Xtno2sLxiz, u_uVrNRP6SnA );
}

__global__ 
void kernel_cuda_wrapper_u_init_zK3zsoJUg7(double*  u1_UstTxnfY90_J45bILqR5e, int32_t*  Nparticles_Xtno2sLxiz, double* u_uVrNRP6SnA ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 15;
	if (tid < 400) {
		exec_range++;
	}
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_zK3zsoJUg7 = tid * exec_range + range_iterator + 400;
		if (tid < 400) {
			INDEX_zK3zsoJUg7 -= 400;
		}
		u_uVrNRP6SnA[(INDEX_zK3zsoJUg7)] = u1_UstTxnfY90_J45bILqR5e[0] + INDEX_zK3zsoJUg7 / Nparticles_Xtno2sLxiz[0];
	}
}

void cuda_wrapper_u_init_vlOhsxzeZk(double* u1_UstTxnfY90_J45bILqR5e, int32_t* Nparticles_Xtno2sLxiz, double* u_uVrNRP6SnA ) {

	kernel_cuda_wrapper_u_init_vlOhsxzeZk<<<10, 64>>> (u1_UstTxnfY90_J45bILqR5e, Nparticles_Xtno2sLxiz, u_uVrNRP6SnA );
}

__global__ 
void kernel_cuda_wrapper_u_init_vlOhsxzeZk(double*  u1_UstTxnfY90_J45bILqR5e, int32_t*  Nparticles_Xtno2sLxiz, double* u_uVrNRP6SnA ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 15;
	if (tid < 400) {
		exec_range++;
	}
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_vlOhsxzeZk = tid * exec_range + range_iterator + 400;
		if (tid < 400) {
			INDEX_vlOhsxzeZk -= 400;
		}
		u_uVrNRP6SnA[(INDEX_vlOhsxzeZk)] = u1_UstTxnfY90_J45bILqR5e[0] + INDEX_vlOhsxzeZk / Nparticles_Xtno2sLxiz[0];
	}
}


