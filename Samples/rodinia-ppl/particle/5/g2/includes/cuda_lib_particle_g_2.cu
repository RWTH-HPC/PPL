/**


This file implements the header for the Thread Pool and barrier implementation with PThreads.


*/
#include <stdio.h>
#include <stdlib.h>
#include "cuda_lib_particle_g_2.hxx"
#include "cuda_lib_particle_g_2.cuh"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include "Patternlib.hxx"


float powi(float x, int32_t n) {
	float res_VhtXd1nHXM;
	res_VhtXd1nHXM = 1;
	for ( int32_t i = 0; i < n; i++ ) {
		res_VhtXd1nHXM *= x;
	}
		return res_VhtXd1nHXM;
}
int32_t roundDouble(double value) {
	int32_t newValue_eGtZFfUMTO;
	newValue_eGtZFfUMTO = Cast2Int(value);
	if ((value - newValue_eGtZFfUMTO < 0.5)) {
				return newValue_eGtZFfUMTO;
	} else {
				return newValue_eGtZFfUMTO + 1;
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

void cuda_wrapper_u_init_r5ejBanMyg(int32_t* Nparticles_A3WboS1Ear, double* u1_z1OL8FprVl_IX1WUErsW5, double* u_PPkYFxALrb ) {

	kernel_cuda_wrapper_u_init_r5ejBanMyg<<<10, 64>>> (Nparticles_A3WboS1Ear, u1_z1OL8FprVl_IX1WUErsW5, u_PPkYFxALrb );
}

__global__ 
void kernel_cuda_wrapper_u_init_r5ejBanMyg(int32_t*  Nparticles_A3WboS1Ear, double*  u1_z1OL8FprVl_IX1WUErsW5, double* u_PPkYFxALrb ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 15;
	if (tid < 400) {
		exec_range++;
	}
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_r5ejBanMyg = tid * exec_range + range_iterator + 400;
		if (tid < 400) {
			INDEX_r5ejBanMyg -= 400;
		}
		u_PPkYFxALrb[(INDEX_r5ejBanMyg)] = u1_z1OL8FprVl_IX1WUErsW5[0] + INDEX_r5ejBanMyg / Nparticles_A3WboS1Ear[0];
	}
}

void cuda_wrapper_u_init_AcGUvJkq7y(double* u1_z1OL8FprVl_IX1WUErsW5, int32_t* Nparticles_A3WboS1Ear, double* u_PPkYFxALrb ) {

	kernel_cuda_wrapper_u_init_AcGUvJkq7y<<<10, 64>>> (u1_z1OL8FprVl_IX1WUErsW5, Nparticles_A3WboS1Ear, u_PPkYFxALrb );
}

__global__ 
void kernel_cuda_wrapper_u_init_AcGUvJkq7y(double*  u1_z1OL8FprVl_IX1WUErsW5, int32_t*  Nparticles_A3WboS1Ear, double* u_PPkYFxALrb ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 15;
	if (tid < 400) {
		exec_range++;
	}
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_AcGUvJkq7y = tid * exec_range + range_iterator + 400;
		if (tid < 400) {
			INDEX_AcGUvJkq7y -= 400;
		}
		u_PPkYFxALrb[(INDEX_AcGUvJkq7y)] = u1_z1OL8FprVl_IX1WUErsW5[0] + INDEX_AcGUvJkq7y / Nparticles_A3WboS1Ear[0];
	}
}

void cuda_wrapper_u_init_gCRghlciun(double* u1_z1OL8FprVl_IX1WUErsW5, int32_t* Nparticles_A3WboS1Ear, double* u_PPkYFxALrb ) {

	kernel_cuda_wrapper_u_init_gCRghlciun<<<10, 64>>> (u1_z1OL8FprVl_IX1WUErsW5, Nparticles_A3WboS1Ear, u_PPkYFxALrb );
}

__global__ 
void kernel_cuda_wrapper_u_init_gCRghlciun(double*  u1_z1OL8FprVl_IX1WUErsW5, int32_t*  Nparticles_A3WboS1Ear, double* u_PPkYFxALrb ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 15;
	if (tid < 400) {
		exec_range++;
	}
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_gCRghlciun = tid * exec_range + range_iterator + 400;
		if (tid < 400) {
			INDEX_gCRghlciun -= 400;
		}
		u_PPkYFxALrb[(INDEX_gCRghlciun)] = u1_z1OL8FprVl_IX1WUErsW5[0] + INDEX_gCRghlciun / Nparticles_A3WboS1Ear[0];
	}
}

void cuda_wrapper_u_init_g6JsLN2Lqm(int32_t* Nparticles_A3WboS1Ear, double* u1_z1OL8FprVl_IX1WUErsW5, double* u_PPkYFxALrb ) {

	kernel_cuda_wrapper_u_init_g6JsLN2Lqm<<<10, 64>>> (Nparticles_A3WboS1Ear, u1_z1OL8FprVl_IX1WUErsW5, u_PPkYFxALrb );
}

__global__ 
void kernel_cuda_wrapper_u_init_g6JsLN2Lqm(int32_t*  Nparticles_A3WboS1Ear, double*  u1_z1OL8FprVl_IX1WUErsW5, double* u_PPkYFxALrb ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 15;
	if (tid < 400) {
		exec_range++;
	}
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_g6JsLN2Lqm = tid * exec_range + range_iterator + 400;
		if (tid < 400) {
			INDEX_g6JsLN2Lqm -= 400;
		}
		u_PPkYFxALrb[(INDEX_g6JsLN2Lqm)] = u1_z1OL8FprVl_IX1WUErsW5[0] + INDEX_g6JsLN2Lqm / Nparticles_A3WboS1Ear[0];
	}
}


