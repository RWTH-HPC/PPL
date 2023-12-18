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
	float res_ZaDZXN8bU7;
	res_ZaDZXN8bU7 = 1;
	for ( int32_t i = 0; i < n; i++ ) {
		res_ZaDZXN8bU7 *= x;
	}
		return res_ZaDZXN8bU7;
}
int32_t roundDouble(double value) {
	int32_t newValue_ZfCXdWYvtz;
	newValue_ZfCXdWYvtz = Cast2Int(value);
	if ((value - newValue_ZfCXdWYvtz < 0.5)) {
				return newValue_ZfCXdWYvtz;
	} else {
				return newValue_ZfCXdWYvtz + 1;
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

void cuda_wrapper_u_init_EuFs5tKDLa(int32_t* Nparticles_GEZW4pzeoW, double* u1_cRx97fdDWf_s2dfuBCP2s, double* u_bceK7yP7v7 ) {

	kernel_cuda_wrapper_u_init_EuFs5tKDLa<<<10, 64>>> (Nparticles_GEZW4pzeoW, u1_cRx97fdDWf_s2dfuBCP2s, u_bceK7yP7v7 );
}

__global__ 
void kernel_cuda_wrapper_u_init_EuFs5tKDLa(int32_t*  Nparticles_GEZW4pzeoW, double*  u1_cRx97fdDWf_s2dfuBCP2s, double* u_bceK7yP7v7 ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 15;
	if (tid < 400) {
		exec_range++;
	}
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_EuFs5tKDLa = tid * exec_range + range_iterator + 400;
		if (tid < 400) {
			INDEX_EuFs5tKDLa -= 400;
		}
		u_bceK7yP7v7[(INDEX_EuFs5tKDLa)] = u1_cRx97fdDWf_s2dfuBCP2s[0] + INDEX_EuFs5tKDLa / Nparticles_GEZW4pzeoW[0];
	}
}

void cuda_wrapper_u_init_gkZtexcP2p(double* u1_cRx97fdDWf_s2dfuBCP2s, int32_t* Nparticles_GEZW4pzeoW, double* u_bceK7yP7v7 ) {

	kernel_cuda_wrapper_u_init_gkZtexcP2p<<<10, 64>>> (u1_cRx97fdDWf_s2dfuBCP2s, Nparticles_GEZW4pzeoW, u_bceK7yP7v7 );
}

__global__ 
void kernel_cuda_wrapper_u_init_gkZtexcP2p(double*  u1_cRx97fdDWf_s2dfuBCP2s, int32_t*  Nparticles_GEZW4pzeoW, double* u_bceK7yP7v7 ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 15;
	if (tid < 400) {
		exec_range++;
	}
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_gkZtexcP2p = tid * exec_range + range_iterator + 400;
		if (tid < 400) {
			INDEX_gkZtexcP2p -= 400;
		}
		u_bceK7yP7v7[(INDEX_gkZtexcP2p)] = u1_cRx97fdDWf_s2dfuBCP2s[0] + INDEX_gkZtexcP2p / Nparticles_GEZW4pzeoW[0];
	}
}

void cuda_wrapper_u_init_3SBzj8X8zZ(double* u1_cRx97fdDWf_s2dfuBCP2s, int32_t* Nparticles_GEZW4pzeoW, double* u_bceK7yP7v7 ) {

	kernel_cuda_wrapper_u_init_3SBzj8X8zZ<<<10, 64>>> (u1_cRx97fdDWf_s2dfuBCP2s, Nparticles_GEZW4pzeoW, u_bceK7yP7v7 );
}

__global__ 
void kernel_cuda_wrapper_u_init_3SBzj8X8zZ(double*  u1_cRx97fdDWf_s2dfuBCP2s, int32_t*  Nparticles_GEZW4pzeoW, double* u_bceK7yP7v7 ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 15;
	if (tid < 400) {
		exec_range++;
	}
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_3SBzj8X8zZ = tid * exec_range + range_iterator + 400;
		if (tid < 400) {
			INDEX_3SBzj8X8zZ -= 400;
		}
		u_bceK7yP7v7[(INDEX_3SBzj8X8zZ)] = u1_cRx97fdDWf_s2dfuBCP2s[0] + INDEX_3SBzj8X8zZ / Nparticles_GEZW4pzeoW[0];
	}
}

void cuda_wrapper_u_init_zYHeikJnbS(int32_t* Nparticles_GEZW4pzeoW, double* u1_cRx97fdDWf_s2dfuBCP2s, double* u_bceK7yP7v7 ) {

	kernel_cuda_wrapper_u_init_zYHeikJnbS<<<10, 64>>> (Nparticles_GEZW4pzeoW, u1_cRx97fdDWf_s2dfuBCP2s, u_bceK7yP7v7 );
}

__global__ 
void kernel_cuda_wrapper_u_init_zYHeikJnbS(int32_t*  Nparticles_GEZW4pzeoW, double*  u1_cRx97fdDWf_s2dfuBCP2s, double* u_bceK7yP7v7 ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 15;
	if (tid < 400) {
		exec_range++;
	}
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_zYHeikJnbS = tid * exec_range + range_iterator + 400;
		if (tid < 400) {
			INDEX_zYHeikJnbS -= 400;
		}
		u_bceK7yP7v7[(INDEX_zYHeikJnbS)] = u1_cRx97fdDWf_s2dfuBCP2s[0] + INDEX_zYHeikJnbS / Nparticles_GEZW4pzeoW[0];
	}
}


