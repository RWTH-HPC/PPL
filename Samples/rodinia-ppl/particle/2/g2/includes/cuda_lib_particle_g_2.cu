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
	float res_3f29uPsAeK;
	res_3f29uPsAeK = 1;
	for ( int32_t i = 0; i < n; i++ ) {
		res_3f29uPsAeK *= x;
	}
		return res_3f29uPsAeK;
}
int32_t roundDouble(double value) {
	int32_t newValue_jAc3zQPTEW;
	newValue_jAc3zQPTEW = Cast2Int(value);
	if ((value - newValue_jAc3zQPTEW < 0.5)) {
				return newValue_jAc3zQPTEW;
	} else {
				return newValue_jAc3zQPTEW + 1;
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

void cuda_wrapper_u_init_zEljNqeMNI(int32_t* Nparticles_AflVLOUd4r, double* u1_JjqTLgAmUm_alOS831NNS, double* u_usdhqA1DH6 ) {

	kernel_cuda_wrapper_u_init_zEljNqeMNI<<<10, 64>>> (Nparticles_AflVLOUd4r, u1_JjqTLgAmUm_alOS831NNS, u_usdhqA1DH6 );
}

__global__ 
void kernel_cuda_wrapper_u_init_zEljNqeMNI(int32_t*  Nparticles_AflVLOUd4r, double*  u1_JjqTLgAmUm_alOS831NNS, double* u_usdhqA1DH6 ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 15;
	if (tid < 400) {
		exec_range++;
	}
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_zEljNqeMNI = tid * exec_range + range_iterator + 400;
		if (tid < 400) {
			INDEX_zEljNqeMNI -= 400;
		}
		u_usdhqA1DH6[(INDEX_zEljNqeMNI)] = u1_JjqTLgAmUm_alOS831NNS[0] + INDEX_zEljNqeMNI / Nparticles_AflVLOUd4r[0];
	}
}

void cuda_wrapper_u_init_LDt82V3Q4y(double* u1_JjqTLgAmUm_alOS831NNS, int32_t* Nparticles_AflVLOUd4r, double* u_usdhqA1DH6 ) {

	kernel_cuda_wrapper_u_init_LDt82V3Q4y<<<10, 64>>> (u1_JjqTLgAmUm_alOS831NNS, Nparticles_AflVLOUd4r, u_usdhqA1DH6 );
}

__global__ 
void kernel_cuda_wrapper_u_init_LDt82V3Q4y(double*  u1_JjqTLgAmUm_alOS831NNS, int32_t*  Nparticles_AflVLOUd4r, double* u_usdhqA1DH6 ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 15;
	if (tid < 400) {
		exec_range++;
	}
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_LDt82V3Q4y = tid * exec_range + range_iterator + 400;
		if (tid < 400) {
			INDEX_LDt82V3Q4y -= 400;
		}
		u_usdhqA1DH6[(INDEX_LDt82V3Q4y)] = u1_JjqTLgAmUm_alOS831NNS[0] + INDEX_LDt82V3Q4y / Nparticles_AflVLOUd4r[0];
	}
}

void cuda_wrapper_u_init_mcL9deInrq(double* u1_JjqTLgAmUm_alOS831NNS, int32_t* Nparticles_AflVLOUd4r, double* u_usdhqA1DH6 ) {

	kernel_cuda_wrapper_u_init_mcL9deInrq<<<10, 64>>> (u1_JjqTLgAmUm_alOS831NNS, Nparticles_AflVLOUd4r, u_usdhqA1DH6 );
}

__global__ 
void kernel_cuda_wrapper_u_init_mcL9deInrq(double*  u1_JjqTLgAmUm_alOS831NNS, int32_t*  Nparticles_AflVLOUd4r, double* u_usdhqA1DH6 ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 15;
	if (tid < 400) {
		exec_range++;
	}
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_mcL9deInrq = tid * exec_range + range_iterator + 400;
		if (tid < 400) {
			INDEX_mcL9deInrq -= 400;
		}
		u_usdhqA1DH6[(INDEX_mcL9deInrq)] = u1_JjqTLgAmUm_alOS831NNS[0] + INDEX_mcL9deInrq / Nparticles_AflVLOUd4r[0];
	}
}

void cuda_wrapper_u_init_8u9rMtQzit(int32_t* Nparticles_AflVLOUd4r, double* u1_JjqTLgAmUm_alOS831NNS, double* u_usdhqA1DH6 ) {

	kernel_cuda_wrapper_u_init_8u9rMtQzit<<<10, 64>>> (Nparticles_AflVLOUd4r, u1_JjqTLgAmUm_alOS831NNS, u_usdhqA1DH6 );
}

__global__ 
void kernel_cuda_wrapper_u_init_8u9rMtQzit(int32_t*  Nparticles_AflVLOUd4r, double*  u1_JjqTLgAmUm_alOS831NNS, double* u_usdhqA1DH6 ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 15;
	if (tid < 400) {
		exec_range++;
	}
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_8u9rMtQzit = tid * exec_range + range_iterator + 400;
		if (tid < 400) {
			INDEX_8u9rMtQzit -= 400;
		}
		u_usdhqA1DH6[(INDEX_8u9rMtQzit)] = u1_JjqTLgAmUm_alOS831NNS[0] + INDEX_8u9rMtQzit / Nparticles_AflVLOUd4r[0];
	}
}


