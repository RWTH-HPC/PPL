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
	float res_cpIGzlQeQ2;
	res_cpIGzlQeQ2 = 1;
	for ( int32_t i = 0; i < n; i++ ) {
		res_cpIGzlQeQ2 *= x;
	}
		return res_cpIGzlQeQ2;
}
int32_t roundDouble(double value) {
	int32_t newValue_If443zDlkg;
	newValue_If443zDlkg = Cast2Int(value);
	if ((value - newValue_If443zDlkg < 0.5)) {
				return newValue_If443zDlkg;
	} else {
				return newValue_If443zDlkg + 1;
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

void cuda_wrapper_u_init_8bIU6Vh2Lc(int32_t* Nparticles_9WpkAzeGZz, double* u1_iWRHIHKWBp_RNut6JFnfX, double* u_4nVTHiRif8 ) {

	kernel_cuda_wrapper_u_init_8bIU6Vh2Lc<<<10, 64>>> (Nparticles_9WpkAzeGZz, u1_iWRHIHKWBp_RNut6JFnfX, u_4nVTHiRif8 );
}

__global__ 
void kernel_cuda_wrapper_u_init_8bIU6Vh2Lc(int32_t*  Nparticles_9WpkAzeGZz, double*  u1_iWRHIHKWBp_RNut6JFnfX, double* u_4nVTHiRif8 ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 15;
	if (tid < 400) {
		exec_range++;
	}
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_8bIU6Vh2Lc = tid * exec_range + range_iterator + 400;
		if (tid < 400) {
			INDEX_8bIU6Vh2Lc -= 400;
		}
		u_4nVTHiRif8[(INDEX_8bIU6Vh2Lc)] = u1_iWRHIHKWBp_RNut6JFnfX[0] + INDEX_8bIU6Vh2Lc / Nparticles_9WpkAzeGZz[0];
	}
}

void cuda_wrapper_u_init_0qFIgBT9bz(double* u1_iWRHIHKWBp_RNut6JFnfX, int32_t* Nparticles_9WpkAzeGZz, double* u_4nVTHiRif8 ) {

	kernel_cuda_wrapper_u_init_0qFIgBT9bz<<<10, 64>>> (u1_iWRHIHKWBp_RNut6JFnfX, Nparticles_9WpkAzeGZz, u_4nVTHiRif8 );
}

__global__ 
void kernel_cuda_wrapper_u_init_0qFIgBT9bz(double*  u1_iWRHIHKWBp_RNut6JFnfX, int32_t*  Nparticles_9WpkAzeGZz, double* u_4nVTHiRif8 ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 15;
	if (tid < 400) {
		exec_range++;
	}
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_0qFIgBT9bz = tid * exec_range + range_iterator + 400;
		if (tid < 400) {
			INDEX_0qFIgBT9bz -= 400;
		}
		u_4nVTHiRif8[(INDEX_0qFIgBT9bz)] = u1_iWRHIHKWBp_RNut6JFnfX[0] + INDEX_0qFIgBT9bz / Nparticles_9WpkAzeGZz[0];
	}
}

void cuda_wrapper_u_init_4N21VvnF41(double* u1_iWRHIHKWBp_RNut6JFnfX, int32_t* Nparticles_9WpkAzeGZz, double* u_4nVTHiRif8 ) {

	kernel_cuda_wrapper_u_init_4N21VvnF41<<<10, 64>>> (u1_iWRHIHKWBp_RNut6JFnfX, Nparticles_9WpkAzeGZz, u_4nVTHiRif8 );
}

__global__ 
void kernel_cuda_wrapper_u_init_4N21VvnF41(double*  u1_iWRHIHKWBp_RNut6JFnfX, int32_t*  Nparticles_9WpkAzeGZz, double* u_4nVTHiRif8 ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 15;
	if (tid < 400) {
		exec_range++;
	}
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_4N21VvnF41 = tid * exec_range + range_iterator + 400;
		if (tid < 400) {
			INDEX_4N21VvnF41 -= 400;
		}
		u_4nVTHiRif8[(INDEX_4N21VvnF41)] = u1_iWRHIHKWBp_RNut6JFnfX[0] + INDEX_4N21VvnF41 / Nparticles_9WpkAzeGZz[0];
	}
}

void cuda_wrapper_u_init_ZjrW763TVm(int32_t* Nparticles_9WpkAzeGZz, double* u1_iWRHIHKWBp_RNut6JFnfX, double* u_4nVTHiRif8 ) {

	kernel_cuda_wrapper_u_init_ZjrW763TVm<<<10, 64>>> (Nparticles_9WpkAzeGZz, u1_iWRHIHKWBp_RNut6JFnfX, u_4nVTHiRif8 );
}

__global__ 
void kernel_cuda_wrapper_u_init_ZjrW763TVm(int32_t*  Nparticles_9WpkAzeGZz, double*  u1_iWRHIHKWBp_RNut6JFnfX, double* u_4nVTHiRif8 ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 15;
	if (tid < 400) {
		exec_range++;
	}
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_ZjrW763TVm = tid * exec_range + range_iterator + 400;
		if (tid < 400) {
			INDEX_ZjrW763TVm -= 400;
		}
		u_4nVTHiRif8[(INDEX_ZjrW763TVm)] = u1_iWRHIHKWBp_RNut6JFnfX[0] + INDEX_ZjrW763TVm / Nparticles_9WpkAzeGZz[0];
	}
}


