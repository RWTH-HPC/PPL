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
	float res_1r3Ebr3Yff;
	res_1r3Ebr3Yff = 1;
	for ( int32_t i = 0; i < n; i++ ) {
		res_1r3Ebr3Yff *= x;
	}
		return res_1r3Ebr3Yff;
}
int32_t roundDouble(double value) {
	int32_t newValue_Q44AYnMtep;
	newValue_Q44AYnMtep = Cast2Int(value);
	if ((value - newValue_Q44AYnMtep < 0.5)) {
				return newValue_Q44AYnMtep;
	} else {
				return newValue_Q44AYnMtep + 1;
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

void cuda_wrapper_u_init_jbSoE3vhKa(int32_t* Nparticles_CcjxmD75Zp, double* u1_iSCTjLSB7b_36NhHedM04, double* u_CYHoLeWYXx ) {

	kernel_cuda_wrapper_u_init_jbSoE3vhKa<<<10, 64>>> (Nparticles_CcjxmD75Zp, u1_iSCTjLSB7b_36NhHedM04, u_CYHoLeWYXx );
}

__global__ 
void kernel_cuda_wrapper_u_init_jbSoE3vhKa(int32_t*  Nparticles_CcjxmD75Zp, double*  u1_iSCTjLSB7b_36NhHedM04, double* u_CYHoLeWYXx ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 15;
	if (tid < 400) {
		exec_range++;
	}
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_jbSoE3vhKa = tid * exec_range + range_iterator + 400;
		if (tid < 400) {
			INDEX_jbSoE3vhKa -= 400;
		}
		u_CYHoLeWYXx[(INDEX_jbSoE3vhKa)] = u1_iSCTjLSB7b_36NhHedM04[0] + INDEX_jbSoE3vhKa / Nparticles_CcjxmD75Zp[0];
	}
}

void cuda_wrapper_u_init_vg8QPBCDrI(double* u1_iSCTjLSB7b_36NhHedM04, int32_t* Nparticles_CcjxmD75Zp, double* u_CYHoLeWYXx ) {

	kernel_cuda_wrapper_u_init_vg8QPBCDrI<<<10, 64>>> (u1_iSCTjLSB7b_36NhHedM04, Nparticles_CcjxmD75Zp, u_CYHoLeWYXx );
}

__global__ 
void kernel_cuda_wrapper_u_init_vg8QPBCDrI(double*  u1_iSCTjLSB7b_36NhHedM04, int32_t*  Nparticles_CcjxmD75Zp, double* u_CYHoLeWYXx ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 15;
	if (tid < 400) {
		exec_range++;
	}
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_vg8QPBCDrI = tid * exec_range + range_iterator + 400;
		if (tid < 400) {
			INDEX_vg8QPBCDrI -= 400;
		}
		u_CYHoLeWYXx[(INDEX_vg8QPBCDrI)] = u1_iSCTjLSB7b_36NhHedM04[0] + INDEX_vg8QPBCDrI / Nparticles_CcjxmD75Zp[0];
	}
}

void cuda_wrapper_u_init_EPUPPspMoQ(double* u1_iSCTjLSB7b_36NhHedM04, int32_t* Nparticles_CcjxmD75Zp, double* u_CYHoLeWYXx ) {

	kernel_cuda_wrapper_u_init_EPUPPspMoQ<<<10, 64>>> (u1_iSCTjLSB7b_36NhHedM04, Nparticles_CcjxmD75Zp, u_CYHoLeWYXx );
}

__global__ 
void kernel_cuda_wrapper_u_init_EPUPPspMoQ(double*  u1_iSCTjLSB7b_36NhHedM04, int32_t*  Nparticles_CcjxmD75Zp, double* u_CYHoLeWYXx ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 15;
	if (tid < 400) {
		exec_range++;
	}
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_EPUPPspMoQ = tid * exec_range + range_iterator + 400;
		if (tid < 400) {
			INDEX_EPUPPspMoQ -= 400;
		}
		u_CYHoLeWYXx[(INDEX_EPUPPspMoQ)] = u1_iSCTjLSB7b_36NhHedM04[0] + INDEX_EPUPPspMoQ / Nparticles_CcjxmD75Zp[0];
	}
}

void cuda_wrapper_u_init_ioGLUgAge2(int32_t* Nparticles_CcjxmD75Zp, double* u1_iSCTjLSB7b_36NhHedM04, double* u_CYHoLeWYXx ) {

	kernel_cuda_wrapper_u_init_ioGLUgAge2<<<10, 64>>> (Nparticles_CcjxmD75Zp, u1_iSCTjLSB7b_36NhHedM04, u_CYHoLeWYXx );
}

__global__ 
void kernel_cuda_wrapper_u_init_ioGLUgAge2(int32_t*  Nparticles_CcjxmD75Zp, double*  u1_iSCTjLSB7b_36NhHedM04, double* u_CYHoLeWYXx ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 15;
	if (tid < 400) {
		exec_range++;
	}
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_ioGLUgAge2 = tid * exec_range + range_iterator + 400;
		if (tid < 400) {
			INDEX_ioGLUgAge2 -= 400;
		}
		u_CYHoLeWYXx[(INDEX_ioGLUgAge2)] = u1_iSCTjLSB7b_36NhHedM04[0] + INDEX_ioGLUgAge2 / Nparticles_CcjxmD75Zp[0];
	}
}


