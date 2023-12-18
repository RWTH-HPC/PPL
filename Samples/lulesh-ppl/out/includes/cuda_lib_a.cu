/**


This file implements the header for the Thread Pool and barrier implementation with PThreads.


*/
#include <stdio.h>
#include <stdlib.h>
#include "cuda_lib_a.hxx"
#include "cuda_lib_a.cuh"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include "Patternlib.hxx"


double* VoluDer(double x0, double x1, double x2, double x3, double x4, double x5, double y0, double y1, double y2, double y3, double y4, double y5, double z0, double z1, double z2, double z3, double z4, double z5) {
	double* dvd_9SOmpZbH7x;
	double twelfth_xvaPXoH8Gk;
	twelfth_xvaPXoH8Gk = 1.0 / 12.0;
	dvd_9SOmpZbH7x = Init_List(dvd_9SOmpZbH7x, 3LL * 1LL);
	dvd_9SOmpZbH7x[(xIdx)] = (y1 + y2) * (z0 + z1) - (y0 + y1) * (z1 + z2) + (y0 + y4) * (z3 + z4) - (y3 + y4) * (z0 + z4) - (y2 + y5) * (z3 + z5) + (y3 + y5) * (z2 + z5);
	dvd_9SOmpZbH7x[(yIdx)] = 0 - (x1 + x2) * (z0 + z1) + (x0 + x1) * (z1 + z2) - (x0 + x4) * (z3 + z4) + (x3 + x4) * (z0 + z4) + (x2 + x5) * (z3 + z5) - (x3 + x5) * (z2 + z5);
	dvd_9SOmpZbH7x[(yIdx)] = 0 - (y1 + y2) * (x0 + x1) + (y0 + y1) * (x1 + x2) - (y0 + y4) * (x3 + x4) + (y3 + y4) * (x0 + x4) + (y2 + y5) * (x3 + x5) - (y3 + y5) * (x2 + x5);
	dvd_9SOmpZbH7x[(xIdx)] *= twelfth_xvaPXoH8Gk;
	dvd_9SOmpZbH7x[(yIdx)] *= twelfth_xvaPXoH8Gk;
	dvd_9SOmpZbH7x[(zIdx)] *= twelfth_xvaPXoH8Gk;
		return dvd_9SOmpZbH7x;
}
double TRIPLE_PRODUCT(double x1, double y1, double z1, double x2, double y2, double z2, double x3, double y3, double z3) {
		return ((x1) * ((y2) * (z3) - (z2) * (y3)) + (x2) * ((z1) * (y3) - (y1) * (z3)) + (x3) * ((y1) * (z2) - (z1) * (y2)));
}
double* luleshSeq_CalcPressureForElems(double vnewc, double compression, double e_old) {
	double c1s_n8XOjuKDtG;
	double res_bvc_RtdPu0AFJV;
	double res_pbvc_nQV847UoRB;
	double res_p_new_5WTKxSA0IT;
	double* result_VG4vKjMdqF;
	c1s_n8XOjuKDtG = 2.0 / 3.0;
	res_bvc_RtdPu0AFJV = c1s_n8XOjuKDtG * (compression + 1.0);
	res_pbvc_nQV847UoRB = c1s_n8XOjuKDtG;
	res_p_new_5WTKxSA0IT = res_bvc_RtdPu0AFJV * e_old;
	if ((fabs(res_p_new_5WTKxSA0IT) < m_p_cut)) {
		res_p_new_5WTKxSA0IT = 0.0;
	}
	if ((vnewc >= m_eosvmax)) {
		res_p_new_5WTKxSA0IT = 0.0;
	}
	if ((res_p_new_5WTKxSA0IT < m_pmin)) {
		res_p_new_5WTKxSA0IT = m_pmin;
	}
	result_VG4vKjMdqF = Init_List(result_VG4vKjMdqF, 3LL * 1LL);
	result_VG4vKjMdqF[(0)] = res_bvc_RtdPu0AFJV;
	result_VG4vKjMdqF[(1)] = res_pbvc_nQV847UoRB;
	result_VG4vKjMdqF[(2)] = res_p_new_5WTKxSA0IT;
		return result_VG4vKjMdqF;
}
double luleshSeq_CalcSoundSpeedForElems(double pbvc, double bvc, double enewc, double pnewc, double vnewc) {
	double ssTmp_1hyscJgLZF;
	ssTmp_1hyscJgLZF = (pbvc * enewc + vnewc * vnewc * bvc * pnewc) / m_refdens;
	if ((ssTmp_1hyscJgLZF <= 1.111111E-37)) {
		ssTmp_1hyscJgLZF = 3.333333E-19;
	} else {
		ssTmp_1hyscJgLZF = sqrt(ssTmp_1hyscJgLZF);
	}
		return ssTmp_1hyscJgLZF;
}
double AreaFace(double x0, double x1, double x2, double x3, double y0, double y1, double y2, double y3, double z0, double z1, double z2, double z3) {
	double area_J3uVYXBTYj;
	double fx_n5ZNCcLTzA;
	double fy_R4fzYipWZC;
	double fz_DEVpcaB1IB;
	double gx_0jf0HYGSPQ;
	double gy_QQnjaL83n4;
	double gz_Th0TdPSaSg;
	fx_n5ZNCcLTzA = (x2 - x0) - (x3 - x1);
	fy_R4fzYipWZC = (y2 - y0) - (y3 - y1);
	fz_DEVpcaB1IB = (z2 - z0) - (z3 - z1);
	gx_0jf0HYGSPQ = (x2 - x0) + (x3 - x1);
	gy_QQnjaL83n4 = (y2 - y0) + (y3 - y1);
	gz_Th0TdPSaSg = (z2 - z0) + (z3 - z1);
	area_J3uVYXBTYj = (fx_n5ZNCcLTzA * fx_n5ZNCcLTzA + fy_R4fzYipWZC * fy_R4fzYipWZC + fz_DEVpcaB1IB * fz_DEVpcaB1IB) * (gx_0jf0HYGSPQ * gx_0jf0HYGSPQ + gy_QQnjaL83n4 * gy_QQnjaL83n4 + gz_Th0TdPSaSg * gz_Th0TdPSaSg) - (fx_n5ZNCcLTzA * gx_0jf0HYGSPQ + fy_R4fzYipWZC * gy_QQnjaL83n4 + fz_DEVpcaB1IB * gz_Th0TdPSaSg) * (fx_n5ZNCcLTzA * gx_0jf0HYGSPQ + fy_R4fzYipWZC * gy_QQnjaL83n4 + fz_DEVpcaB1IB * gz_Th0TdPSaSg);
		return area_J3uVYXBTYj;
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


