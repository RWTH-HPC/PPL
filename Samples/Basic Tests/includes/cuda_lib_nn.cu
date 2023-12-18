/**


This file implements the header for the Thread Pool and barrier implementation with PThreads.


*/
#include <stdio.h>
#include <stdlib.h>
#include "cuda_lib_nn.hxx"
#include "cuda_lib_nn.cuh"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include "Patternlib.hxx"




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

void cuda_wrapper_rand_init_eeNUSUs3M2_WLX2ob9p0V(float* weights_6_SpUa0qY8sP ) {

	kernel_cuda_wrapper_rand_init_eeNUSUs3M2_WLX2ob9p0V<<<1, 64>>> (weights_6_SpUa0qY8sP );
}

__global__ 
void kernel_cuda_wrapper_rand_init_eeNUSUs3M2_WLX2ob9p0V(float* weights_6_SpUa0qY8sP ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 1;
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_WLX2ob9p0V = tid * exec_range + range_iterator;
		for (size_t INDEX_56xn1THmlI = 0; INDEX_56xn1THmlI < 0 + 64; ++INDEX_56xn1THmlI) {
			weights_6_SpUa0qY8sP[64LL * (INDEX_WLX2ob9p0V) + (INDEX_56xn1THmlI)] = randomizerGPU() / MAX_INT;
		}
	}
}

void cuda_wrapper_rand_init_PlH53hIC5U_eDdEMWtudn(float* weights_4_WDk6GW21eM ) {

	kernel_cuda_wrapper_rand_init_PlH53hIC5U_eDdEMWtudn<<<1, 64>>> (weights_4_WDk6GW21eM );
}

__global__ 
void kernel_cuda_wrapper_rand_init_PlH53hIC5U_eDdEMWtudn(float* weights_4_WDk6GW21eM ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 1;
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_eDdEMWtudn = tid * exec_range + range_iterator;
		for (size_t INDEX_AmHEJhWUzm = 0; INDEX_AmHEJhWUzm < 0 + 64; ++INDEX_AmHEJhWUzm) {
			weights_4_WDk6GW21eM[64LL * (INDEX_eDdEMWtudn) + (INDEX_AmHEJhWUzm)] = randomizerGPU() / MAX_INT;
		}
	}
}

void cuda_wrapper_rand_init_eq1HFX6DF2_ePAri9DhVW(float* weights_3_aPXpgT7mlK ) {

	kernel_cuda_wrapper_rand_init_eq1HFX6DF2_ePAri9DhVW<<<1, 64>>> (weights_3_aPXpgT7mlK );
}

__global__ 
void kernel_cuda_wrapper_rand_init_eq1HFX6DF2_ePAri9DhVW(float* weights_3_aPXpgT7mlK ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 1;
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_ePAri9DhVW = tid * exec_range + range_iterator;
		for (size_t INDEX_nPeGb9Lx4G = 0; INDEX_nPeGb9Lx4G < 0 + 64; ++INDEX_nPeGb9Lx4G) {
			weights_3_aPXpgT7mlK[64LL * (INDEX_ePAri9DhVW) + (INDEX_nPeGb9Lx4G)] = randomizerGPU() / MAX_INT;
		}
	}
}

void cuda_wrapper_rand_init_YvV3g1HMS2(float* weights_8_exzdbedc44 ) {

	kernel_cuda_wrapper_rand_init_YvV3g1HMS2<<<1, 64>>> (weights_8_exzdbedc44 );
}

__global__ 
void kernel_cuda_wrapper_rand_init_YvV3g1HMS2(float* weights_8_exzdbedc44 ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 0;
	if (tid < 10) {
		exec_range++;
	}
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_YvV3g1HMS2 = tid * exec_range + range_iterator + 10;
		if (tid < 10) {
			INDEX_YvV3g1HMS2 -= 10;
		}
		for (size_t INDEX_aC2lxftQxv = 0; INDEX_aC2lxftQxv < 0 + 64; ++INDEX_aC2lxftQxv) {
			weights_8_exzdbedc44[64LL * (INDEX_YvV3g1HMS2) + (INDEX_aC2lxftQxv)] = randomizerGPU() / MAX_INT;
		}
	}
}

void cuda_wrapper_rand_init_ndBZfLHlBF_krQlq6Omgs(float* weights_2_PULl08Tzft ) {

	kernel_cuda_wrapper_rand_init_ndBZfLHlBF_krQlq6Omgs<<<1, 64>>> (weights_2_PULl08Tzft );
}

__global__ 
void kernel_cuda_wrapper_rand_init_ndBZfLHlBF_krQlq6Omgs(float* weights_2_PULl08Tzft ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 1;
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_krQlq6Omgs = tid * exec_range + range_iterator;
		for (size_t INDEX_dOkbOgixCv = 0; INDEX_dOkbOgixCv < 0 + 64; ++INDEX_dOkbOgixCv) {
			weights_2_PULl08Tzft[64LL * (INDEX_krQlq6Omgs) + (INDEX_dOkbOgixCv)] = randomizerGPU() / MAX_INT;
		}
	}
}

void cuda_wrapper_rand_init_KeVA7GQAem_bzOG6cJAdH(float* weights_1_1gUuD5JLuk ) {

	kernel_cuda_wrapper_rand_init_KeVA7GQAem_bzOG6cJAdH<<<1, 64>>> (weights_1_1gUuD5JLuk );
}

__global__ 
void kernel_cuda_wrapper_rand_init_KeVA7GQAem_bzOG6cJAdH(float* weights_1_1gUuD5JLuk ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 1;
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_bzOG6cJAdH = tid * exec_range + range_iterator;
		for (size_t INDEX_hkL9u8rbjV = 0; INDEX_hkL9u8rbjV < 0 + 64; ++INDEX_hkL9u8rbjV) {
			weights_1_1gUuD5JLuk[64LL * (INDEX_bzOG6cJAdH) + (INDEX_hkL9u8rbjV)] = randomizerGPU() / MAX_INT;
		}
	}
}

void cuda_wrapper_rand_init_VrA8skJMcp_N0ZhsA9PUC(float* weights_7_LJVmWXj3HV ) {

	kernel_cuda_wrapper_rand_init_VrA8skJMcp_N0ZhsA9PUC<<<1, 64>>> (weights_7_LJVmWXj3HV );
}

__global__ 
void kernel_cuda_wrapper_rand_init_VrA8skJMcp_N0ZhsA9PUC(float* weights_7_LJVmWXj3HV ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 1;
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_N0ZhsA9PUC = tid * exec_range + range_iterator;
		for (size_t INDEX_t5MQhlmnXE = 0; INDEX_t5MQhlmnXE < 0 + 64; ++INDEX_t5MQhlmnXE) {
			weights_7_LJVmWXj3HV[64LL * (INDEX_N0ZhsA9PUC) + (INDEX_t5MQhlmnXE)] = randomizerGPU() / MAX_INT;
		}
	}
}

void cuda_wrapper_rand_init_png5hzlbR4_1Xe4UrPyk8(float* weights_5_dUKeqRRt9x ) {

	kernel_cuda_wrapper_rand_init_png5hzlbR4_1Xe4UrPyk8<<<1, 64>>> (weights_5_dUKeqRRt9x );
}

__global__ 
void kernel_cuda_wrapper_rand_init_png5hzlbR4_1Xe4UrPyk8(float* weights_5_dUKeqRRt9x ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 1;
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_1Xe4UrPyk8 = tid * exec_range + range_iterator;
		for (size_t INDEX_tyqq8XYi64 = 0; INDEX_tyqq8XYi64 < 0 + 64; ++INDEX_tyqq8XYi64) {
			weights_5_dUKeqRRt9x[64LL * (INDEX_1Xe4UrPyk8) + (INDEX_tyqq8XYi64)] = randomizerGPU() / MAX_INT;
		}
	}
}

void cuda_wrapper_forward_BBhRO6II0W_tKpfCHXWeK(float* weights_1_1gUuD5JLuk, float* batch_GkpriWeNHS, float* activations_1_fDjHO1CRW1 ) {

	kernel_cuda_wrapper_forward_BBhRO6II0W_tKpfCHXWeK<<<64, 64>>> (weights_1_1gUuD5JLuk, batch_GkpriWeNHS, activations_1_fDjHO1CRW1 );
}

__global__ 
void kernel_cuda_wrapper_forward_BBhRO6II0W_tKpfCHXWeK(float*  weights_1_1gUuD5JLuk, float*  batch_GkpriWeNHS, float* activations_1_fDjHO1CRW1 ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 64;
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_tKpfCHXWeK = tid * exec_range + range_iterator;
		for (size_t INDEX_7PfypRXRL0 = 0; INDEX_7PfypRXRL0 < 0 + 64; ++INDEX_7PfypRXRL0) {
			float act_yHobtAoOMA;
			float z_pSbBhk1nKP;
			z_pSbBhk1nKP = 0;
			act_yHobtAoOMA = 0;
			for (size_t INDEX_xrbvJPDEME = 0; INDEX_xrbvJPDEME < 0 + 64; ++INDEX_xrbvJPDEME) {
				z_pSbBhk1nKP += weights_1_1gUuD5JLuk[64LL * (INDEX_7PfypRXRL0) + (INDEX_xrbvJPDEME)] * batch_GkpriWeNHS[64LL * (INDEX_tKpfCHXWeK) + (INDEX_xrbvJPDEME)];
			}
			float inlineFunctionValue_U8KM875cOn_gqAxFCbi1k;
			{
				float x_gqAxFCbi1k = z_pSbBhk1nKP;
				inlineFunctionValue_U8KM875cOn_gqAxFCbi1k = x_gqAxFCbi1k - (1.0 / 3.0) * (x_gqAxFCbi1k * x_gqAxFCbi1k * x_gqAxFCbi1k) + (2.0 / 15.0) * (x_gqAxFCbi1k * x_gqAxFCbi1k * x_gqAxFCbi1k * x_gqAxFCbi1k * x_gqAxFCbi1k) - (17 / 315) * (x_gqAxFCbi1k * x_gqAxFCbi1k * x_gqAxFCbi1k * x_gqAxFCbi1k * x_gqAxFCbi1k * x_gqAxFCbi1k * x_gqAxFCbi1k);
				goto STOP_LABEL_gqAxFCbi1k;
				STOP_LABEL_gqAxFCbi1k:
				noop;
			}
			act_yHobtAoOMA = inlineFunctionValue_U8KM875cOn_gqAxFCbi1k;
			activations_1_fDjHO1CRW1[64LL * (INDEX_tKpfCHXWeK) + (INDEX_7PfypRXRL0)] = act_yHobtAoOMA;
		}
	}
}

void cuda_wrapper_forward_F4SnxAXfxf_eed7oxtgNP(float* activations_1_fDjHO1CRW1, float* weights_2_PULl08Tzft, float* activations_2_j3fti3BN6j ) {

	kernel_cuda_wrapper_forward_F4SnxAXfxf_eed7oxtgNP<<<64, 64>>> (activations_1_fDjHO1CRW1, weights_2_PULl08Tzft, activations_2_j3fti3BN6j );
}

__global__ 
void kernel_cuda_wrapper_forward_F4SnxAXfxf_eed7oxtgNP(float*  activations_1_fDjHO1CRW1, float*  weights_2_PULl08Tzft, float* activations_2_j3fti3BN6j ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 64;
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_eed7oxtgNP = tid * exec_range + range_iterator;
		for (size_t INDEX_MGjNt0FHcx = 0; INDEX_MGjNt0FHcx < 0 + 64; ++INDEX_MGjNt0FHcx) {
			float act_TpycoVXR6p;
			float z_WHFQdOW1cL;
			z_WHFQdOW1cL = 0;
			act_TpycoVXR6p = 0;
			for (size_t INDEX_hpPKuQK4RI = 0; INDEX_hpPKuQK4RI < 0 + 64; ++INDEX_hpPKuQK4RI) {
				z_WHFQdOW1cL += weights_2_PULl08Tzft[64LL * (INDEX_MGjNt0FHcx) + (INDEX_hpPKuQK4RI)] * activations_1_fDjHO1CRW1[64LL * (INDEX_eed7oxtgNP) + (INDEX_hpPKuQK4RI)];
			}
			float inlineFunctionValue_U8KM875cOn_mqSomJ2YQE;
			{
				float x_mqSomJ2YQE = z_WHFQdOW1cL;
				inlineFunctionValue_U8KM875cOn_mqSomJ2YQE = x_mqSomJ2YQE - (1.0 / 3.0) * (x_mqSomJ2YQE * x_mqSomJ2YQE * x_mqSomJ2YQE) + (2.0 / 15.0) * (x_mqSomJ2YQE * x_mqSomJ2YQE * x_mqSomJ2YQE * x_mqSomJ2YQE * x_mqSomJ2YQE) - (17 / 315) * (x_mqSomJ2YQE * x_mqSomJ2YQE * x_mqSomJ2YQE * x_mqSomJ2YQE * x_mqSomJ2YQE * x_mqSomJ2YQE * x_mqSomJ2YQE);
				goto STOP_LABEL_mqSomJ2YQE;
				STOP_LABEL_mqSomJ2YQE:
				noop;
			}
			act_TpycoVXR6p = inlineFunctionValue_U8KM875cOn_mqSomJ2YQE;
			activations_2_j3fti3BN6j[64LL * (INDEX_eed7oxtgNP) + (INDEX_MGjNt0FHcx)] = act_TpycoVXR6p;
		}
	}
}

void cuda_wrapper_forward_qAsnKyB9OD_oSeefsNCRE(float* activations_2_j3fti3BN6j, float* weights_3_aPXpgT7mlK, float* activations_3_d2FySJrNO8 ) {

	kernel_cuda_wrapper_forward_qAsnKyB9OD_oSeefsNCRE<<<64, 64>>> (activations_2_j3fti3BN6j, weights_3_aPXpgT7mlK, activations_3_d2FySJrNO8 );
}

__global__ 
void kernel_cuda_wrapper_forward_qAsnKyB9OD_oSeefsNCRE(float*  activations_2_j3fti3BN6j, float*  weights_3_aPXpgT7mlK, float* activations_3_d2FySJrNO8 ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 64;
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_oSeefsNCRE = tid * exec_range + range_iterator;
		for (size_t INDEX_8MWjRbaJBp = 0; INDEX_8MWjRbaJBp < 0 + 64; ++INDEX_8MWjRbaJBp) {
			float act_cNivMqLLzq;
			float z_JYrQTueLC4;
			z_JYrQTueLC4 = 0;
			act_cNivMqLLzq = 0;
			for (size_t INDEX_0zBqHdcQSO = 0; INDEX_0zBqHdcQSO < 0 + 64; ++INDEX_0zBqHdcQSO) {
				z_JYrQTueLC4 += weights_3_aPXpgT7mlK[64LL * (INDEX_8MWjRbaJBp) + (INDEX_0zBqHdcQSO)] * activations_2_j3fti3BN6j[64LL * (INDEX_oSeefsNCRE) + (INDEX_0zBqHdcQSO)];
			}
			float inlineFunctionValue_U8KM875cOn_ddzJ8kfMBG;
			{
				float x_ddzJ8kfMBG = z_JYrQTueLC4;
				inlineFunctionValue_U8KM875cOn_ddzJ8kfMBG = x_ddzJ8kfMBG - (1.0 / 3.0) * (x_ddzJ8kfMBG * x_ddzJ8kfMBG * x_ddzJ8kfMBG) + (2.0 / 15.0) * (x_ddzJ8kfMBG * x_ddzJ8kfMBG * x_ddzJ8kfMBG * x_ddzJ8kfMBG * x_ddzJ8kfMBG) - (17 / 315) * (x_ddzJ8kfMBG * x_ddzJ8kfMBG * x_ddzJ8kfMBG * x_ddzJ8kfMBG * x_ddzJ8kfMBG * x_ddzJ8kfMBG * x_ddzJ8kfMBG);
				goto STOP_LABEL_ddzJ8kfMBG;
				STOP_LABEL_ddzJ8kfMBG:
				noop;
			}
			act_cNivMqLLzq = inlineFunctionValue_U8KM875cOn_ddzJ8kfMBG;
			activations_3_d2FySJrNO8[64LL * (INDEX_oSeefsNCRE) + (INDEX_8MWjRbaJBp)] = act_cNivMqLLzq;
		}
	}
}

void cuda_wrapper_forward_HmKGZlyhss_2S7PyBeDBC(float* weights_4_WDk6GW21eM, float* activations_3_d2FySJrNO8, float* activations_4_UuAc29EIlG ) {

	kernel_cuda_wrapper_forward_HmKGZlyhss_2S7PyBeDBC<<<64, 64>>> (weights_4_WDk6GW21eM, activations_3_d2FySJrNO8, activations_4_UuAc29EIlG );
}

__global__ 
void kernel_cuda_wrapper_forward_HmKGZlyhss_2S7PyBeDBC(float*  weights_4_WDk6GW21eM, float*  activations_3_d2FySJrNO8, float* activations_4_UuAc29EIlG ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 64;
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_2S7PyBeDBC = tid * exec_range + range_iterator;
		for (size_t INDEX_jF8xQQTF60 = 0; INDEX_jF8xQQTF60 < 0 + 64; ++INDEX_jF8xQQTF60) {
			float act_NHERYSu1qk;
			float z_0Ijy88QCrA;
			z_0Ijy88QCrA = 0;
			act_NHERYSu1qk = 0;
			for (size_t INDEX_9XdvyWnAZH = 0; INDEX_9XdvyWnAZH < 0 + 64; ++INDEX_9XdvyWnAZH) {
				z_0Ijy88QCrA += weights_4_WDk6GW21eM[64LL * (INDEX_jF8xQQTF60) + (INDEX_9XdvyWnAZH)] * activations_3_d2FySJrNO8[64LL * (INDEX_2S7PyBeDBC) + (INDEX_9XdvyWnAZH)];
			}
			float inlineFunctionValue_U8KM875cOn_2Uv4rvuhlU;
			{
				float x_2Uv4rvuhlU = z_0Ijy88QCrA;
				inlineFunctionValue_U8KM875cOn_2Uv4rvuhlU = x_2Uv4rvuhlU - (1.0 / 3.0) * (x_2Uv4rvuhlU * x_2Uv4rvuhlU * x_2Uv4rvuhlU) + (2.0 / 15.0) * (x_2Uv4rvuhlU * x_2Uv4rvuhlU * x_2Uv4rvuhlU * x_2Uv4rvuhlU * x_2Uv4rvuhlU) - (17 / 315) * (x_2Uv4rvuhlU * x_2Uv4rvuhlU * x_2Uv4rvuhlU * x_2Uv4rvuhlU * x_2Uv4rvuhlU * x_2Uv4rvuhlU * x_2Uv4rvuhlU);
				goto STOP_LABEL_2Uv4rvuhlU;
				STOP_LABEL_2Uv4rvuhlU:
				noop;
			}
			act_NHERYSu1qk = inlineFunctionValue_U8KM875cOn_2Uv4rvuhlU;
			activations_4_UuAc29EIlG[64LL * (INDEX_2S7PyBeDBC) + (INDEX_jF8xQQTF60)] = act_NHERYSu1qk;
		}
	}
}

void cuda_wrapper_forward_ojXO3RetP1_rOG65YEVoL(float* activations_4_UuAc29EIlG, float* weights_5_dUKeqRRt9x, float* activations_5_gielxoWvAU ) {

	kernel_cuda_wrapper_forward_ojXO3RetP1_rOG65YEVoL<<<64, 64>>> (activations_4_UuAc29EIlG, weights_5_dUKeqRRt9x, activations_5_gielxoWvAU );
}

__global__ 
void kernel_cuda_wrapper_forward_ojXO3RetP1_rOG65YEVoL(float*  activations_4_UuAc29EIlG, float*  weights_5_dUKeqRRt9x, float* activations_5_gielxoWvAU ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 64;
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_rOG65YEVoL = tid * exec_range + range_iterator;
		for (size_t INDEX_54U84EjWnT = 0; INDEX_54U84EjWnT < 0 + 64; ++INDEX_54U84EjWnT) {
			float act_XEAY69Z7TH;
			float z_JtupfzG81r;
			z_JtupfzG81r = 0;
			act_XEAY69Z7TH = 0;
			for (size_t INDEX_uc2DAmOFJR = 0; INDEX_uc2DAmOFJR < 0 + 64; ++INDEX_uc2DAmOFJR) {
				z_JtupfzG81r += weights_5_dUKeqRRt9x[64LL * (INDEX_54U84EjWnT) + (INDEX_uc2DAmOFJR)] * activations_4_UuAc29EIlG[64LL * (INDEX_rOG65YEVoL) + (INDEX_uc2DAmOFJR)];
			}
			float inlineFunctionValue_U8KM875cOn_mYetlTkdJI;
			{
				float x_mYetlTkdJI = z_JtupfzG81r;
				inlineFunctionValue_U8KM875cOn_mYetlTkdJI = x_mYetlTkdJI - (1.0 / 3.0) * (x_mYetlTkdJI * x_mYetlTkdJI * x_mYetlTkdJI) + (2.0 / 15.0) * (x_mYetlTkdJI * x_mYetlTkdJI * x_mYetlTkdJI * x_mYetlTkdJI * x_mYetlTkdJI) - (17 / 315) * (x_mYetlTkdJI * x_mYetlTkdJI * x_mYetlTkdJI * x_mYetlTkdJI * x_mYetlTkdJI * x_mYetlTkdJI * x_mYetlTkdJI);
				goto STOP_LABEL_mYetlTkdJI;
				STOP_LABEL_mYetlTkdJI:
				noop;
			}
			act_XEAY69Z7TH = inlineFunctionValue_U8KM875cOn_mYetlTkdJI;
			activations_5_gielxoWvAU[64LL * (INDEX_rOG65YEVoL) + (INDEX_54U84EjWnT)] = act_XEAY69Z7TH;
		}
	}
}

void cuda_wrapper_forward_QemoMWU8Ac_d2Bsln4BkM(float* weights_6_SpUa0qY8sP, float* activations_5_gielxoWvAU, float* activations_6_H4d7Vnb3tP ) {

	kernel_cuda_wrapper_forward_QemoMWU8Ac_d2Bsln4BkM<<<64, 64>>> (weights_6_SpUa0qY8sP, activations_5_gielxoWvAU, activations_6_H4d7Vnb3tP );
}

__global__ 
void kernel_cuda_wrapper_forward_QemoMWU8Ac_d2Bsln4BkM(float*  weights_6_SpUa0qY8sP, float*  activations_5_gielxoWvAU, float* activations_6_H4d7Vnb3tP ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 64;
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_d2Bsln4BkM = tid * exec_range + range_iterator;
		for (size_t INDEX_Dz3cxcPg6U = 0; INDEX_Dz3cxcPg6U < 0 + 64; ++INDEX_Dz3cxcPg6U) {
			float act_MsEAK4Nqlg;
			float z_N9DT02PJXR;
			z_N9DT02PJXR = 0;
			act_MsEAK4Nqlg = 0;
			for (size_t INDEX_Si653OzKH3 = 0; INDEX_Si653OzKH3 < 0 + 64; ++INDEX_Si653OzKH3) {
				z_N9DT02PJXR += weights_6_SpUa0qY8sP[64LL * (INDEX_Dz3cxcPg6U) + (INDEX_Si653OzKH3)] * activations_5_gielxoWvAU[64LL * (INDEX_d2Bsln4BkM) + (INDEX_Si653OzKH3)];
			}
			float inlineFunctionValue_U8KM875cOn_zarpfQrSuG;
			{
				float x_zarpfQrSuG = z_N9DT02PJXR;
				inlineFunctionValue_U8KM875cOn_zarpfQrSuG = x_zarpfQrSuG - (1.0 / 3.0) * (x_zarpfQrSuG * x_zarpfQrSuG * x_zarpfQrSuG) + (2.0 / 15.0) * (x_zarpfQrSuG * x_zarpfQrSuG * x_zarpfQrSuG * x_zarpfQrSuG * x_zarpfQrSuG) - (17 / 315) * (x_zarpfQrSuG * x_zarpfQrSuG * x_zarpfQrSuG * x_zarpfQrSuG * x_zarpfQrSuG * x_zarpfQrSuG * x_zarpfQrSuG);
				goto STOP_LABEL_zarpfQrSuG;
				STOP_LABEL_zarpfQrSuG:
				noop;
			}
			act_MsEAK4Nqlg = inlineFunctionValue_U8KM875cOn_zarpfQrSuG;
			activations_6_H4d7Vnb3tP[64LL * (INDEX_d2Bsln4BkM) + (INDEX_Dz3cxcPg6U)] = act_MsEAK4Nqlg;
		}
	}
}

void cuda_wrapper_forward_l600xy7egX_XZVJnvrdB7(float* activations_6_H4d7Vnb3tP, float* weights_7_LJVmWXj3HV, float* activations_7_RJgsycQ8Eh ) {

	kernel_cuda_wrapper_forward_l600xy7egX_XZVJnvrdB7<<<64, 64>>> (activations_6_H4d7Vnb3tP, weights_7_LJVmWXj3HV, activations_7_RJgsycQ8Eh );
}

__global__ 
void kernel_cuda_wrapper_forward_l600xy7egX_XZVJnvrdB7(float*  activations_6_H4d7Vnb3tP, float*  weights_7_LJVmWXj3HV, float* activations_7_RJgsycQ8Eh ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 64;
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_XZVJnvrdB7 = tid * exec_range + range_iterator;
		for (size_t INDEX_pbETzgMrBr = 0; INDEX_pbETzgMrBr < 0 + 64; ++INDEX_pbETzgMrBr) {
			float act_ZPHLRRpFvR;
			float z_cXVPbtSJtP;
			z_cXVPbtSJtP = 0;
			act_ZPHLRRpFvR = 0;
			for (size_t INDEX_XIJmNHyjEA = 0; INDEX_XIJmNHyjEA < 0 + 64; ++INDEX_XIJmNHyjEA) {
				z_cXVPbtSJtP += weights_7_LJVmWXj3HV[64LL * (INDEX_pbETzgMrBr) + (INDEX_XIJmNHyjEA)] * activations_6_H4d7Vnb3tP[64LL * (INDEX_XZVJnvrdB7) + (INDEX_XIJmNHyjEA)];
			}
			float inlineFunctionValue_U8KM875cOn_vaX1bVWvIL;
			{
				float x_vaX1bVWvIL = z_cXVPbtSJtP;
				inlineFunctionValue_U8KM875cOn_vaX1bVWvIL = x_vaX1bVWvIL - (1.0 / 3.0) * (x_vaX1bVWvIL * x_vaX1bVWvIL * x_vaX1bVWvIL) + (2.0 / 15.0) * (x_vaX1bVWvIL * x_vaX1bVWvIL * x_vaX1bVWvIL * x_vaX1bVWvIL * x_vaX1bVWvIL) - (17 / 315) * (x_vaX1bVWvIL * x_vaX1bVWvIL * x_vaX1bVWvIL * x_vaX1bVWvIL * x_vaX1bVWvIL * x_vaX1bVWvIL * x_vaX1bVWvIL);
				goto STOP_LABEL_vaX1bVWvIL;
				STOP_LABEL_vaX1bVWvIL:
				noop;
			}
			act_ZPHLRRpFvR = inlineFunctionValue_U8KM875cOn_vaX1bVWvIL;
			activations_7_RJgsycQ8Eh[64LL * (INDEX_XZVJnvrdB7) + (INDEX_pbETzgMrBr)] = act_ZPHLRRpFvR;
		}
	}
}

void cuda_wrapper_forward_sXHzo9XR1x(float* activations_7_RJgsycQ8Eh, float* weights_8_exzdbedc44, float* result_lB8Y2jSu7K ) {

	kernel_cuda_wrapper_forward_sXHzo9XR1x<<<64, 64>>> (activations_7_RJgsycQ8Eh, weights_8_exzdbedc44, result_lB8Y2jSu7K );
}

__global__ 
void kernel_cuda_wrapper_forward_sXHzo9XR1x(float*  activations_7_RJgsycQ8Eh, float*  weights_8_exzdbedc44, float* result_lB8Y2jSu7K ) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int exec_range = 64;
	for ( int range_iterator = 0; range_iterator < exec_range + 0; range_iterator++) {
		int INDEX_sXHzo9XR1x = tid * exec_range + range_iterator;
		for (size_t INDEX_l1kYS9Z6re = 0; INDEX_l1kYS9Z6re < 0 + 10; ++INDEX_l1kYS9Z6re) {
			float act_vCzqiWvdiz;
			float z_lfZfv66gRx;
			z_lfZfv66gRx = 0;
			act_vCzqiWvdiz = 0;
			for (size_t INDEX_BS7XtEH4R0 = 0; INDEX_BS7XtEH4R0 < 0 + 64; ++INDEX_BS7XtEH4R0) {
				z_lfZfv66gRx += weights_8_exzdbedc44[64LL * (INDEX_l1kYS9Z6re) + (INDEX_BS7XtEH4R0)] * activations_7_RJgsycQ8Eh[64LL * (INDEX_sXHzo9XR1x) + (INDEX_BS7XtEH4R0)];
			}
			float inlineFunctionValue_U8KM875cOn_JLFDJKXLIQ;
			{
				float x_JLFDJKXLIQ = z_lfZfv66gRx;
				inlineFunctionValue_U8KM875cOn_JLFDJKXLIQ = x_JLFDJKXLIQ - (1.0 / 3.0) * (x_JLFDJKXLIQ * x_JLFDJKXLIQ * x_JLFDJKXLIQ) + (2.0 / 15.0) * (x_JLFDJKXLIQ * x_JLFDJKXLIQ * x_JLFDJKXLIQ * x_JLFDJKXLIQ * x_JLFDJKXLIQ) - (17 / 315) * (x_JLFDJKXLIQ * x_JLFDJKXLIQ * x_JLFDJKXLIQ * x_JLFDJKXLIQ * x_JLFDJKXLIQ * x_JLFDJKXLIQ * x_JLFDJKXLIQ);
				goto STOP_LABEL_JLFDJKXLIQ;
				STOP_LABEL_JLFDJKXLIQ:
				noop;
			}
			act_vCzqiWvdiz = inlineFunctionValue_U8KM875cOn_JLFDJKXLIQ;
			result_lB8Y2jSu7K[10LL * (INDEX_sXHzo9XR1x) + (INDEX_l1kYS9Z6re)] = act_vCzqiWvdiz;
		}
	}
}


