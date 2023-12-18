
#ifndef CUDA_LIB_nn_CUH
#define CUDA_LIB_nn_CUH



__device__ uint randomizerGPU(int n = 42) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(n, tid, 0, &state);
    return curand(&state);
}

__global__ 
void kernel_cuda_wrapper_rand_init_eeNUSUs3M2_WLX2ob9p0V(float* weights_6_SpUa0qY8sP );

__global__ 
void kernel_cuda_wrapper_rand_init_PlH53hIC5U_eDdEMWtudn(float* weights_4_WDk6GW21eM );

__global__ 
void kernel_cuda_wrapper_rand_init_eq1HFX6DF2_ePAri9DhVW(float* weights_3_aPXpgT7mlK );

__global__ 
void kernel_cuda_wrapper_rand_init_YvV3g1HMS2(float* weights_8_exzdbedc44 );

__global__ 
void kernel_cuda_wrapper_rand_init_ndBZfLHlBF_krQlq6Omgs(float* weights_2_PULl08Tzft );

__global__ 
void kernel_cuda_wrapper_rand_init_KeVA7GQAem_bzOG6cJAdH(float* weights_1_1gUuD5JLuk );

__global__ 
void kernel_cuda_wrapper_rand_init_VrA8skJMcp_N0ZhsA9PUC(float* weights_7_LJVmWXj3HV );

__global__ 
void kernel_cuda_wrapper_rand_init_png5hzlbR4_1Xe4UrPyk8(float* weights_5_dUKeqRRt9x );

__global__ 
void kernel_cuda_wrapper_forward_BBhRO6II0W_tKpfCHXWeK(float*  weights_1_1gUuD5JLuk, float*  batch_GkpriWeNHS, float* activations_1_fDjHO1CRW1 );

__global__ 
void kernel_cuda_wrapper_forward_F4SnxAXfxf_eed7oxtgNP(float*  activations_1_fDjHO1CRW1, float*  weights_2_PULl08Tzft, float* activations_2_j3fti3BN6j );

__global__ 
void kernel_cuda_wrapper_forward_qAsnKyB9OD_oSeefsNCRE(float*  activations_2_j3fti3BN6j, float*  weights_3_aPXpgT7mlK, float* activations_3_d2FySJrNO8 );

__global__ 
void kernel_cuda_wrapper_forward_HmKGZlyhss_2S7PyBeDBC(float*  weights_4_WDk6GW21eM, float*  activations_3_d2FySJrNO8, float* activations_4_UuAc29EIlG );

__global__ 
void kernel_cuda_wrapper_forward_ojXO3RetP1_rOG65YEVoL(float*  activations_4_UuAc29EIlG, float*  weights_5_dUKeqRRt9x, float* activations_5_gielxoWvAU );

__global__ 
void kernel_cuda_wrapper_forward_QemoMWU8Ac_d2Bsln4BkM(float*  weights_6_SpUa0qY8sP, float*  activations_5_gielxoWvAU, float* activations_6_H4d7Vnb3tP );

__global__ 
void kernel_cuda_wrapper_forward_l600xy7egX_XZVJnvrdB7(float*  activations_6_H4d7Vnb3tP, float*  weights_7_LJVmWXj3HV, float* activations_7_RJgsycQ8Eh );

__global__ 
void kernel_cuda_wrapper_forward_sXHzo9XR1x(float*  activations_7_RJgsycQ8Eh, float*  weights_8_exzdbedc44, float* result_lB8Y2jSu7K );



#endif
