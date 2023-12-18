
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <string>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstddef>
#include <sys/resource.h>

#ifndef CUDA_LIB_nn_HXX
#define CUDA_LIB_nn_HXX

#define MAX_INT (2147483647)


template<typename T, typename T1>
__host__ __device__
inline T* Init_List(T1 element, T* array, size_t n) {
    array = (T*) std::malloc(n * sizeof(T));
    for(size_t i = 0; i < n; i++) {
        array[i] = element;
    }
    return array;
}

template<typename T>
__host__ __device__
inline T* Init_List(T* array, size_t n) {
    array = (T*) std::malloc(n * sizeof(T));
    return array;
}

template<typename T>
__host__ __device__
inline int32_t Cast2Int(T input) {
    return static_cast<int32_t>(input);
}

template<typename T>
__host__ __device__
inline int16_t Cast2Int16(T input) {
    return static_cast<int16_t>(input);
}

template<typename T>
__host__ __device__
inline long Cast2Long(T input) {
    return static_cast<long>(input);
}

template<typename T>
__host__ __device__
inline int8_t Cast2Int8(T input) {
    return static_cast<int8_t>(input);
}

template<typename T>
__host__ __device__
inline float Cast2Float(T input) {
    return static_cast<float>(input);
}

template<typename T>
__host__ __device__
inline double Cast2Double(T input) {
    return static_cast<double>(input);
}

template<typename T>
__host__ __device__
inline std::string Cast2String(T input) {
    return std::to_string(input);
}

template<typename T>
__host__ __device__
inline T max(T first, T second) {
    if (first > second) {
        return first;
    }
    return second;
}


template<typename T>
__host__ __device__
inline T min(T first, T second) {
    if (first < second) {
        return first;
    }
    return second;
}

template<typename T>
__host__ __device__
inline T reduction_max(T initial, T* array, size_t n, size_t offset) {
    T result = initial;
    for (size_t i = offset; i < offset + n; ++i) {
         if (result > array[i]) {
            result = array[i];
         }
    }
    return result;
}

template<typename T>
__host__ __device__
inline T reduction_min(T initial, T* array, size_t n, size_t offset) {
    T result = initial;
    for (size_t i = offset; i < offset + n; ++i) {
         if (result < array[i]) {
            result = array[i];
         }
    }
    return result;
}

template<typename T>
__host__ __device__
inline T reduction_sum(T initial, T* array, size_t n, size_t offset) {
    T result = initial;
    for (size_t i = offset; i < offset + n; ++i) {
         result += array[i];
    }
    return result;
}

template<typename T>
__host__ __device__
inline T reduction_mult(T initial, T* array, size_t n, size_t offset) {
    T result = initial;
    for (size_t i = offset; i < offset + n; ++i) {
         result *= array[i];
    }
    return result;
}


template<typename T>
__host__ __device__
inline T* copy(T* array, size_t n, bool doRealCopy = false) {
    if (doRealCopy) {
        T* copied = (T*) std::malloc(n * sizeof(T));
        for(size_t i = 0; i < n; i++) {
            copied[i] = array[i];
        }
        return copied;
    }

    return array;
}

template<typename T>
__host__ __device__
inline void Set_Partial_Array( T result, T input, size_t n) {
     for(size_t i = 0; i < n; i++) {
        result[i] = input[i];
     }
}

template<typename T>
__host__ __device__
inline void Set_Partial_Array_Ref( T* result, T input, size_t n) {
     *result = input;
}

template<typename ...Args>
__host__
inline void print(Args... args) {
	(std::cout << ... << args) << std::endl;
}


__host__
inline long get_maxRusage() {
    struct rusage myEndRusage;
    getrusage(RUSAGE_SELF, &myEndRusage);
    return myEndRusage.ru_maxrss;
}


template<typename T>
__host__ __device__
inline bool Element_Exists_In_Vector(T element, T* array, size_t n) {

	for (size_t i = 0; i < n; i++) {
	    if (element == array[i]) {
	        return true;
	    }
	}
	return false;
}

template<typename ...Args>
__host__
inline void write(std::string file, Args... args) {

    std::ofstream myfile (file);
    if (myfile.is_open()) {
	    (myfile << ... << args) << std::endl;
	    myfile.close();
	} else {
	    std::cout << "Unable to open file" << std::endl;
	}
}

template<typename T>
__host__
inline void array_write(std::string file, T* array, size_t elements) {

    std::ofstream myfile (file);
    if (myfile.is_open()) {
        for ( size_t i = 0; i < elements; i++) {
	        myfile << array[i] << std::endl;
	    }
	    myfile.close();
	} else {
	    std::cout << "Unable to open file" << std::endl;
	}
}

template<typename T>
__global__
void cuda_alloc_wrapper(T** var, size_t length) {
    cudaMalloc(var, length);
}

template<typename T>
__global__
void cuda_dealloc_wrapper(T var) {
    cudaFree(var);
}

template<typename T, typename T1>
__global__
void cuda_host2device_wrapper(T* target, T1* source, size_t length) {
    cudaMemcpy(target, source, length, cudaMemcpyHostToDevice);
}

template<typename T, typename T1>
__global__
void cuda_device2host_wrapper(T1* target, T* source, size_t length) {
    cudaMemcpy(target, source, length, cudaMemcpyDeviceToHost);
}

void cuda_wrapper_rand_init_eeNUSUs3M2_WLX2ob9p0V(float* weights_6_SpUa0qY8sP );

void cuda_wrapper_rand_init_PlH53hIC5U_eDdEMWtudn(float* weights_4_WDk6GW21eM );

void cuda_wrapper_rand_init_eq1HFX6DF2_ePAri9DhVW(float* weights_3_aPXpgT7mlK );

void cuda_wrapper_rand_init_YvV3g1HMS2(float* weights_8_exzdbedc44 );

void cuda_wrapper_rand_init_ndBZfLHlBF_krQlq6Omgs(float* weights_2_PULl08Tzft );

void cuda_wrapper_rand_init_KeVA7GQAem_bzOG6cJAdH(float* weights_1_1gUuD5JLuk );

void cuda_wrapper_rand_init_VrA8skJMcp_N0ZhsA9PUC(float* weights_7_LJVmWXj3HV );

void cuda_wrapper_rand_init_png5hzlbR4_1Xe4UrPyk8(float* weights_5_dUKeqRRt9x );

void cuda_wrapper_forward_BBhRO6II0W_tKpfCHXWeK(float* weights_1_1gUuD5JLuk, float* batch_GkpriWeNHS, float* activations_1_fDjHO1CRW1 );

void cuda_wrapper_forward_F4SnxAXfxf_eed7oxtgNP(float* activations_1_fDjHO1CRW1, float* weights_2_PULl08Tzft, float* activations_2_j3fti3BN6j );

void cuda_wrapper_forward_qAsnKyB9OD_oSeefsNCRE(float* activations_2_j3fti3BN6j, float* weights_3_aPXpgT7mlK, float* activations_3_d2FySJrNO8 );

void cuda_wrapper_forward_HmKGZlyhss_2S7PyBeDBC(float* weights_4_WDk6GW21eM, float* activations_3_d2FySJrNO8, float* activations_4_UuAc29EIlG );

void cuda_wrapper_forward_ojXO3RetP1_rOG65YEVoL(float* activations_4_UuAc29EIlG, float* weights_5_dUKeqRRt9x, float* activations_5_gielxoWvAU );

void cuda_wrapper_forward_QemoMWU8Ac_d2Bsln4BkM(float* weights_6_SpUa0qY8sP, float* activations_5_gielxoWvAU, float* activations_6_H4d7Vnb3tP );

void cuda_wrapper_forward_l600xy7egX_XZVJnvrdB7(float* activations_6_H4d7Vnb3tP, float* weights_7_LJVmWXj3HV, float* activations_7_RJgsycQ8Eh );

void cuda_wrapper_forward_sXHzo9XR1x(float* activations_7_RJgsycQ8Eh, float* weights_8_exzdbedc44, float* result_lB8Y2jSu7K );



#endif
