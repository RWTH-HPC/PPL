/*********************************************************************/
/*        This is a generated C++ source file.                       */
/*        Generated by PatternDSL.                                   */
/*********************************************************************/

#include <vector>
#include <cstdint>
#include <string>
#include <limits>
#include <iostream>
#include "includes/Patternlib.hxx"
#include "includes/PThreadsLib.hxx"
#include "includes/ReduceNestedTest.hxx"
#include "includes/cuda_lib_ReduceNestedTest.hxx"
#include "mpi.h"
#include "includes/cuda_pool_lib.hxx"
#include "math.h"







/*********************************************************************/
/*        Global Variables                                           */
/*        Generated by PatternDSL .                                  */
/*********************************************************************/



/*********************************************************************/
/*        Function Declarations                                      */
/*        Generated by PatternDSL.                                   */
/*********************************************************************/



int main(int argc, char** argv) {

	int32_t* result_mrCUu80Ufm;
	int32_t* result_seq_DqsAVByZz5;
	int32_t* initial_Cro7ZoaMZD;
	int NUM_CORES;
	int NUM_GPUS;

	int rank, nprocs;
	MPI_Status Stat;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (rank == 0 ) {
		NUM_CORES = 48;
		NUM_GPUS = 2;
	} else 	if (rank == 1 ) {
		NUM_CORES = 48;
		NUM_GPUS = 2;
	}
	std::vector<Thread> pool(NUM_CORES);
	setPool(&pool);
	std::vector<Thread> gpu_pool(NUM_GPUS);
	setGPUPool(&gpu_pool);
	startExecution();
	startGPUExecution();
	initial_Cro7ZoaMZD = Init_List(1, initial_Cro7ZoaMZD, 200LL * 200LL * 1LL);
	result_mrCUu80Ufm = Init_List(result_mrCUu80Ufm, 200LL * 1LL);
	result_seq_DqsAVByZz5 = Init_List(0, result_seq_DqsAVByZz5, 200LL * 1LL);
	int32_t* GPU_Data_hzfDbA1G9H;
	if (rank == 0) {
		auto f_alloc_1VPGYPWLOG = [&] () {
			cuda_alloc_wrapper(&GPU_Data_hzfDbA1G9H, sizeof(int32_t) * 40000);
		};
		getGPUPool()->at(1).addWork(f_alloc_1VPGYPWLOG);
	}
	if (rank == 0) {
		auto f_movement_eagGSQ23Sg = [&] () {
			cuda_host2device_wrapper(&GPU_Data_hzfDbA1G9H[0], &initial_Cro7ZoaMZD[0], sizeof(int32_t) * 40000);
		};
		getGPUPool()->at(1).addWork(f_movement_eagGSQ23Sg);
	}
	int32_t* GPU_Data_2tdFstkg1L;
	if (rank == 0) {
		auto f_alloc_bzFKrSP1bp = [&] () {
			cuda_alloc_wrapper(&GPU_Data_2tdFstkg1L, sizeof(int32_t) * 200);
		};
		getGPUPool()->at(1).addWork(f_alloc_bzFKrSP1bp);
	}
	if (rank == 0) {
		auto f_gpu_JTkoi9qzzS = [&] () {
			cuda_wrapper_rowSum_JTkoi9qzzS(GPU_Data_hzfDbA1G9H, GPU_Data_2tdFstkg1L );
		};
		getGPUPool()->at(1).addWork(f_gpu_JTkoi9qzzS);
	}
	if (rank == 0) {
		auto f_movement_YNRNGB0SyN = [&] () {
			cuda_device2host_wrapper(&result_mrCUu80Ufm[0], &GPU_Data_2tdFstkg1L[0], sizeof(int32_t) * 200);
		};
		getGPUPool()->at(1).addWork(f_movement_YNRNGB0SyN);
	}
	if (rank == 0) {
		auto f_dealloc_0zTkMblgll = [&] () {
			cuda_dealloc_wrapper(GPU_Data_2tdFstkg1L);
		};
		getGPUPool()->at(1).addWork(f_dealloc_0zTkMblgll);
	}
	if (rank == 0) {
		auto f_dealloc_r99lDG3YuK = [&] () {
			cuda_dealloc_wrapper(GPU_Data_hzfDbA1G9H);
		};
		getGPUPool()->at(1).addWork(f_dealloc_r99lDG3YuK);
	}
	if (rank == 0) {
		Bit_Mask * mask_ptr_RVETAExc6N = new Bit_Mask(2,true);
		mask_ptr_RVETAExc6N->setBarrier(1);
		boost::shared_ptr<Bit_Mask>boost_mask_ptr_RVETAExc6N (mask_ptr_RVETAExc6N);
		cuda_sync_device(boost_mask_ptr_RVETAExc6N);
	}
	if (rank == 0) {
		for ( int32_t i = 0; i < 200; i++ ) {
			for ( int32_t j = 0; j < 200; j++ ) {
				result_seq_DqsAVByZz5[(i)] += initial_Cro7ZoaMZD[200LL * (i) + (j)] + 1;
			}
			if (result_seq_DqsAVByZz5[(i)] != result_mrCUu80Ufm[(i)]) {
				print("value at element ", i, " is wrong!");
				MPI_Abort(MPI_COMM_WORLD, 1);
			}
		}
	}
	finishExecution();
	finishGPUExecution();
	std::free(result_mrCUu80Ufm);
	std::free(result_seq_DqsAVByZz5);
	std::free(initial_Cro7ZoaMZD);
	MPI_Finalize();
	return 0;


}
