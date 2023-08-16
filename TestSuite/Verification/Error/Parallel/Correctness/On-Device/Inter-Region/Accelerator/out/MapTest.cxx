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
#include "includes/MapTest.hxx"
#include "includes/cuda_lib_MapTest.hxx"
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

	int32_t* result_UcMz61H54z;
	int32_t* result_seq_Jz0g3vL0iC;
	int32_t* initial_s6xJ8pOmvy;
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
	initial_s6xJ8pOmvy = Init_List(1, initial_s6xJ8pOmvy, 200LL * 1LL);
	result_UcMz61H54z = Init_List(result_UcMz61H54z, 200LL * 1LL);
	result_seq_Jz0g3vL0iC = Init_List(result_seq_Jz0g3vL0iC, 200LL * 1LL);
	int32_t* GPU_Data_xs0MQrh9Si;
	if (rank == 0) {
		auto f_alloc_lFpNRlO63J = [&] () {
			cuda_alloc_wrapper(&GPU_Data_xs0MQrh9Si, sizeof(int32_t) * 100);
		};
		getGPUPool()->at(1).addWork(f_alloc_lFpNRlO63J);
	}
	if (rank == 0) {
		auto f_movement_iaW7Fcpm8e = [&] () {
			cuda_host2device_wrapper(&GPU_Data_xs0MQrh9Si[0], &initial_s6xJ8pOmvy[0], sizeof(int32_t) * 100);
		};
		getGPUPool()->at(1).addWork(f_movement_iaW7Fcpm8e);
	}
	int32_t* GPU_Data_UleetQv3Ys;
	if (rank == 0) {
		auto f_alloc_4mFMz2mDsY = [&] () {
			cuda_alloc_wrapper(&GPU_Data_UleetQv3Ys, sizeof(int32_t) * 100);
		};
		getGPUPool()->at(1).addWork(f_alloc_4mFMz2mDsY);
	}
	if (rank == 0) {
		auto f_gpu_N0ebKTuSmh = [&] () {
			cuda_wrapper_increment_N0ebKTuSmh(GPU_Data_xs0MQrh9Si, GPU_Data_UleetQv3Ys );
		};
		getGPUPool()->at(1).addWork(f_gpu_N0ebKTuSmh);
	}
	int32_t* GPU_Data_xgVeabfuiu;
	if (rank == 0) {
		auto f_alloc_mClneq1ZCE = [&] () {
			cuda_alloc_wrapper(&GPU_Data_xgVeabfuiu, sizeof(int32_t) * 100);
		};
		getGPUPool()->at(1).addWork(f_alloc_mClneq1ZCE);
	}
	if (rank == 0) {
		auto f_movement_oDpt1lWUjY = [&] () {
			cuda_host2device_wrapper(&GPU_Data_xgVeabfuiu[0], &initial_s6xJ8pOmvy[100], sizeof(int32_t) * 100);
		};
		getGPUPool()->at(1).addWork(f_movement_oDpt1lWUjY);
	}
	int32_t* GPU_Data_IyCLkbPWtt;
	if (rank == 0) {
		auto f_alloc_GmeTHvGznB = [&] () {
			cuda_alloc_wrapper(&GPU_Data_IyCLkbPWtt, sizeof(int32_t) * 100);
		};
		getGPUPool()->at(1).addWork(f_alloc_GmeTHvGznB);
	}
	if (rank == 0) {
		auto f_gpu_KZd7TM9SPp = [&] () {
			cuda_wrapper_increment_KZd7TM9SPp(GPU_Data_xgVeabfuiu, GPU_Data_IyCLkbPWtt );
		};
		getGPUPool()->at(1).addWork(f_gpu_KZd7TM9SPp);
	}
	if (rank == 0) {
		auto f_movement_X5F0nOUniE = [&] () {
			cuda_device2host_wrapper(&result_UcMz61H54z[100], &GPU_Data_IyCLkbPWtt[0], sizeof(int32_t) * 100);
		};
		getGPUPool()->at(1).addWork(f_movement_X5F0nOUniE);
	}
	if (rank == 0) {
		auto f_dealloc_R4SnCRUS9R = [&] () {
			cuda_dealloc_wrapper(GPU_Data_IyCLkbPWtt);
		};
		getGPUPool()->at(1).addWork(f_dealloc_R4SnCRUS9R);
	}
	if (rank == 0) {
		auto f_dealloc_KYXub3nW0I = [&] () {
			cuda_dealloc_wrapper(GPU_Data_xs0MQrh9Si);
		};
		getGPUPool()->at(1).addWork(f_dealloc_KYXub3nW0I);
	}
	if (rank == 0) {
		auto f_movement_xA0eV6xf2z = [&] () {
			cuda_device2host_wrapper(&result_UcMz61H54z[0], &GPU_Data_UleetQv3Ys[0], sizeof(int32_t) * 100);
		};
		getGPUPool()->at(1).addWork(f_movement_xA0eV6xf2z);
	}
	if (rank == 0) {
		auto f_dealloc_6yepsjx1tq = [&] () {
			cuda_dealloc_wrapper(GPU_Data_UleetQv3Ys);
		};
		getGPUPool()->at(1).addWork(f_dealloc_6yepsjx1tq);
	}
	if (rank == 0) {
		auto f_dealloc_fKTIguULIm = [&] () {
			cuda_dealloc_wrapper(GPU_Data_xgVeabfuiu);
		};
		getGPUPool()->at(1).addWork(f_dealloc_fKTIguULIm);
	}
	if (rank == 0) {
		Bit_Mask * mask_ptr_Qo9kmEcID4 = new Bit_Mask(2,true);
		mask_ptr_Qo9kmEcID4->setBarrier(1);
		boost::shared_ptr<Bit_Mask>boost_mask_ptr_Qo9kmEcID4 (mask_ptr_Qo9kmEcID4);
		cuda_sync_device(boost_mask_ptr_Qo9kmEcID4);
	}
	if (rank == 0) {
		for ( int32_t i = 0; i < 200; i++ ) {
			result_seq_Jz0g3vL0iC[(i)] = initial_s6xJ8pOmvy[(i)] + 1;
			if (result_seq_Jz0g3vL0iC[(i)] != result_UcMz61H54z[(i)]) {
				print("value at element ", i, " is wrong!");
				MPI_Abort(MPI_COMM_WORLD, 1);
			}
		}
	}
	finishExecution();
	finishGPUExecution();
	std::free(result_UcMz61H54z);
	std::free(result_seq_Jz0g3vL0iC);
	std::free(initial_s6xJ8pOmvy);
	MPI_Finalize();
	return 0;


}
