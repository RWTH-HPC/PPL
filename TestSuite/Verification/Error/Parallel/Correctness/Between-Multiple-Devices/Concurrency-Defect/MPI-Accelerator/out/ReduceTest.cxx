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
#include "includes/ReduceTest.hxx"
#include "includes/cuda_lib_ReduceTest.hxx"
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

	int32_t result_YbFznPajHi;
	int32_t* initial_pb7pYYGUGI;
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
	initial_pb7pYYGUGI = Init_List(1, initial_pb7pYYGUGI, 200LL * 1LL);
	if (rank == 0) {
		result_YbFznPajHi = 0;
	}
	if (rank == 0) {
		MPI_Send(&initial_pb7pYYGUGI[100], 100, MPI_INT, 1, 0, MPI_COMM_WORLD);
	} else if (rank == 1) {
		MPI_Recv(&initial_pb7pYYGUGI[100], 100, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	int32_t* GPU_Data_aUPlKfKq8b;
	if (rank == 0) {
		auto f_alloc_95Bo5LZNdD = [&] () {
			cuda_alloc_wrapper(&GPU_Data_aUPlKfKq8b, sizeof(int32_t) * 100);
		};
		getGPUPool()->at(1).addWork(f_alloc_95Bo5LZNdD);
	}
	if (rank == 0) {
		auto f_movement_zJg3bW8WF8 = [&] () {
			cuda_host2device_wrapper(&GPU_Data_aUPlKfKq8b[0], &initial_pb7pYYGUGI[0], sizeof(int32_t) * 100);
		};
		getGPUPool()->at(1).addWork(f_movement_zJg3bW8WF8);
	}
	int32_t* GPU_Data_q5BuOmVVAb;
	if (rank == 0) {
		auto f_alloc_dYahB36aG9 = [&] () {
			cuda_alloc_wrapper(&GPU_Data_q5BuOmVVAb, sizeof(int32_t) * 1);
		};
		getGPUPool()->at(1).addWork(f_alloc_dYahB36aG9);
	}
	pthread_mutex_t reduction_lock_Ie7GuQ5GJh = PTHREAD_MUTEX_INITIALIZER;
	int32_t temp_data_Ie7GuQ5GJh = 0;
	if (rank == 0) {
		auto f_gpu_G67QyBpe5H = [&] () {
			cuda_wrapper_sum_G67QyBpe5H(GPU_Data_aUPlKfKq8b, &temp_data_Ie7GuQ5GJh, reduction_lock_Ie7GuQ5GJh);
		};
		getGPUPool()->at(1).addWork(f_gpu_G67QyBpe5H);
	}
	int32_t* GPU_Data_6iEzKs5xse;
	if (rank == 1) {
		auto f_alloc_6mJB1XnAYu = [&] () {
			cuda_alloc_wrapper(&GPU_Data_6iEzKs5xse, sizeof(int32_t) * 100);
		};
		getGPUPool()->at(1).addWork(f_alloc_6mJB1XnAYu);
	}
	if (rank == 1) {
		auto f_movement_5l5ndrTIOB = [&] () {
			cuda_host2device_wrapper(&GPU_Data_6iEzKs5xse[0], &initial_pb7pYYGUGI[100], sizeof(int32_t) * 100);
		};
		getGPUPool()->at(1).addWork(f_movement_5l5ndrTIOB);
	}
	int32_t* GPU_Data_AO5E9xWOS6;
	if (rank == 1) {
		auto f_alloc_y6g7HRMrVN = [&] () {
			cuda_alloc_wrapper(&GPU_Data_AO5E9xWOS6, sizeof(int32_t) * 1);
		};
		getGPUPool()->at(1).addWork(f_alloc_y6g7HRMrVN);
	}
	if (rank == 1) {
		auto f_gpu_ZeOzFvempS = [&] () {
			cuda_wrapper_sum_ZeOzFvempS(GPU_Data_6iEzKs5xse, &temp_data_Ie7GuQ5GJh, reduction_lock_Ie7GuQ5GJh);
		};
		getGPUPool()->at(1).addWork(f_gpu_ZeOzFvempS);
	}
	if (rank == 0) {
		auto f_dealloc_nfbt8lX0W9 = [&] () {
			cuda_dealloc_wrapper(GPU_Data_q5BuOmVVAb);
		};
		getGPUPool()->at(1).addWork(f_dealloc_nfbt8lX0W9);
	}
	if (rank == 1) {
		auto f_dealloc_UeDA0DkXrl = [&] () {
			cuda_dealloc_wrapper(GPU_Data_6iEzKs5xse);
		};
		getGPUPool()->at(1).addWork(f_dealloc_UeDA0DkXrl);
	}
	if (rank == 0) {
		auto f_dealloc_2aXODqXQrh = [&] () {
			cuda_dealloc_wrapper(GPU_Data_aUPlKfKq8b);
		};
		getGPUPool()->at(1).addWork(f_dealloc_2aXODqXQrh);
	}
	if (rank == 1) {
		auto f_dealloc_QbfOXUHoSF = [&] () {
			cuda_dealloc_wrapper(GPU_Data_AO5E9xWOS6);
		};
		getGPUPool()->at(1).addWork(f_dealloc_QbfOXUHoSF);
	}
	if (rank == 0) {
		Bit_Mask * mask_ptr_KldzoLq2Cl = new Bit_Mask(2,true);
		mask_ptr_KldzoLq2Cl->setBarrier(1);
		boost::shared_ptr<Bit_Mask>boost_mask_ptr_KldzoLq2Cl (mask_ptr_KldzoLq2Cl);
		cuda_sync_device(boost_mask_ptr_KldzoLq2Cl);
	}
	if (rank == 1) {
		Bit_Mask * mask_ptr_6f1BleoUtz = new Bit_Mask(2,true);
		mask_ptr_6f1BleoUtz->setBarrier(1);
		boost::shared_ptr<Bit_Mask>boost_mask_ptr_6f1BleoUtz (mask_ptr_6f1BleoUtz);
		cuda_sync_device(boost_mask_ptr_6f1BleoUtz);
	}
	int32_t MPI_Reduction_Combiner_3JWuNXyR6E = 0;
	MPI_Reduce(&temp_data_Ie7GuQ5GJh, &MPI_Reduction_Combiner_3JWuNXyR6E, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	if (rank == 0) {
		result_YbFznPajHi = MPI_Reduction_Combiner_3JWuNXyR6E;
	}
	if (rank == 0) {
		if (result_YbFznPajHi != 200) {
			print("summation not correct!");
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	}
	finishExecution();
	finishGPUExecution();
	std::free(initial_pb7pYYGUGI);
	MPI_Finalize();
	return 0;


}
