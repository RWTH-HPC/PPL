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
#include "includes/RecursionTest.hxx"
#include "includes/cuda_lib_RecursionTest.hxx"
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

	int32_t result_RQS6fq0cnu;
	int32_t initial_hsDmWq7QjW;
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
	if (rank == 0) {
		initial_hsDmWq7QjW = 0;
		result_RQS6fq0cnu = 0;
	}
	result_RQS6fq0cnu = counter(initial_hsDmWq7QjW);
	if (rank == 0) {
		Bit_Mask * mask_ptr_ccXm2SXMpc = new Bit_Mask(48,true);
		for (size_t i_PBiM7LQ5BC = 0; i_PBiM7LQ5BC < 24; ++i_PBiM7LQ5BC) {
			mask_ptr_ccXm2SXMpc->setBarrier(i_PBiM7LQ5BC);
		}
		boost::shared_ptr<Bit_Mask>boost_mask_ptr_ccXm2SXMpc (mask_ptr_ccXm2SXMpc);
		self_barrier(boost_mask_ptr_ccXm2SXMpc);
	}
	if (rank == 0) {
		if (result_RQS6fq0cnu != 4) {
			print("Recursion not correct!");
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	}
	finishExecution();
	finishGPUExecution();
	MPI_Finalize();
	return 0;


}
