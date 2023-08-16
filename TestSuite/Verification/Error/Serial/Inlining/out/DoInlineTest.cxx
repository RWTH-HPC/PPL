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
#include "includes/DoInlineTest.hxx"
#include "includes/cuda_lib_DoInlineTest.hxx"
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

	int32_t* a_889ZrckpS1;
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
	a_889ZrckpS1 = Init_List(a_889ZrckpS1, 2LL * 1LL);
	if (rank == 0) {
		int32_t* input_CI6j44bSjB = copy(a_889ZrckpS1, 1LL * 2LL);
		int32_t inlineFunctionValue_MzzVVoMhFB_CI6j44bSjB;
		inlineFunctionValue_MzzVVoMhFB_CI6j44bSjB = 2;
		goto STOP_LABEL_CI6j44bSjB;
		STOP_LABEL_CI6j44bSjB:
		inlineFunctionValue_MzzVVoMhFB_CI6j44bSjB;
	}
	finishExecution();
	finishGPUExecution();
	std::free(a_889ZrckpS1);
	MPI_Finalize();
	return 0;


}
