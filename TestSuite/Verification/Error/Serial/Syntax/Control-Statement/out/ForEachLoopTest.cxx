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
#include "includes/ForEachLoopTest.hxx"
#include "includes/cuda_lib_ForEachLoopTest.hxx"
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

	int32_t* a_6vHQn2SNjl;
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
	a_6vHQn2SNjl = Init_List(a_6vHQn2SNjl, 10LL * 1LL);
	if (rank == 0) {
		int32_t* forEachLoopNodeArray_DuWVdhXOIz = a_6vHQn2SNjl;
		for ( size_t forEachLoopCounter_DuWVdhXOIz = 0; forEachLoopCounter_DuWVdhXOIz < 10; forEachLoopCounter_DuWVdhXOIz++ ) {
			int32_t* x_DuWVdhXOIz = &forEachLoopNodeArray_DuWVdhXOIz[forEachLoopCounter_DuWVdhXOIz ];
			x_DuWVdhXOIz[0] = 1;
		}
	}
	finishExecution();
	finishGPUExecution();
	std::free(a_6vHQn2SNjl);
	MPI_Finalize();
	return 0;


}
