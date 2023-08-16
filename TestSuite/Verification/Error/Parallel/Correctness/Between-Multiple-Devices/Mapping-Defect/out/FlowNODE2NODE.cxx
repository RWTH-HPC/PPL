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
#include "includes/FlowNODE2NODE.hxx"
#include "includes/cuda_lib_FlowNODE2NODE.hxx"
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

	int32_t* result_PdjdsjaWG5;
	int32_t* result_seq_t8b2E1gtJz;
	int32_t* initial_GFPC2vAxPV;
	int32_t* intermediate_evYkRRKdLJ;
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
	initial_GFPC2vAxPV = Init_List(1, initial_GFPC2vAxPV, 200LL * 1LL);
	intermediate_evYkRRKdLJ = Init_List(intermediate_evYkRRKdLJ, 200LL * 1LL);
	result_PdjdsjaWG5 = Init_List(result_PdjdsjaWG5, 200LL * 1LL);
	result_seq_t8b2E1gtJz = Init_List(result_seq_t8b2E1gtJz, 200LL * 1LL);
	if (rank == 0) {
		auto f_0_14kQeLOQlF = [&] () {
			for (size_t INDEX_14kQeLOQlF = 0; INDEX_14kQeLOQlF < 0 + 8; ++INDEX_14kQeLOQlF) {
				intermediate_evYkRRKdLJ[(INDEX_14kQeLOQlF)] = initial_GFPC2vAxPV[(INDEX_14kQeLOQlF)] + 1;
			}
		};
		getPool()->at(0).addWork(f_0_14kQeLOQlF);
		auto f_1_14kQeLOQlF = [&] () {
			for (size_t INDEX_14kQeLOQlF = 8; INDEX_14kQeLOQlF < 8 + 8; ++INDEX_14kQeLOQlF) {
				intermediate_evYkRRKdLJ[(INDEX_14kQeLOQlF)] = initial_GFPC2vAxPV[(INDEX_14kQeLOQlF)] + 1;
			}
		};
		getPool()->at(1).addWork(f_1_14kQeLOQlF);
		auto f_2_14kQeLOQlF = [&] () {
			for (size_t INDEX_14kQeLOQlF = 16; INDEX_14kQeLOQlF < 16 + 8; ++INDEX_14kQeLOQlF) {
				intermediate_evYkRRKdLJ[(INDEX_14kQeLOQlF)] = initial_GFPC2vAxPV[(INDEX_14kQeLOQlF)] + 1;
			}
		};
		getPool()->at(2).addWork(f_2_14kQeLOQlF);
		auto f_3_14kQeLOQlF = [&] () {
			for (size_t INDEX_14kQeLOQlF = 24; INDEX_14kQeLOQlF < 24 + 8; ++INDEX_14kQeLOQlF) {
				intermediate_evYkRRKdLJ[(INDEX_14kQeLOQlF)] = initial_GFPC2vAxPV[(INDEX_14kQeLOQlF)] + 1;
			}
		};
		getPool()->at(3).addWork(f_3_14kQeLOQlF);
		auto f_4_14kQeLOQlF = [&] () {
			for (size_t INDEX_14kQeLOQlF = 32; INDEX_14kQeLOQlF < 32 + 8; ++INDEX_14kQeLOQlF) {
				intermediate_evYkRRKdLJ[(INDEX_14kQeLOQlF)] = initial_GFPC2vAxPV[(INDEX_14kQeLOQlF)] + 1;
			}
		};
		getPool()->at(4).addWork(f_4_14kQeLOQlF);
		auto f_5_14kQeLOQlF = [&] () {
			for (size_t INDEX_14kQeLOQlF = 40; INDEX_14kQeLOQlF < 40 + 8; ++INDEX_14kQeLOQlF) {
				intermediate_evYkRRKdLJ[(INDEX_14kQeLOQlF)] = initial_GFPC2vAxPV[(INDEX_14kQeLOQlF)] + 1;
			}
		};
		getPool()->at(5).addWork(f_5_14kQeLOQlF);
		auto f_6_14kQeLOQlF = [&] () {
			for (size_t INDEX_14kQeLOQlF = 48; INDEX_14kQeLOQlF < 48 + 8; ++INDEX_14kQeLOQlF) {
				intermediate_evYkRRKdLJ[(INDEX_14kQeLOQlF)] = initial_GFPC2vAxPV[(INDEX_14kQeLOQlF)] + 1;
			}
		};
		getPool()->at(6).addWork(f_6_14kQeLOQlF);
		auto f_7_14kQeLOQlF = [&] () {
			for (size_t INDEX_14kQeLOQlF = 56; INDEX_14kQeLOQlF < 56 + 8; ++INDEX_14kQeLOQlF) {
				intermediate_evYkRRKdLJ[(INDEX_14kQeLOQlF)] = initial_GFPC2vAxPV[(INDEX_14kQeLOQlF)] + 1;
			}
		};
		getPool()->at(7).addWork(f_7_14kQeLOQlF);
		auto f_8_14kQeLOQlF = [&] () {
			for (size_t INDEX_14kQeLOQlF = 64; INDEX_14kQeLOQlF < 64 + 8; ++INDEX_14kQeLOQlF) {
				intermediate_evYkRRKdLJ[(INDEX_14kQeLOQlF)] = initial_GFPC2vAxPV[(INDEX_14kQeLOQlF)] + 1;
			}
		};
		getPool()->at(8).addWork(f_8_14kQeLOQlF);
		auto f_9_14kQeLOQlF = [&] () {
			for (size_t INDEX_14kQeLOQlF = 72; INDEX_14kQeLOQlF < 72 + 8; ++INDEX_14kQeLOQlF) {
				intermediate_evYkRRKdLJ[(INDEX_14kQeLOQlF)] = initial_GFPC2vAxPV[(INDEX_14kQeLOQlF)] + 1;
			}
		};
		getPool()->at(9).addWork(f_9_14kQeLOQlF);
		auto f_10_14kQeLOQlF = [&] () {
			for (size_t INDEX_14kQeLOQlF = 80; INDEX_14kQeLOQlF < 80 + 8; ++INDEX_14kQeLOQlF) {
				intermediate_evYkRRKdLJ[(INDEX_14kQeLOQlF)] = initial_GFPC2vAxPV[(INDEX_14kQeLOQlF)] + 1;
			}
		};
		getPool()->at(10).addWork(f_10_14kQeLOQlF);
		auto f_11_14kQeLOQlF = [&] () {
			for (size_t INDEX_14kQeLOQlF = 88; INDEX_14kQeLOQlF < 88 + 8; ++INDEX_14kQeLOQlF) {
				intermediate_evYkRRKdLJ[(INDEX_14kQeLOQlF)] = initial_GFPC2vAxPV[(INDEX_14kQeLOQlF)] + 1;
			}
		};
		getPool()->at(11).addWork(f_11_14kQeLOQlF);
		auto f_12_14kQeLOQlF = [&] () {
			for (size_t INDEX_14kQeLOQlF = 96; INDEX_14kQeLOQlF < 96 + 8; ++INDEX_14kQeLOQlF) {
				intermediate_evYkRRKdLJ[(INDEX_14kQeLOQlF)] = initial_GFPC2vAxPV[(INDEX_14kQeLOQlF)] + 1;
			}
		};
		getPool()->at(12).addWork(f_12_14kQeLOQlF);
		auto f_13_14kQeLOQlF = [&] () {
			for (size_t INDEX_14kQeLOQlF = 104; INDEX_14kQeLOQlF < 104 + 8; ++INDEX_14kQeLOQlF) {
				intermediate_evYkRRKdLJ[(INDEX_14kQeLOQlF)] = initial_GFPC2vAxPV[(INDEX_14kQeLOQlF)] + 1;
			}
		};
		getPool()->at(13).addWork(f_13_14kQeLOQlF);
		auto f_14_14kQeLOQlF = [&] () {
			for (size_t INDEX_14kQeLOQlF = 112; INDEX_14kQeLOQlF < 112 + 8; ++INDEX_14kQeLOQlF) {
				intermediate_evYkRRKdLJ[(INDEX_14kQeLOQlF)] = initial_GFPC2vAxPV[(INDEX_14kQeLOQlF)] + 1;
			}
		};
		getPool()->at(14).addWork(f_14_14kQeLOQlF);
		auto f_15_14kQeLOQlF = [&] () {
			for (size_t INDEX_14kQeLOQlF = 120; INDEX_14kQeLOQlF < 120 + 8; ++INDEX_14kQeLOQlF) {
				intermediate_evYkRRKdLJ[(INDEX_14kQeLOQlF)] = initial_GFPC2vAxPV[(INDEX_14kQeLOQlF)] + 1;
			}
		};
		getPool()->at(15).addWork(f_15_14kQeLOQlF);
		auto f_16_14kQeLOQlF = [&] () {
			for (size_t INDEX_14kQeLOQlF = 128; INDEX_14kQeLOQlF < 128 + 8; ++INDEX_14kQeLOQlF) {
				intermediate_evYkRRKdLJ[(INDEX_14kQeLOQlF)] = initial_GFPC2vAxPV[(INDEX_14kQeLOQlF)] + 1;
			}
		};
		getPool()->at(16).addWork(f_16_14kQeLOQlF);
		auto f_17_14kQeLOQlF = [&] () {
			for (size_t INDEX_14kQeLOQlF = 136; INDEX_14kQeLOQlF < 136 + 8; ++INDEX_14kQeLOQlF) {
				intermediate_evYkRRKdLJ[(INDEX_14kQeLOQlF)] = initial_GFPC2vAxPV[(INDEX_14kQeLOQlF)] + 1;
			}
		};
		getPool()->at(17).addWork(f_17_14kQeLOQlF);
		auto f_18_14kQeLOQlF = [&] () {
			for (size_t INDEX_14kQeLOQlF = 144; INDEX_14kQeLOQlF < 144 + 8; ++INDEX_14kQeLOQlF) {
				intermediate_evYkRRKdLJ[(INDEX_14kQeLOQlF)] = initial_GFPC2vAxPV[(INDEX_14kQeLOQlF)] + 1;
			}
		};
		getPool()->at(18).addWork(f_18_14kQeLOQlF);
		auto f_19_14kQeLOQlF = [&] () {
			for (size_t INDEX_14kQeLOQlF = 152; INDEX_14kQeLOQlF < 152 + 8; ++INDEX_14kQeLOQlF) {
				intermediate_evYkRRKdLJ[(INDEX_14kQeLOQlF)] = initial_GFPC2vAxPV[(INDEX_14kQeLOQlF)] + 1;
			}
		};
		getPool()->at(19).addWork(f_19_14kQeLOQlF);
		auto f_20_14kQeLOQlF = [&] () {
			for (size_t INDEX_14kQeLOQlF = 160; INDEX_14kQeLOQlF < 160 + 8; ++INDEX_14kQeLOQlF) {
				intermediate_evYkRRKdLJ[(INDEX_14kQeLOQlF)] = initial_GFPC2vAxPV[(INDEX_14kQeLOQlF)] + 1;
			}
		};
		getPool()->at(20).addWork(f_20_14kQeLOQlF);
		auto f_21_14kQeLOQlF = [&] () {
			for (size_t INDEX_14kQeLOQlF = 168; INDEX_14kQeLOQlF < 168 + 8; ++INDEX_14kQeLOQlF) {
				intermediate_evYkRRKdLJ[(INDEX_14kQeLOQlF)] = initial_GFPC2vAxPV[(INDEX_14kQeLOQlF)] + 1;
			}
		};
		getPool()->at(21).addWork(f_21_14kQeLOQlF);
		auto f_22_14kQeLOQlF = [&] () {
			for (size_t INDEX_14kQeLOQlF = 176; INDEX_14kQeLOQlF < 176 + 8; ++INDEX_14kQeLOQlF) {
				intermediate_evYkRRKdLJ[(INDEX_14kQeLOQlF)] = initial_GFPC2vAxPV[(INDEX_14kQeLOQlF)] + 1;
			}
		};
		getPool()->at(22).addWork(f_22_14kQeLOQlF);
		auto f_23_14kQeLOQlF = [&] () {
			for (size_t INDEX_14kQeLOQlF = 184; INDEX_14kQeLOQlF < 184 + 16; ++INDEX_14kQeLOQlF) {
				intermediate_evYkRRKdLJ[(INDEX_14kQeLOQlF)] = initial_GFPC2vAxPV[(INDEX_14kQeLOQlF)] + 1;
			}
		};
		getPool()->at(23).addWork(f_23_14kQeLOQlF);
	}
	if (rank == 0) {
		Bit_Mask * mask_ptr_eyWlHRQ21g = new Bit_Mask(48,true);
		for (size_t i_N6Yirn3JFc = 0; i_N6Yirn3JFc < 24; ++i_N6Yirn3JFc) {
			mask_ptr_eyWlHRQ21g->setBarrier(i_N6Yirn3JFc);
		}
		boost::shared_ptr<Bit_Mask>boost_mask_ptr_eyWlHRQ21g (mask_ptr_eyWlHRQ21g);
		self_barrier(boost_mask_ptr_eyWlHRQ21g);
	}
	if (rank == 0) {
		MPI_Send(&intermediate_evYkRRKdLJ[0], 200, MPI_INT, 1, 0, MPI_COMM_WORLD);
	} else if (rank == 1) {
		MPI_Recv(&intermediate_evYkRRKdLJ[0], 200, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	if (rank == 1) {
		auto f_0_aXAUbP7iHY = [&] () {
			for (size_t INDEX_aXAUbP7iHY = 0; INDEX_aXAUbP7iHY < 0 + 8; ++INDEX_aXAUbP7iHY) {
				result_PdjdsjaWG5[(INDEX_aXAUbP7iHY)] = intermediate_evYkRRKdLJ[(INDEX_aXAUbP7iHY)] + 1;
			}
		};
		getPool()->at(0).addWork(f_0_aXAUbP7iHY);
		auto f_1_aXAUbP7iHY = [&] () {
			for (size_t INDEX_aXAUbP7iHY = 8; INDEX_aXAUbP7iHY < 8 + 8; ++INDEX_aXAUbP7iHY) {
				result_PdjdsjaWG5[(INDEX_aXAUbP7iHY)] = intermediate_evYkRRKdLJ[(INDEX_aXAUbP7iHY)] + 1;
			}
		};
		getPool()->at(1).addWork(f_1_aXAUbP7iHY);
		auto f_2_aXAUbP7iHY = [&] () {
			for (size_t INDEX_aXAUbP7iHY = 16; INDEX_aXAUbP7iHY < 16 + 8; ++INDEX_aXAUbP7iHY) {
				result_PdjdsjaWG5[(INDEX_aXAUbP7iHY)] = intermediate_evYkRRKdLJ[(INDEX_aXAUbP7iHY)] + 1;
			}
		};
		getPool()->at(2).addWork(f_2_aXAUbP7iHY);
		auto f_3_aXAUbP7iHY = [&] () {
			for (size_t INDEX_aXAUbP7iHY = 24; INDEX_aXAUbP7iHY < 24 + 8; ++INDEX_aXAUbP7iHY) {
				result_PdjdsjaWG5[(INDEX_aXAUbP7iHY)] = intermediate_evYkRRKdLJ[(INDEX_aXAUbP7iHY)] + 1;
			}
		};
		getPool()->at(3).addWork(f_3_aXAUbP7iHY);
		auto f_4_aXAUbP7iHY = [&] () {
			for (size_t INDEX_aXAUbP7iHY = 32; INDEX_aXAUbP7iHY < 32 + 8; ++INDEX_aXAUbP7iHY) {
				result_PdjdsjaWG5[(INDEX_aXAUbP7iHY)] = intermediate_evYkRRKdLJ[(INDEX_aXAUbP7iHY)] + 1;
			}
		};
		getPool()->at(4).addWork(f_4_aXAUbP7iHY);
		auto f_5_aXAUbP7iHY = [&] () {
			for (size_t INDEX_aXAUbP7iHY = 40; INDEX_aXAUbP7iHY < 40 + 8; ++INDEX_aXAUbP7iHY) {
				result_PdjdsjaWG5[(INDEX_aXAUbP7iHY)] = intermediate_evYkRRKdLJ[(INDEX_aXAUbP7iHY)] + 1;
			}
		};
		getPool()->at(5).addWork(f_5_aXAUbP7iHY);
		auto f_6_aXAUbP7iHY = [&] () {
			for (size_t INDEX_aXAUbP7iHY = 48; INDEX_aXAUbP7iHY < 48 + 8; ++INDEX_aXAUbP7iHY) {
				result_PdjdsjaWG5[(INDEX_aXAUbP7iHY)] = intermediate_evYkRRKdLJ[(INDEX_aXAUbP7iHY)] + 1;
			}
		};
		getPool()->at(6).addWork(f_6_aXAUbP7iHY);
		auto f_7_aXAUbP7iHY = [&] () {
			for (size_t INDEX_aXAUbP7iHY = 56; INDEX_aXAUbP7iHY < 56 + 8; ++INDEX_aXAUbP7iHY) {
				result_PdjdsjaWG5[(INDEX_aXAUbP7iHY)] = intermediate_evYkRRKdLJ[(INDEX_aXAUbP7iHY)] + 1;
			}
		};
		getPool()->at(7).addWork(f_7_aXAUbP7iHY);
		auto f_8_aXAUbP7iHY = [&] () {
			for (size_t INDEX_aXAUbP7iHY = 64; INDEX_aXAUbP7iHY < 64 + 8; ++INDEX_aXAUbP7iHY) {
				result_PdjdsjaWG5[(INDEX_aXAUbP7iHY)] = intermediate_evYkRRKdLJ[(INDEX_aXAUbP7iHY)] + 1;
			}
		};
		getPool()->at(8).addWork(f_8_aXAUbP7iHY);
		auto f_9_aXAUbP7iHY = [&] () {
			for (size_t INDEX_aXAUbP7iHY = 72; INDEX_aXAUbP7iHY < 72 + 8; ++INDEX_aXAUbP7iHY) {
				result_PdjdsjaWG5[(INDEX_aXAUbP7iHY)] = intermediate_evYkRRKdLJ[(INDEX_aXAUbP7iHY)] + 1;
			}
		};
		getPool()->at(9).addWork(f_9_aXAUbP7iHY);
		auto f_10_aXAUbP7iHY = [&] () {
			for (size_t INDEX_aXAUbP7iHY = 80; INDEX_aXAUbP7iHY < 80 + 8; ++INDEX_aXAUbP7iHY) {
				result_PdjdsjaWG5[(INDEX_aXAUbP7iHY)] = intermediate_evYkRRKdLJ[(INDEX_aXAUbP7iHY)] + 1;
			}
		};
		getPool()->at(10).addWork(f_10_aXAUbP7iHY);
		auto f_11_aXAUbP7iHY = [&] () {
			for (size_t INDEX_aXAUbP7iHY = 88; INDEX_aXAUbP7iHY < 88 + 8; ++INDEX_aXAUbP7iHY) {
				result_PdjdsjaWG5[(INDEX_aXAUbP7iHY)] = intermediate_evYkRRKdLJ[(INDEX_aXAUbP7iHY)] + 1;
			}
		};
		getPool()->at(11).addWork(f_11_aXAUbP7iHY);
		auto f_12_aXAUbP7iHY = [&] () {
			for (size_t INDEX_aXAUbP7iHY = 96; INDEX_aXAUbP7iHY < 96 + 8; ++INDEX_aXAUbP7iHY) {
				result_PdjdsjaWG5[(INDEX_aXAUbP7iHY)] = intermediate_evYkRRKdLJ[(INDEX_aXAUbP7iHY)] + 1;
			}
		};
		getPool()->at(12).addWork(f_12_aXAUbP7iHY);
		auto f_13_aXAUbP7iHY = [&] () {
			for (size_t INDEX_aXAUbP7iHY = 104; INDEX_aXAUbP7iHY < 104 + 8; ++INDEX_aXAUbP7iHY) {
				result_PdjdsjaWG5[(INDEX_aXAUbP7iHY)] = intermediate_evYkRRKdLJ[(INDEX_aXAUbP7iHY)] + 1;
			}
		};
		getPool()->at(13).addWork(f_13_aXAUbP7iHY);
		auto f_14_aXAUbP7iHY = [&] () {
			for (size_t INDEX_aXAUbP7iHY = 112; INDEX_aXAUbP7iHY < 112 + 8; ++INDEX_aXAUbP7iHY) {
				result_PdjdsjaWG5[(INDEX_aXAUbP7iHY)] = intermediate_evYkRRKdLJ[(INDEX_aXAUbP7iHY)] + 1;
			}
		};
		getPool()->at(14).addWork(f_14_aXAUbP7iHY);
		auto f_15_aXAUbP7iHY = [&] () {
			for (size_t INDEX_aXAUbP7iHY = 120; INDEX_aXAUbP7iHY < 120 + 8; ++INDEX_aXAUbP7iHY) {
				result_PdjdsjaWG5[(INDEX_aXAUbP7iHY)] = intermediate_evYkRRKdLJ[(INDEX_aXAUbP7iHY)] + 1;
			}
		};
		getPool()->at(15).addWork(f_15_aXAUbP7iHY);
		auto f_16_aXAUbP7iHY = [&] () {
			for (size_t INDEX_aXAUbP7iHY = 128; INDEX_aXAUbP7iHY < 128 + 8; ++INDEX_aXAUbP7iHY) {
				result_PdjdsjaWG5[(INDEX_aXAUbP7iHY)] = intermediate_evYkRRKdLJ[(INDEX_aXAUbP7iHY)] + 1;
			}
		};
		getPool()->at(16).addWork(f_16_aXAUbP7iHY);
		auto f_17_aXAUbP7iHY = [&] () {
			for (size_t INDEX_aXAUbP7iHY = 136; INDEX_aXAUbP7iHY < 136 + 8; ++INDEX_aXAUbP7iHY) {
				result_PdjdsjaWG5[(INDEX_aXAUbP7iHY)] = intermediate_evYkRRKdLJ[(INDEX_aXAUbP7iHY)] + 1;
			}
		};
		getPool()->at(17).addWork(f_17_aXAUbP7iHY);
		auto f_18_aXAUbP7iHY = [&] () {
			for (size_t INDEX_aXAUbP7iHY = 144; INDEX_aXAUbP7iHY < 144 + 8; ++INDEX_aXAUbP7iHY) {
				result_PdjdsjaWG5[(INDEX_aXAUbP7iHY)] = intermediate_evYkRRKdLJ[(INDEX_aXAUbP7iHY)] + 1;
			}
		};
		getPool()->at(18).addWork(f_18_aXAUbP7iHY);
		auto f_19_aXAUbP7iHY = [&] () {
			for (size_t INDEX_aXAUbP7iHY = 152; INDEX_aXAUbP7iHY < 152 + 8; ++INDEX_aXAUbP7iHY) {
				result_PdjdsjaWG5[(INDEX_aXAUbP7iHY)] = intermediate_evYkRRKdLJ[(INDEX_aXAUbP7iHY)] + 1;
			}
		};
		getPool()->at(19).addWork(f_19_aXAUbP7iHY);
		auto f_20_aXAUbP7iHY = [&] () {
			for (size_t INDEX_aXAUbP7iHY = 160; INDEX_aXAUbP7iHY < 160 + 8; ++INDEX_aXAUbP7iHY) {
				result_PdjdsjaWG5[(INDEX_aXAUbP7iHY)] = intermediate_evYkRRKdLJ[(INDEX_aXAUbP7iHY)] + 1;
			}
		};
		getPool()->at(20).addWork(f_20_aXAUbP7iHY);
		auto f_21_aXAUbP7iHY = [&] () {
			for (size_t INDEX_aXAUbP7iHY = 168; INDEX_aXAUbP7iHY < 168 + 8; ++INDEX_aXAUbP7iHY) {
				result_PdjdsjaWG5[(INDEX_aXAUbP7iHY)] = intermediate_evYkRRKdLJ[(INDEX_aXAUbP7iHY)] + 1;
			}
		};
		getPool()->at(21).addWork(f_21_aXAUbP7iHY);
		auto f_22_aXAUbP7iHY = [&] () {
			for (size_t INDEX_aXAUbP7iHY = 176; INDEX_aXAUbP7iHY < 176 + 8; ++INDEX_aXAUbP7iHY) {
				result_PdjdsjaWG5[(INDEX_aXAUbP7iHY)] = intermediate_evYkRRKdLJ[(INDEX_aXAUbP7iHY)] + 1;
			}
		};
		getPool()->at(22).addWork(f_22_aXAUbP7iHY);
		auto f_23_aXAUbP7iHY = [&] () {
			for (size_t INDEX_aXAUbP7iHY = 184; INDEX_aXAUbP7iHY < 184 + 16; ++INDEX_aXAUbP7iHY) {
				result_PdjdsjaWG5[(INDEX_aXAUbP7iHY)] = intermediate_evYkRRKdLJ[(INDEX_aXAUbP7iHY)] + 1;
			}
		};
		getPool()->at(23).addWork(f_23_aXAUbP7iHY);
	}
	if (rank == 1) {
		Bit_Mask * mask_ptr_rGKnRXDO3F = new Bit_Mask(48,true);
		for (size_t i_5N2ExbesM0 = 0; i_5N2ExbesM0 < 24; ++i_5N2ExbesM0) {
			mask_ptr_rGKnRXDO3F->setBarrier(i_5N2ExbesM0);
		}
		boost::shared_ptr<Bit_Mask>boost_mask_ptr_rGKnRXDO3F (mask_ptr_rGKnRXDO3F);
		self_barrier(boost_mask_ptr_rGKnRXDO3F);
	}
	if (rank == 1) {
		MPI_Send(&result_PdjdsjaWG5[0], 200, MPI_INT, 0, 0, MPI_COMM_WORLD);
	} else if (rank == 0) {
		MPI_Recv(&result_PdjdsjaWG5[0], 200, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	if (rank == 0) {
		for ( int32_t i = 0; i < 200; i++ ) {
			result_seq_t8b2E1gtJz[(i)] = initial_GFPC2vAxPV[(i)] + 2;
			if (result_seq_t8b2E1gtJz[(i)] != result_PdjdsjaWG5[(i)]) {
				print("value at element ", i, " is wrong!");
				MPI_Abort(MPI_COMM_WORLD, 1);
			}
		}
	}
	finishExecution();
	finishGPUExecution();
	std::free(result_PdjdsjaWG5);
	std::free(result_seq_t8b2E1gtJz);
	std::free(initial_GFPC2vAxPV);
	std::free(intermediate_evYkRRKdLJ);
	MPI_Finalize();
	return 0;


}
