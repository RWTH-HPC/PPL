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
#include "includes/ReadNODE2NODE.hxx"
#include "includes/cuda_lib_ReadNODE2NODE.hxx"
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

	int32_t* result_8aefa1e18t;
	int32_t* result_seq_bCIpyXXkO8;
	int32_t* initial_GNlGSH6Hn8;
	int32_t* intermediate_G9pb9SFuN8;
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
	initial_GNlGSH6Hn8 = Init_List(1, initial_GNlGSH6Hn8, 200LL * 1LL);
	intermediate_G9pb9SFuN8 = Init_List(intermediate_G9pb9SFuN8, 200LL * 1LL);
	result_8aefa1e18t = Init_List(result_8aefa1e18t, 200LL * 1LL);
	result_seq_bCIpyXXkO8 = Init_List(result_seq_bCIpyXXkO8, 200LL * 1LL);
	if (rank == 0) {
		auto f_0_UV9Sp4r0ER = [&] () {
			for (size_t INDEX_UV9Sp4r0ER = 0; INDEX_UV9Sp4r0ER < 0 + 8; ++INDEX_UV9Sp4r0ER) {
				intermediate_G9pb9SFuN8[(INDEX_UV9Sp4r0ER)] = initial_GNlGSH6Hn8[(INDEX_UV9Sp4r0ER)] + 1;
			}
		};
		getPool()->at(0).addWork(f_0_UV9Sp4r0ER);
		auto f_1_UV9Sp4r0ER = [&] () {
			for (size_t INDEX_UV9Sp4r0ER = 8; INDEX_UV9Sp4r0ER < 8 + 8; ++INDEX_UV9Sp4r0ER) {
				intermediate_G9pb9SFuN8[(INDEX_UV9Sp4r0ER)] = initial_GNlGSH6Hn8[(INDEX_UV9Sp4r0ER)] + 1;
			}
		};
		getPool()->at(1).addWork(f_1_UV9Sp4r0ER);
		auto f_2_UV9Sp4r0ER = [&] () {
			for (size_t INDEX_UV9Sp4r0ER = 16; INDEX_UV9Sp4r0ER < 16 + 8; ++INDEX_UV9Sp4r0ER) {
				intermediate_G9pb9SFuN8[(INDEX_UV9Sp4r0ER)] = initial_GNlGSH6Hn8[(INDEX_UV9Sp4r0ER)] + 1;
			}
		};
		getPool()->at(2).addWork(f_2_UV9Sp4r0ER);
		auto f_3_UV9Sp4r0ER = [&] () {
			for (size_t INDEX_UV9Sp4r0ER = 24; INDEX_UV9Sp4r0ER < 24 + 8; ++INDEX_UV9Sp4r0ER) {
				intermediate_G9pb9SFuN8[(INDEX_UV9Sp4r0ER)] = initial_GNlGSH6Hn8[(INDEX_UV9Sp4r0ER)] + 1;
			}
		};
		getPool()->at(3).addWork(f_3_UV9Sp4r0ER);
		auto f_4_UV9Sp4r0ER = [&] () {
			for (size_t INDEX_UV9Sp4r0ER = 32; INDEX_UV9Sp4r0ER < 32 + 8; ++INDEX_UV9Sp4r0ER) {
				intermediate_G9pb9SFuN8[(INDEX_UV9Sp4r0ER)] = initial_GNlGSH6Hn8[(INDEX_UV9Sp4r0ER)] + 1;
			}
		};
		getPool()->at(4).addWork(f_4_UV9Sp4r0ER);
		auto f_5_UV9Sp4r0ER = [&] () {
			for (size_t INDEX_UV9Sp4r0ER = 40; INDEX_UV9Sp4r0ER < 40 + 8; ++INDEX_UV9Sp4r0ER) {
				intermediate_G9pb9SFuN8[(INDEX_UV9Sp4r0ER)] = initial_GNlGSH6Hn8[(INDEX_UV9Sp4r0ER)] + 1;
			}
		};
		getPool()->at(5).addWork(f_5_UV9Sp4r0ER);
		auto f_6_UV9Sp4r0ER = [&] () {
			for (size_t INDEX_UV9Sp4r0ER = 48; INDEX_UV9Sp4r0ER < 48 + 8; ++INDEX_UV9Sp4r0ER) {
				intermediate_G9pb9SFuN8[(INDEX_UV9Sp4r0ER)] = initial_GNlGSH6Hn8[(INDEX_UV9Sp4r0ER)] + 1;
			}
		};
		getPool()->at(6).addWork(f_6_UV9Sp4r0ER);
		auto f_7_UV9Sp4r0ER = [&] () {
			for (size_t INDEX_UV9Sp4r0ER = 56; INDEX_UV9Sp4r0ER < 56 + 8; ++INDEX_UV9Sp4r0ER) {
				intermediate_G9pb9SFuN8[(INDEX_UV9Sp4r0ER)] = initial_GNlGSH6Hn8[(INDEX_UV9Sp4r0ER)] + 1;
			}
		};
		getPool()->at(7).addWork(f_7_UV9Sp4r0ER);
		auto f_8_UV9Sp4r0ER = [&] () {
			for (size_t INDEX_UV9Sp4r0ER = 64; INDEX_UV9Sp4r0ER < 64 + 8; ++INDEX_UV9Sp4r0ER) {
				intermediate_G9pb9SFuN8[(INDEX_UV9Sp4r0ER)] = initial_GNlGSH6Hn8[(INDEX_UV9Sp4r0ER)] + 1;
			}
		};
		getPool()->at(8).addWork(f_8_UV9Sp4r0ER);
		auto f_9_UV9Sp4r0ER = [&] () {
			for (size_t INDEX_UV9Sp4r0ER = 72; INDEX_UV9Sp4r0ER < 72 + 8; ++INDEX_UV9Sp4r0ER) {
				intermediate_G9pb9SFuN8[(INDEX_UV9Sp4r0ER)] = initial_GNlGSH6Hn8[(INDEX_UV9Sp4r0ER)] + 1;
			}
		};
		getPool()->at(9).addWork(f_9_UV9Sp4r0ER);
		auto f_10_UV9Sp4r0ER = [&] () {
			for (size_t INDEX_UV9Sp4r0ER = 80; INDEX_UV9Sp4r0ER < 80 + 8; ++INDEX_UV9Sp4r0ER) {
				intermediate_G9pb9SFuN8[(INDEX_UV9Sp4r0ER)] = initial_GNlGSH6Hn8[(INDEX_UV9Sp4r0ER)] + 1;
			}
		};
		getPool()->at(10).addWork(f_10_UV9Sp4r0ER);
		auto f_11_UV9Sp4r0ER = [&] () {
			for (size_t INDEX_UV9Sp4r0ER = 88; INDEX_UV9Sp4r0ER < 88 + 8; ++INDEX_UV9Sp4r0ER) {
				intermediate_G9pb9SFuN8[(INDEX_UV9Sp4r0ER)] = initial_GNlGSH6Hn8[(INDEX_UV9Sp4r0ER)] + 1;
			}
		};
		getPool()->at(11).addWork(f_11_UV9Sp4r0ER);
		auto f_12_UV9Sp4r0ER = [&] () {
			for (size_t INDEX_UV9Sp4r0ER = 96; INDEX_UV9Sp4r0ER < 96 + 8; ++INDEX_UV9Sp4r0ER) {
				intermediate_G9pb9SFuN8[(INDEX_UV9Sp4r0ER)] = initial_GNlGSH6Hn8[(INDEX_UV9Sp4r0ER)] + 1;
			}
		};
		getPool()->at(12).addWork(f_12_UV9Sp4r0ER);
		auto f_13_UV9Sp4r0ER = [&] () {
			for (size_t INDEX_UV9Sp4r0ER = 104; INDEX_UV9Sp4r0ER < 104 + 8; ++INDEX_UV9Sp4r0ER) {
				intermediate_G9pb9SFuN8[(INDEX_UV9Sp4r0ER)] = initial_GNlGSH6Hn8[(INDEX_UV9Sp4r0ER)] + 1;
			}
		};
		getPool()->at(13).addWork(f_13_UV9Sp4r0ER);
		auto f_14_UV9Sp4r0ER = [&] () {
			for (size_t INDEX_UV9Sp4r0ER = 112; INDEX_UV9Sp4r0ER < 112 + 8; ++INDEX_UV9Sp4r0ER) {
				intermediate_G9pb9SFuN8[(INDEX_UV9Sp4r0ER)] = initial_GNlGSH6Hn8[(INDEX_UV9Sp4r0ER)] + 1;
			}
		};
		getPool()->at(14).addWork(f_14_UV9Sp4r0ER);
		auto f_15_UV9Sp4r0ER = [&] () {
			for (size_t INDEX_UV9Sp4r0ER = 120; INDEX_UV9Sp4r0ER < 120 + 8; ++INDEX_UV9Sp4r0ER) {
				intermediate_G9pb9SFuN8[(INDEX_UV9Sp4r0ER)] = initial_GNlGSH6Hn8[(INDEX_UV9Sp4r0ER)] + 1;
			}
		};
		getPool()->at(15).addWork(f_15_UV9Sp4r0ER);
		auto f_16_UV9Sp4r0ER = [&] () {
			for (size_t INDEX_UV9Sp4r0ER = 128; INDEX_UV9Sp4r0ER < 128 + 8; ++INDEX_UV9Sp4r0ER) {
				intermediate_G9pb9SFuN8[(INDEX_UV9Sp4r0ER)] = initial_GNlGSH6Hn8[(INDEX_UV9Sp4r0ER)] + 1;
			}
		};
		getPool()->at(16).addWork(f_16_UV9Sp4r0ER);
		auto f_17_UV9Sp4r0ER = [&] () {
			for (size_t INDEX_UV9Sp4r0ER = 136; INDEX_UV9Sp4r0ER < 136 + 8; ++INDEX_UV9Sp4r0ER) {
				intermediate_G9pb9SFuN8[(INDEX_UV9Sp4r0ER)] = initial_GNlGSH6Hn8[(INDEX_UV9Sp4r0ER)] + 1;
			}
		};
		getPool()->at(17).addWork(f_17_UV9Sp4r0ER);
		auto f_18_UV9Sp4r0ER = [&] () {
			for (size_t INDEX_UV9Sp4r0ER = 144; INDEX_UV9Sp4r0ER < 144 + 8; ++INDEX_UV9Sp4r0ER) {
				intermediate_G9pb9SFuN8[(INDEX_UV9Sp4r0ER)] = initial_GNlGSH6Hn8[(INDEX_UV9Sp4r0ER)] + 1;
			}
		};
		getPool()->at(18).addWork(f_18_UV9Sp4r0ER);
		auto f_19_UV9Sp4r0ER = [&] () {
			for (size_t INDEX_UV9Sp4r0ER = 152; INDEX_UV9Sp4r0ER < 152 + 8; ++INDEX_UV9Sp4r0ER) {
				intermediate_G9pb9SFuN8[(INDEX_UV9Sp4r0ER)] = initial_GNlGSH6Hn8[(INDEX_UV9Sp4r0ER)] + 1;
			}
		};
		getPool()->at(19).addWork(f_19_UV9Sp4r0ER);
		auto f_20_UV9Sp4r0ER = [&] () {
			for (size_t INDEX_UV9Sp4r0ER = 160; INDEX_UV9Sp4r0ER < 160 + 8; ++INDEX_UV9Sp4r0ER) {
				intermediate_G9pb9SFuN8[(INDEX_UV9Sp4r0ER)] = initial_GNlGSH6Hn8[(INDEX_UV9Sp4r0ER)] + 1;
			}
		};
		getPool()->at(20).addWork(f_20_UV9Sp4r0ER);
		auto f_21_UV9Sp4r0ER = [&] () {
			for (size_t INDEX_UV9Sp4r0ER = 168; INDEX_UV9Sp4r0ER < 168 + 8; ++INDEX_UV9Sp4r0ER) {
				intermediate_G9pb9SFuN8[(INDEX_UV9Sp4r0ER)] = initial_GNlGSH6Hn8[(INDEX_UV9Sp4r0ER)] + 1;
			}
		};
		getPool()->at(21).addWork(f_21_UV9Sp4r0ER);
		auto f_22_UV9Sp4r0ER = [&] () {
			for (size_t INDEX_UV9Sp4r0ER = 176; INDEX_UV9Sp4r0ER < 176 + 8; ++INDEX_UV9Sp4r0ER) {
				intermediate_G9pb9SFuN8[(INDEX_UV9Sp4r0ER)] = initial_GNlGSH6Hn8[(INDEX_UV9Sp4r0ER)] + 1;
			}
		};
		getPool()->at(22).addWork(f_22_UV9Sp4r0ER);
		auto f_23_UV9Sp4r0ER = [&] () {
			for (size_t INDEX_UV9Sp4r0ER = 184; INDEX_UV9Sp4r0ER < 184 + 16; ++INDEX_UV9Sp4r0ER) {
				intermediate_G9pb9SFuN8[(INDEX_UV9Sp4r0ER)] = initial_GNlGSH6Hn8[(INDEX_UV9Sp4r0ER)] + 1;
			}
		};
		getPool()->at(23).addWork(f_23_UV9Sp4r0ER);
	}
	if (rank == 0) {
		Bit_Mask * mask_ptr_Vd6ASLzfHS = new Bit_Mask(48,true);
		for (size_t i_n92DUOMigD = 0; i_n92DUOMigD < 24; ++i_n92DUOMigD) {
			mask_ptr_Vd6ASLzfHS->setBarrier(i_n92DUOMigD);
		}
		boost::shared_ptr<Bit_Mask>boost_mask_ptr_Vd6ASLzfHS (mask_ptr_Vd6ASLzfHS);
		self_barrier(boost_mask_ptr_Vd6ASLzfHS);
	}
	if (rank == 0) {
		MPI_Send(&initial_GNlGSH6Hn8[0], 200, MPI_INT, 1, 0, MPI_COMM_WORLD);
	} else if (rank == 1) {
		MPI_Recv(&initial_GNlGSH6Hn8[0], 200, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	if (rank == 1) {
		auto f_0_5a0AtGFei3 = [&] () {
			for (size_t INDEX_5a0AtGFei3 = 0; INDEX_5a0AtGFei3 < 0 + 8; ++INDEX_5a0AtGFei3) {
				result_8aefa1e18t[(INDEX_5a0AtGFei3)] = initial_GNlGSH6Hn8[(INDEX_5a0AtGFei3)] + 1;
			}
		};
		getPool()->at(0).addWork(f_0_5a0AtGFei3);
		auto f_1_5a0AtGFei3 = [&] () {
			for (size_t INDEX_5a0AtGFei3 = 8; INDEX_5a0AtGFei3 < 8 + 8; ++INDEX_5a0AtGFei3) {
				result_8aefa1e18t[(INDEX_5a0AtGFei3)] = initial_GNlGSH6Hn8[(INDEX_5a0AtGFei3)] + 1;
			}
		};
		getPool()->at(1).addWork(f_1_5a0AtGFei3);
		auto f_2_5a0AtGFei3 = [&] () {
			for (size_t INDEX_5a0AtGFei3 = 16; INDEX_5a0AtGFei3 < 16 + 8; ++INDEX_5a0AtGFei3) {
				result_8aefa1e18t[(INDEX_5a0AtGFei3)] = initial_GNlGSH6Hn8[(INDEX_5a0AtGFei3)] + 1;
			}
		};
		getPool()->at(2).addWork(f_2_5a0AtGFei3);
		auto f_3_5a0AtGFei3 = [&] () {
			for (size_t INDEX_5a0AtGFei3 = 24; INDEX_5a0AtGFei3 < 24 + 8; ++INDEX_5a0AtGFei3) {
				result_8aefa1e18t[(INDEX_5a0AtGFei3)] = initial_GNlGSH6Hn8[(INDEX_5a0AtGFei3)] + 1;
			}
		};
		getPool()->at(3).addWork(f_3_5a0AtGFei3);
		auto f_4_5a0AtGFei3 = [&] () {
			for (size_t INDEX_5a0AtGFei3 = 32; INDEX_5a0AtGFei3 < 32 + 8; ++INDEX_5a0AtGFei3) {
				result_8aefa1e18t[(INDEX_5a0AtGFei3)] = initial_GNlGSH6Hn8[(INDEX_5a0AtGFei3)] + 1;
			}
		};
		getPool()->at(4).addWork(f_4_5a0AtGFei3);
		auto f_5_5a0AtGFei3 = [&] () {
			for (size_t INDEX_5a0AtGFei3 = 40; INDEX_5a0AtGFei3 < 40 + 8; ++INDEX_5a0AtGFei3) {
				result_8aefa1e18t[(INDEX_5a0AtGFei3)] = initial_GNlGSH6Hn8[(INDEX_5a0AtGFei3)] + 1;
			}
		};
		getPool()->at(5).addWork(f_5_5a0AtGFei3);
		auto f_6_5a0AtGFei3 = [&] () {
			for (size_t INDEX_5a0AtGFei3 = 48; INDEX_5a0AtGFei3 < 48 + 8; ++INDEX_5a0AtGFei3) {
				result_8aefa1e18t[(INDEX_5a0AtGFei3)] = initial_GNlGSH6Hn8[(INDEX_5a0AtGFei3)] + 1;
			}
		};
		getPool()->at(6).addWork(f_6_5a0AtGFei3);
		auto f_7_5a0AtGFei3 = [&] () {
			for (size_t INDEX_5a0AtGFei3 = 56; INDEX_5a0AtGFei3 < 56 + 8; ++INDEX_5a0AtGFei3) {
				result_8aefa1e18t[(INDEX_5a0AtGFei3)] = initial_GNlGSH6Hn8[(INDEX_5a0AtGFei3)] + 1;
			}
		};
		getPool()->at(7).addWork(f_7_5a0AtGFei3);
		auto f_8_5a0AtGFei3 = [&] () {
			for (size_t INDEX_5a0AtGFei3 = 64; INDEX_5a0AtGFei3 < 64 + 8; ++INDEX_5a0AtGFei3) {
				result_8aefa1e18t[(INDEX_5a0AtGFei3)] = initial_GNlGSH6Hn8[(INDEX_5a0AtGFei3)] + 1;
			}
		};
		getPool()->at(8).addWork(f_8_5a0AtGFei3);
		auto f_9_5a0AtGFei3 = [&] () {
			for (size_t INDEX_5a0AtGFei3 = 72; INDEX_5a0AtGFei3 < 72 + 8; ++INDEX_5a0AtGFei3) {
				result_8aefa1e18t[(INDEX_5a0AtGFei3)] = initial_GNlGSH6Hn8[(INDEX_5a0AtGFei3)] + 1;
			}
		};
		getPool()->at(9).addWork(f_9_5a0AtGFei3);
		auto f_10_5a0AtGFei3 = [&] () {
			for (size_t INDEX_5a0AtGFei3 = 80; INDEX_5a0AtGFei3 < 80 + 8; ++INDEX_5a0AtGFei3) {
				result_8aefa1e18t[(INDEX_5a0AtGFei3)] = initial_GNlGSH6Hn8[(INDEX_5a0AtGFei3)] + 1;
			}
		};
		getPool()->at(10).addWork(f_10_5a0AtGFei3);
		auto f_11_5a0AtGFei3 = [&] () {
			for (size_t INDEX_5a0AtGFei3 = 88; INDEX_5a0AtGFei3 < 88 + 8; ++INDEX_5a0AtGFei3) {
				result_8aefa1e18t[(INDEX_5a0AtGFei3)] = initial_GNlGSH6Hn8[(INDEX_5a0AtGFei3)] + 1;
			}
		};
		getPool()->at(11).addWork(f_11_5a0AtGFei3);
		auto f_12_5a0AtGFei3 = [&] () {
			for (size_t INDEX_5a0AtGFei3 = 96; INDEX_5a0AtGFei3 < 96 + 8; ++INDEX_5a0AtGFei3) {
				result_8aefa1e18t[(INDEX_5a0AtGFei3)] = initial_GNlGSH6Hn8[(INDEX_5a0AtGFei3)] + 1;
			}
		};
		getPool()->at(12).addWork(f_12_5a0AtGFei3);
		auto f_13_5a0AtGFei3 = [&] () {
			for (size_t INDEX_5a0AtGFei3 = 104; INDEX_5a0AtGFei3 < 104 + 8; ++INDEX_5a0AtGFei3) {
				result_8aefa1e18t[(INDEX_5a0AtGFei3)] = initial_GNlGSH6Hn8[(INDEX_5a0AtGFei3)] + 1;
			}
		};
		getPool()->at(13).addWork(f_13_5a0AtGFei3);
		auto f_14_5a0AtGFei3 = [&] () {
			for (size_t INDEX_5a0AtGFei3 = 112; INDEX_5a0AtGFei3 < 112 + 8; ++INDEX_5a0AtGFei3) {
				result_8aefa1e18t[(INDEX_5a0AtGFei3)] = initial_GNlGSH6Hn8[(INDEX_5a0AtGFei3)] + 1;
			}
		};
		getPool()->at(14).addWork(f_14_5a0AtGFei3);
		auto f_15_5a0AtGFei3 = [&] () {
			for (size_t INDEX_5a0AtGFei3 = 120; INDEX_5a0AtGFei3 < 120 + 8; ++INDEX_5a0AtGFei3) {
				result_8aefa1e18t[(INDEX_5a0AtGFei3)] = initial_GNlGSH6Hn8[(INDEX_5a0AtGFei3)] + 1;
			}
		};
		getPool()->at(15).addWork(f_15_5a0AtGFei3);
		auto f_16_5a0AtGFei3 = [&] () {
			for (size_t INDEX_5a0AtGFei3 = 128; INDEX_5a0AtGFei3 < 128 + 8; ++INDEX_5a0AtGFei3) {
				result_8aefa1e18t[(INDEX_5a0AtGFei3)] = initial_GNlGSH6Hn8[(INDEX_5a0AtGFei3)] + 1;
			}
		};
		getPool()->at(16).addWork(f_16_5a0AtGFei3);
		auto f_17_5a0AtGFei3 = [&] () {
			for (size_t INDEX_5a0AtGFei3 = 136; INDEX_5a0AtGFei3 < 136 + 8; ++INDEX_5a0AtGFei3) {
				result_8aefa1e18t[(INDEX_5a0AtGFei3)] = initial_GNlGSH6Hn8[(INDEX_5a0AtGFei3)] + 1;
			}
		};
		getPool()->at(17).addWork(f_17_5a0AtGFei3);
		auto f_18_5a0AtGFei3 = [&] () {
			for (size_t INDEX_5a0AtGFei3 = 144; INDEX_5a0AtGFei3 < 144 + 8; ++INDEX_5a0AtGFei3) {
				result_8aefa1e18t[(INDEX_5a0AtGFei3)] = initial_GNlGSH6Hn8[(INDEX_5a0AtGFei3)] + 1;
			}
		};
		getPool()->at(18).addWork(f_18_5a0AtGFei3);
		auto f_19_5a0AtGFei3 = [&] () {
			for (size_t INDEX_5a0AtGFei3 = 152; INDEX_5a0AtGFei3 < 152 + 8; ++INDEX_5a0AtGFei3) {
				result_8aefa1e18t[(INDEX_5a0AtGFei3)] = initial_GNlGSH6Hn8[(INDEX_5a0AtGFei3)] + 1;
			}
		};
		getPool()->at(19).addWork(f_19_5a0AtGFei3);
		auto f_20_5a0AtGFei3 = [&] () {
			for (size_t INDEX_5a0AtGFei3 = 160; INDEX_5a0AtGFei3 < 160 + 8; ++INDEX_5a0AtGFei3) {
				result_8aefa1e18t[(INDEX_5a0AtGFei3)] = initial_GNlGSH6Hn8[(INDEX_5a0AtGFei3)] + 1;
			}
		};
		getPool()->at(20).addWork(f_20_5a0AtGFei3);
		auto f_21_5a0AtGFei3 = [&] () {
			for (size_t INDEX_5a0AtGFei3 = 168; INDEX_5a0AtGFei3 < 168 + 8; ++INDEX_5a0AtGFei3) {
				result_8aefa1e18t[(INDEX_5a0AtGFei3)] = initial_GNlGSH6Hn8[(INDEX_5a0AtGFei3)] + 1;
			}
		};
		getPool()->at(21).addWork(f_21_5a0AtGFei3);
		auto f_22_5a0AtGFei3 = [&] () {
			for (size_t INDEX_5a0AtGFei3 = 176; INDEX_5a0AtGFei3 < 176 + 8; ++INDEX_5a0AtGFei3) {
				result_8aefa1e18t[(INDEX_5a0AtGFei3)] = initial_GNlGSH6Hn8[(INDEX_5a0AtGFei3)] + 1;
			}
		};
		getPool()->at(22).addWork(f_22_5a0AtGFei3);
		auto f_23_5a0AtGFei3 = [&] () {
			for (size_t INDEX_5a0AtGFei3 = 184; INDEX_5a0AtGFei3 < 184 + 16; ++INDEX_5a0AtGFei3) {
				result_8aefa1e18t[(INDEX_5a0AtGFei3)] = initial_GNlGSH6Hn8[(INDEX_5a0AtGFei3)] + 1;
			}
		};
		getPool()->at(23).addWork(f_23_5a0AtGFei3);
	}
	if (rank == 1) {
		Bit_Mask * mask_ptr_prafHaZHO4 = new Bit_Mask(48,true);
		for (size_t i_svpcJxDq7G = 0; i_svpcJxDq7G < 24; ++i_svpcJxDq7G) {
			mask_ptr_prafHaZHO4->setBarrier(i_svpcJxDq7G);
		}
		boost::shared_ptr<Bit_Mask>boost_mask_ptr_prafHaZHO4 (mask_ptr_prafHaZHO4);
		self_barrier(boost_mask_ptr_prafHaZHO4);
	}
	if (rank == 1) {
		MPI_Send(&result_8aefa1e18t[0], 200, MPI_INT, 0, 0, MPI_COMM_WORLD);
	} else if (rank == 0) {
		MPI_Recv(&result_8aefa1e18t[0], 200, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	if (rank == 0) {
		for ( int32_t i = 0; i < 200; i++ ) {
			result_seq_bCIpyXXkO8[(i)] = initial_GNlGSH6Hn8[(i)] + 1;
			if ((result_seq_bCIpyXXkO8[(i)] != result_8aefa1e18t[(i)]) && (result_8aefa1e18t[(i)] == intermediate_G9pb9SFuN8[(i)])) {
				print("value at element ", i, " is wrong!");
				MPI_Abort(MPI_COMM_WORLD, 1);
			}
		}
	}
	finishExecution();
	finishGPUExecution();
	std::free(result_8aefa1e18t);
	std::free(result_seq_bCIpyXXkO8);
	std::free(initial_GNlGSH6Hn8);
	std::free(intermediate_G9pb9SFuN8);
	MPI_Finalize();
	return 0;


}
