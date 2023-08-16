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
#include "includes/ReadPartialOverlap.hxx"
#include "includes/cuda_lib_ReadPartialOverlap.hxx"
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

	int32_t* result_ayGBK1CSQa;
	int32_t* result_seq_BtbQegU8RI;
	int32_t* initial_vSvorDb4Jt;
	int32_t* intermediate_TRilhS3XGF;
	int32_t* result_seq2_7V7LXvoRn8;
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
	initial_vSvorDb4Jt = Init_List(1, initial_vSvorDb4Jt, 200LL * 1LL);
	intermediate_TRilhS3XGF = Init_List(intermediate_TRilhS3XGF, 200LL * 1LL);
	result_ayGBK1CSQa = Init_List(result_ayGBK1CSQa, 200LL * 1LL);
	result_seq_BtbQegU8RI = Init_List(result_seq_BtbQegU8RI, 200LL * 1LL);
	result_seq2_7V7LXvoRn8 = Init_List(result_seq2_7V7LXvoRn8, 200LL * 1LL);
	if (rank == 0) {
		auto f_0_fHmnKuuPla = [&] () {
			for (size_t INDEX_fHmnKuuPla = 0; INDEX_fHmnKuuPla < 0 + 4; ++INDEX_fHmnKuuPla) {
				intermediate_TRilhS3XGF[(2 * INDEX_fHmnKuuPla)] = initial_vSvorDb4Jt[(2 * INDEX_fHmnKuuPla)] + 1;
			}
		};
		getPool()->at(0).addWork(f_0_fHmnKuuPla);
		auto f_1_fHmnKuuPla = [&] () {
			for (size_t INDEX_fHmnKuuPla = 4; INDEX_fHmnKuuPla < 4 + 4; ++INDEX_fHmnKuuPla) {
				intermediate_TRilhS3XGF[(2 * INDEX_fHmnKuuPla)] = initial_vSvorDb4Jt[(2 * INDEX_fHmnKuuPla)] + 1;
			}
		};
		getPool()->at(1).addWork(f_1_fHmnKuuPla);
		auto f_2_fHmnKuuPla = [&] () {
			for (size_t INDEX_fHmnKuuPla = 8; INDEX_fHmnKuuPla < 8 + 4; ++INDEX_fHmnKuuPla) {
				intermediate_TRilhS3XGF[(2 * INDEX_fHmnKuuPla)] = initial_vSvorDb4Jt[(2 * INDEX_fHmnKuuPla)] + 1;
			}
		};
		getPool()->at(2).addWork(f_2_fHmnKuuPla);
		auto f_3_fHmnKuuPla = [&] () {
			for (size_t INDEX_fHmnKuuPla = 12; INDEX_fHmnKuuPla < 12 + 4; ++INDEX_fHmnKuuPla) {
				intermediate_TRilhS3XGF[(2 * INDEX_fHmnKuuPla)] = initial_vSvorDb4Jt[(2 * INDEX_fHmnKuuPla)] + 1;
			}
		};
		getPool()->at(3).addWork(f_3_fHmnKuuPla);
		auto f_4_fHmnKuuPla = [&] () {
			for (size_t INDEX_fHmnKuuPla = 16; INDEX_fHmnKuuPla < 16 + 4; ++INDEX_fHmnKuuPla) {
				intermediate_TRilhS3XGF[(2 * INDEX_fHmnKuuPla)] = initial_vSvorDb4Jt[(2 * INDEX_fHmnKuuPla)] + 1;
			}
		};
		getPool()->at(4).addWork(f_4_fHmnKuuPla);
		auto f_5_fHmnKuuPla = [&] () {
			for (size_t INDEX_fHmnKuuPla = 20; INDEX_fHmnKuuPla < 20 + 4; ++INDEX_fHmnKuuPla) {
				intermediate_TRilhS3XGF[(2 * INDEX_fHmnKuuPla)] = initial_vSvorDb4Jt[(2 * INDEX_fHmnKuuPla)] + 1;
			}
		};
		getPool()->at(5).addWork(f_5_fHmnKuuPla);
		auto f_6_fHmnKuuPla = [&] () {
			for (size_t INDEX_fHmnKuuPla = 24; INDEX_fHmnKuuPla < 24 + 4; ++INDEX_fHmnKuuPla) {
				intermediate_TRilhS3XGF[(2 * INDEX_fHmnKuuPla)] = initial_vSvorDb4Jt[(2 * INDEX_fHmnKuuPla)] + 1;
			}
		};
		getPool()->at(6).addWork(f_6_fHmnKuuPla);
		auto f_7_fHmnKuuPla = [&] () {
			for (size_t INDEX_fHmnKuuPla = 28; INDEX_fHmnKuuPla < 28 + 4; ++INDEX_fHmnKuuPla) {
				intermediate_TRilhS3XGF[(2 * INDEX_fHmnKuuPla)] = initial_vSvorDb4Jt[(2 * INDEX_fHmnKuuPla)] + 1;
			}
		};
		getPool()->at(7).addWork(f_7_fHmnKuuPla);
		auto f_8_fHmnKuuPla = [&] () {
			for (size_t INDEX_fHmnKuuPla = 32; INDEX_fHmnKuuPla < 32 + 4; ++INDEX_fHmnKuuPla) {
				intermediate_TRilhS3XGF[(2 * INDEX_fHmnKuuPla)] = initial_vSvorDb4Jt[(2 * INDEX_fHmnKuuPla)] + 1;
			}
		};
		getPool()->at(8).addWork(f_8_fHmnKuuPla);
		auto f_9_fHmnKuuPla = [&] () {
			for (size_t INDEX_fHmnKuuPla = 36; INDEX_fHmnKuuPla < 36 + 4; ++INDEX_fHmnKuuPla) {
				intermediate_TRilhS3XGF[(2 * INDEX_fHmnKuuPla)] = initial_vSvorDb4Jt[(2 * INDEX_fHmnKuuPla)] + 1;
			}
		};
		getPool()->at(9).addWork(f_9_fHmnKuuPla);
		auto f_10_fHmnKuuPla = [&] () {
			for (size_t INDEX_fHmnKuuPla = 40; INDEX_fHmnKuuPla < 40 + 4; ++INDEX_fHmnKuuPla) {
				intermediate_TRilhS3XGF[(2 * INDEX_fHmnKuuPla)] = initial_vSvorDb4Jt[(2 * INDEX_fHmnKuuPla)] + 1;
			}
		};
		getPool()->at(10).addWork(f_10_fHmnKuuPla);
		auto f_11_fHmnKuuPla = [&] () {
			for (size_t INDEX_fHmnKuuPla = 44; INDEX_fHmnKuuPla < 44 + 4; ++INDEX_fHmnKuuPla) {
				intermediate_TRilhS3XGF[(2 * INDEX_fHmnKuuPla)] = initial_vSvorDb4Jt[(2 * INDEX_fHmnKuuPla)] + 1;
			}
		};
		getPool()->at(11).addWork(f_11_fHmnKuuPla);
		auto f_12_fHmnKuuPla = [&] () {
			for (size_t INDEX_fHmnKuuPla = 48; INDEX_fHmnKuuPla < 48 + 4; ++INDEX_fHmnKuuPla) {
				intermediate_TRilhS3XGF[(2 * INDEX_fHmnKuuPla)] = initial_vSvorDb4Jt[(2 * INDEX_fHmnKuuPla)] + 1;
			}
		};
		getPool()->at(12).addWork(f_12_fHmnKuuPla);
		auto f_13_fHmnKuuPla = [&] () {
			for (size_t INDEX_fHmnKuuPla = 52; INDEX_fHmnKuuPla < 52 + 4; ++INDEX_fHmnKuuPla) {
				intermediate_TRilhS3XGF[(2 * INDEX_fHmnKuuPla)] = initial_vSvorDb4Jt[(2 * INDEX_fHmnKuuPla)] + 1;
			}
		};
		getPool()->at(13).addWork(f_13_fHmnKuuPla);
		auto f_14_fHmnKuuPla = [&] () {
			for (size_t INDEX_fHmnKuuPla = 56; INDEX_fHmnKuuPla < 56 + 4; ++INDEX_fHmnKuuPla) {
				intermediate_TRilhS3XGF[(2 * INDEX_fHmnKuuPla)] = initial_vSvorDb4Jt[(2 * INDEX_fHmnKuuPla)] + 1;
			}
		};
		getPool()->at(14).addWork(f_14_fHmnKuuPla);
		auto f_15_fHmnKuuPla = [&] () {
			for (size_t INDEX_fHmnKuuPla = 60; INDEX_fHmnKuuPla < 60 + 4; ++INDEX_fHmnKuuPla) {
				intermediate_TRilhS3XGF[(2 * INDEX_fHmnKuuPla)] = initial_vSvorDb4Jt[(2 * INDEX_fHmnKuuPla)] + 1;
			}
		};
		getPool()->at(15).addWork(f_15_fHmnKuuPla);
		auto f_16_fHmnKuuPla = [&] () {
			for (size_t INDEX_fHmnKuuPla = 64; INDEX_fHmnKuuPla < 64 + 4; ++INDEX_fHmnKuuPla) {
				intermediate_TRilhS3XGF[(2 * INDEX_fHmnKuuPla)] = initial_vSvorDb4Jt[(2 * INDEX_fHmnKuuPla)] + 1;
			}
		};
		getPool()->at(16).addWork(f_16_fHmnKuuPla);
		auto f_17_fHmnKuuPla = [&] () {
			for (size_t INDEX_fHmnKuuPla = 68; INDEX_fHmnKuuPla < 68 + 4; ++INDEX_fHmnKuuPla) {
				intermediate_TRilhS3XGF[(2 * INDEX_fHmnKuuPla)] = initial_vSvorDb4Jt[(2 * INDEX_fHmnKuuPla)] + 1;
			}
		};
		getPool()->at(17).addWork(f_17_fHmnKuuPla);
		auto f_18_fHmnKuuPla = [&] () {
			for (size_t INDEX_fHmnKuuPla = 72; INDEX_fHmnKuuPla < 72 + 4; ++INDEX_fHmnKuuPla) {
				intermediate_TRilhS3XGF[(2 * INDEX_fHmnKuuPla)] = initial_vSvorDb4Jt[(2 * INDEX_fHmnKuuPla)] + 1;
			}
		};
		getPool()->at(18).addWork(f_18_fHmnKuuPla);
		auto f_19_fHmnKuuPla = [&] () {
			for (size_t INDEX_fHmnKuuPla = 76; INDEX_fHmnKuuPla < 76 + 4; ++INDEX_fHmnKuuPla) {
				intermediate_TRilhS3XGF[(2 * INDEX_fHmnKuuPla)] = initial_vSvorDb4Jt[(2 * INDEX_fHmnKuuPla)] + 1;
			}
		};
		getPool()->at(19).addWork(f_19_fHmnKuuPla);
		auto f_20_fHmnKuuPla = [&] () {
			for (size_t INDEX_fHmnKuuPla = 80; INDEX_fHmnKuuPla < 80 + 4; ++INDEX_fHmnKuuPla) {
				intermediate_TRilhS3XGF[(2 * INDEX_fHmnKuuPla)] = initial_vSvorDb4Jt[(2 * INDEX_fHmnKuuPla)] + 1;
			}
		};
		getPool()->at(20).addWork(f_20_fHmnKuuPla);
		auto f_21_fHmnKuuPla = [&] () {
			for (size_t INDEX_fHmnKuuPla = 84; INDEX_fHmnKuuPla < 84 + 4; ++INDEX_fHmnKuuPla) {
				intermediate_TRilhS3XGF[(2 * INDEX_fHmnKuuPla)] = initial_vSvorDb4Jt[(2 * INDEX_fHmnKuuPla)] + 1;
			}
		};
		getPool()->at(21).addWork(f_21_fHmnKuuPla);
		auto f_22_fHmnKuuPla = [&] () {
			for (size_t INDEX_fHmnKuuPla = 88; INDEX_fHmnKuuPla < 88 + 4; ++INDEX_fHmnKuuPla) {
				intermediate_TRilhS3XGF[(2 * INDEX_fHmnKuuPla)] = initial_vSvorDb4Jt[(2 * INDEX_fHmnKuuPla)] + 1;
			}
		};
		getPool()->at(22).addWork(f_22_fHmnKuuPla);
		auto f_23_fHmnKuuPla = [&] () {
			for (size_t INDEX_fHmnKuuPla = 92; INDEX_fHmnKuuPla < 92 + 8; ++INDEX_fHmnKuuPla) {
				intermediate_TRilhS3XGF[(2 * INDEX_fHmnKuuPla)] = initial_vSvorDb4Jt[(2 * INDEX_fHmnKuuPla)] + 1;
			}
		};
		getPool()->at(23).addWork(f_23_fHmnKuuPla);
	}
	if (rank == 0) {
		Bit_Mask * mask_ptr_CGhxTC5E4y = new Bit_Mask(48,true);
		for (size_t i_ssNz7QnO9T = 0; i_ssNz7QnO9T < 24; ++i_ssNz7QnO9T) {
			mask_ptr_CGhxTC5E4y->setBarrier(i_ssNz7QnO9T);
		}
		boost::shared_ptr<Bit_Mask>boost_mask_ptr_CGhxTC5E4y (mask_ptr_CGhxTC5E4y);
		self_barrier(boost_mask_ptr_CGhxTC5E4y);
	}
	if (rank == 0) {
		auto f_0_hIxfC44PB0 = [&] () {
			for (size_t INDEX_hIxfC44PB0 = 0; INDEX_hIxfC44PB0 < 0 + 2; ++INDEX_hIxfC44PB0) {
				result_ayGBK1CSQa[(3 * INDEX_hIxfC44PB0)] = initial_vSvorDb4Jt[(3 * INDEX_hIxfC44PB0)] + 1;
			}
		};
		getPool()->at(0).addWork(f_0_hIxfC44PB0);
		auto f_1_hIxfC44PB0 = [&] () {
			for (size_t INDEX_hIxfC44PB0 = 2; INDEX_hIxfC44PB0 < 2 + 2; ++INDEX_hIxfC44PB0) {
				result_ayGBK1CSQa[(3 * INDEX_hIxfC44PB0)] = initial_vSvorDb4Jt[(3 * INDEX_hIxfC44PB0)] + 1;
			}
		};
		getPool()->at(1).addWork(f_1_hIxfC44PB0);
		auto f_2_hIxfC44PB0 = [&] () {
			for (size_t INDEX_hIxfC44PB0 = 4; INDEX_hIxfC44PB0 < 4 + 2; ++INDEX_hIxfC44PB0) {
				result_ayGBK1CSQa[(3 * INDEX_hIxfC44PB0)] = initial_vSvorDb4Jt[(3 * INDEX_hIxfC44PB0)] + 1;
			}
		};
		getPool()->at(2).addWork(f_2_hIxfC44PB0);
		auto f_3_hIxfC44PB0 = [&] () {
			for (size_t INDEX_hIxfC44PB0 = 6; INDEX_hIxfC44PB0 < 6 + 2; ++INDEX_hIxfC44PB0) {
				result_ayGBK1CSQa[(3 * INDEX_hIxfC44PB0)] = initial_vSvorDb4Jt[(3 * INDEX_hIxfC44PB0)] + 1;
			}
		};
		getPool()->at(3).addWork(f_3_hIxfC44PB0);
		auto f_4_hIxfC44PB0 = [&] () {
			for (size_t INDEX_hIxfC44PB0 = 8; INDEX_hIxfC44PB0 < 8 + 2; ++INDEX_hIxfC44PB0) {
				result_ayGBK1CSQa[(3 * INDEX_hIxfC44PB0)] = initial_vSvorDb4Jt[(3 * INDEX_hIxfC44PB0)] + 1;
			}
		};
		getPool()->at(4).addWork(f_4_hIxfC44PB0);
		auto f_5_hIxfC44PB0 = [&] () {
			for (size_t INDEX_hIxfC44PB0 = 10; INDEX_hIxfC44PB0 < 10 + 2; ++INDEX_hIxfC44PB0) {
				result_ayGBK1CSQa[(3 * INDEX_hIxfC44PB0)] = initial_vSvorDb4Jt[(3 * INDEX_hIxfC44PB0)] + 1;
			}
		};
		getPool()->at(5).addWork(f_5_hIxfC44PB0);
		auto f_6_hIxfC44PB0 = [&] () {
			for (size_t INDEX_hIxfC44PB0 = 12; INDEX_hIxfC44PB0 < 12 + 2; ++INDEX_hIxfC44PB0) {
				result_ayGBK1CSQa[(3 * INDEX_hIxfC44PB0)] = initial_vSvorDb4Jt[(3 * INDEX_hIxfC44PB0)] + 1;
			}
		};
		getPool()->at(6).addWork(f_6_hIxfC44PB0);
		auto f_7_hIxfC44PB0 = [&] () {
			for (size_t INDEX_hIxfC44PB0 = 14; INDEX_hIxfC44PB0 < 14 + 2; ++INDEX_hIxfC44PB0) {
				result_ayGBK1CSQa[(3 * INDEX_hIxfC44PB0)] = initial_vSvorDb4Jt[(3 * INDEX_hIxfC44PB0)] + 1;
			}
		};
		getPool()->at(7).addWork(f_7_hIxfC44PB0);
		auto f_8_hIxfC44PB0 = [&] () {
			for (size_t INDEX_hIxfC44PB0 = 16; INDEX_hIxfC44PB0 < 16 + 2; ++INDEX_hIxfC44PB0) {
				result_ayGBK1CSQa[(3 * INDEX_hIxfC44PB0)] = initial_vSvorDb4Jt[(3 * INDEX_hIxfC44PB0)] + 1;
			}
		};
		getPool()->at(8).addWork(f_8_hIxfC44PB0);
		auto f_9_hIxfC44PB0 = [&] () {
			for (size_t INDEX_hIxfC44PB0 = 18; INDEX_hIxfC44PB0 < 18 + 2; ++INDEX_hIxfC44PB0) {
				result_ayGBK1CSQa[(3 * INDEX_hIxfC44PB0)] = initial_vSvorDb4Jt[(3 * INDEX_hIxfC44PB0)] + 1;
			}
		};
		getPool()->at(9).addWork(f_9_hIxfC44PB0);
		auto f_10_hIxfC44PB0 = [&] () {
			for (size_t INDEX_hIxfC44PB0 = 20; INDEX_hIxfC44PB0 < 20 + 2; ++INDEX_hIxfC44PB0) {
				result_ayGBK1CSQa[(3 * INDEX_hIxfC44PB0)] = initial_vSvorDb4Jt[(3 * INDEX_hIxfC44PB0)] + 1;
			}
		};
		getPool()->at(10).addWork(f_10_hIxfC44PB0);
		auto f_11_hIxfC44PB0 = [&] () {
			for (size_t INDEX_hIxfC44PB0 = 22; INDEX_hIxfC44PB0 < 22 + 2; ++INDEX_hIxfC44PB0) {
				result_ayGBK1CSQa[(3 * INDEX_hIxfC44PB0)] = initial_vSvorDb4Jt[(3 * INDEX_hIxfC44PB0)] + 1;
			}
		};
		getPool()->at(11).addWork(f_11_hIxfC44PB0);
		auto f_12_hIxfC44PB0 = [&] () {
			for (size_t INDEX_hIxfC44PB0 = 24; INDEX_hIxfC44PB0 < 24 + 2; ++INDEX_hIxfC44PB0) {
				result_ayGBK1CSQa[(3 * INDEX_hIxfC44PB0)] = initial_vSvorDb4Jt[(3 * INDEX_hIxfC44PB0)] + 1;
			}
		};
		getPool()->at(12).addWork(f_12_hIxfC44PB0);
		auto f_13_hIxfC44PB0 = [&] () {
			for (size_t INDEX_hIxfC44PB0 = 26; INDEX_hIxfC44PB0 < 26 + 2; ++INDEX_hIxfC44PB0) {
				result_ayGBK1CSQa[(3 * INDEX_hIxfC44PB0)] = initial_vSvorDb4Jt[(3 * INDEX_hIxfC44PB0)] + 1;
			}
		};
		getPool()->at(13).addWork(f_13_hIxfC44PB0);
		auto f_14_hIxfC44PB0 = [&] () {
			for (size_t INDEX_hIxfC44PB0 = 28; INDEX_hIxfC44PB0 < 28 + 2; ++INDEX_hIxfC44PB0) {
				result_ayGBK1CSQa[(3 * INDEX_hIxfC44PB0)] = initial_vSvorDb4Jt[(3 * INDEX_hIxfC44PB0)] + 1;
			}
		};
		getPool()->at(14).addWork(f_14_hIxfC44PB0);
		auto f_15_hIxfC44PB0 = [&] () {
			for (size_t INDEX_hIxfC44PB0 = 30; INDEX_hIxfC44PB0 < 30 + 2; ++INDEX_hIxfC44PB0) {
				result_ayGBK1CSQa[(3 * INDEX_hIxfC44PB0)] = initial_vSvorDb4Jt[(3 * INDEX_hIxfC44PB0)] + 1;
			}
		};
		getPool()->at(15).addWork(f_15_hIxfC44PB0);
		auto f_16_hIxfC44PB0 = [&] () {
			for (size_t INDEX_hIxfC44PB0 = 32; INDEX_hIxfC44PB0 < 32 + 2; ++INDEX_hIxfC44PB0) {
				result_ayGBK1CSQa[(3 * INDEX_hIxfC44PB0)] = initial_vSvorDb4Jt[(3 * INDEX_hIxfC44PB0)] + 1;
			}
		};
		getPool()->at(16).addWork(f_16_hIxfC44PB0);
		auto f_17_hIxfC44PB0 = [&] () {
			for (size_t INDEX_hIxfC44PB0 = 34; INDEX_hIxfC44PB0 < 34 + 2; ++INDEX_hIxfC44PB0) {
				result_ayGBK1CSQa[(3 * INDEX_hIxfC44PB0)] = initial_vSvorDb4Jt[(3 * INDEX_hIxfC44PB0)] + 1;
			}
		};
		getPool()->at(17).addWork(f_17_hIxfC44PB0);
		auto f_18_hIxfC44PB0 = [&] () {
			for (size_t INDEX_hIxfC44PB0 = 36; INDEX_hIxfC44PB0 < 36 + 2; ++INDEX_hIxfC44PB0) {
				result_ayGBK1CSQa[(3 * INDEX_hIxfC44PB0)] = initial_vSvorDb4Jt[(3 * INDEX_hIxfC44PB0)] + 1;
			}
		};
		getPool()->at(18).addWork(f_18_hIxfC44PB0);
		auto f_19_hIxfC44PB0 = [&] () {
			for (size_t INDEX_hIxfC44PB0 = 38; INDEX_hIxfC44PB0 < 38 + 2; ++INDEX_hIxfC44PB0) {
				result_ayGBK1CSQa[(3 * INDEX_hIxfC44PB0)] = initial_vSvorDb4Jt[(3 * INDEX_hIxfC44PB0)] + 1;
			}
		};
		getPool()->at(19).addWork(f_19_hIxfC44PB0);
		auto f_20_hIxfC44PB0 = [&] () {
			for (size_t INDEX_hIxfC44PB0 = 40; INDEX_hIxfC44PB0 < 40 + 2; ++INDEX_hIxfC44PB0) {
				result_ayGBK1CSQa[(3 * INDEX_hIxfC44PB0)] = initial_vSvorDb4Jt[(3 * INDEX_hIxfC44PB0)] + 1;
			}
		};
		getPool()->at(20).addWork(f_20_hIxfC44PB0);
		auto f_21_hIxfC44PB0 = [&] () {
			for (size_t INDEX_hIxfC44PB0 = 42; INDEX_hIxfC44PB0 < 42 + 2; ++INDEX_hIxfC44PB0) {
				result_ayGBK1CSQa[(3 * INDEX_hIxfC44PB0)] = initial_vSvorDb4Jt[(3 * INDEX_hIxfC44PB0)] + 1;
			}
		};
		getPool()->at(21).addWork(f_21_hIxfC44PB0);
		auto f_22_hIxfC44PB0 = [&] () {
			for (size_t INDEX_hIxfC44PB0 = 44; INDEX_hIxfC44PB0 < 44 + 2; ++INDEX_hIxfC44PB0) {
				result_ayGBK1CSQa[(3 * INDEX_hIxfC44PB0)] = initial_vSvorDb4Jt[(3 * INDEX_hIxfC44PB0)] + 1;
			}
		};
		getPool()->at(22).addWork(f_22_hIxfC44PB0);
		auto f_23_hIxfC44PB0 = [&] () {
			for (size_t INDEX_hIxfC44PB0 = 46; INDEX_hIxfC44PB0 < 46 + 21; ++INDEX_hIxfC44PB0) {
				result_ayGBK1CSQa[(3 * INDEX_hIxfC44PB0)] = initial_vSvorDb4Jt[(3 * INDEX_hIxfC44PB0)] + 1;
			}
		};
		getPool()->at(23).addWork(f_23_hIxfC44PB0);
	}
	if (rank == 0) {
		Bit_Mask * mask_ptr_a843NZI6KS = new Bit_Mask(48,true);
		for (size_t i_ox6a0Nza49 = 0; i_ox6a0Nza49 < 24; ++i_ox6a0Nza49) {
			mask_ptr_a843NZI6KS->setBarrier(i_ox6a0Nza49);
		}
		boost::shared_ptr<Bit_Mask>boost_mask_ptr_a843NZI6KS (mask_ptr_a843NZI6KS);
		self_barrier(boost_mask_ptr_a843NZI6KS);
	}
	if (rank == 0) {
		for ( int32_t i = 0; i < 200; i++ ) {
			if (i % 2 == 0) {
				result_seq_BtbQegU8RI[(i)] = initial_vSvorDb4Jt[(i)] + 1;
			}
			if (i % 3 == 0) {
				result_seq2_7V7LXvoRn8[(i)] = initial_vSvorDb4Jt[(i)] + 1;
			}
			if (result_seq_BtbQegU8RI[(i)] != intermediate_TRilhS3XGF[(i)]) {
				print("value at element ", i, " is wrong!");
				MPI_Abort(MPI_COMM_WORLD, 1);
			}
			if (result_seq2_7V7LXvoRn8[(i)] != result_ayGBK1CSQa[(i)]) {
				print("value at element ", i, " is wrong!");
				MPI_Abort(MPI_COMM_WORLD, 1);
			}
		}
	}
	finishExecution();
	finishGPUExecution();
	std::free(result_ayGBK1CSQa);
	std::free(result_seq_BtbQegU8RI);
	std::free(initial_vSvorDb4Jt);
	std::free(intermediate_TRilhS3XGF);
	std::free(result_seq2_7V7LXvoRn8);
	MPI_Finalize();
	return 0;


}
