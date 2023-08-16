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

	int32_t* result_DCnv470eVc;
	int32_t* result_seq_8jCNPLFiCZ;
	int32_t* initial_CrNoAq512D;
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
	initial_CrNoAq512D = Init_List(1, initial_CrNoAq512D, 200LL * 1LL);
	result_DCnv470eVc = Init_List(result_DCnv470eVc, 200LL * 1LL);
	result_seq_8jCNPLFiCZ = Init_List(result_seq_8jCNPLFiCZ, 200LL * 1LL);
	if (rank == 0) {
		auto f_0_m7FNq3TZTg = [&] () {
			for (size_t INDEX_m7FNq3TZTg = 0; INDEX_m7FNq3TZTg < 0 + 8; ++INDEX_m7FNq3TZTg) {
				result_DCnv470eVc[(INDEX_m7FNq3TZTg)] = initial_CrNoAq512D[(INDEX_m7FNq3TZTg)] + 1;
			}
		};
		getPool()->at(0).addWork(f_0_m7FNq3TZTg);
		auto f_1_m7FNq3TZTg = [&] () {
			for (size_t INDEX_m7FNq3TZTg = 8; INDEX_m7FNq3TZTg < 8 + 8; ++INDEX_m7FNq3TZTg) {
				result_DCnv470eVc[(INDEX_m7FNq3TZTg)] = initial_CrNoAq512D[(INDEX_m7FNq3TZTg)] + 1;
			}
		};
		getPool()->at(1).addWork(f_1_m7FNq3TZTg);
		auto f_2_m7FNq3TZTg = [&] () {
			for (size_t INDEX_m7FNq3TZTg = 16; INDEX_m7FNq3TZTg < 16 + 8; ++INDEX_m7FNq3TZTg) {
				result_DCnv470eVc[(INDEX_m7FNq3TZTg)] = initial_CrNoAq512D[(INDEX_m7FNq3TZTg)] + 1;
			}
		};
		getPool()->at(2).addWork(f_2_m7FNq3TZTg);
		auto f_3_m7FNq3TZTg = [&] () {
			for (size_t INDEX_m7FNq3TZTg = 24; INDEX_m7FNq3TZTg < 24 + 8; ++INDEX_m7FNq3TZTg) {
				result_DCnv470eVc[(INDEX_m7FNq3TZTg)] = initial_CrNoAq512D[(INDEX_m7FNq3TZTg)] + 1;
			}
		};
		getPool()->at(3).addWork(f_3_m7FNq3TZTg);
		auto f_4_m7FNq3TZTg = [&] () {
			for (size_t INDEX_m7FNq3TZTg = 32; INDEX_m7FNq3TZTg < 32 + 8; ++INDEX_m7FNq3TZTg) {
				result_DCnv470eVc[(INDEX_m7FNq3TZTg)] = initial_CrNoAq512D[(INDEX_m7FNq3TZTg)] + 1;
			}
		};
		getPool()->at(4).addWork(f_4_m7FNq3TZTg);
		auto f_5_m7FNq3TZTg = [&] () {
			for (size_t INDEX_m7FNq3TZTg = 40; INDEX_m7FNq3TZTg < 40 + 8; ++INDEX_m7FNq3TZTg) {
				result_DCnv470eVc[(INDEX_m7FNq3TZTg)] = initial_CrNoAq512D[(INDEX_m7FNq3TZTg)] + 1;
			}
		};
		getPool()->at(5).addWork(f_5_m7FNq3TZTg);
		auto f_6_m7FNq3TZTg = [&] () {
			for (size_t INDEX_m7FNq3TZTg = 48; INDEX_m7FNq3TZTg < 48 + 8; ++INDEX_m7FNq3TZTg) {
				result_DCnv470eVc[(INDEX_m7FNq3TZTg)] = initial_CrNoAq512D[(INDEX_m7FNq3TZTg)] + 1;
			}
		};
		getPool()->at(6).addWork(f_6_m7FNq3TZTg);
		auto f_7_m7FNq3TZTg = [&] () {
			for (size_t INDEX_m7FNq3TZTg = 56; INDEX_m7FNq3TZTg < 56 + 8; ++INDEX_m7FNq3TZTg) {
				result_DCnv470eVc[(INDEX_m7FNq3TZTg)] = initial_CrNoAq512D[(INDEX_m7FNq3TZTg)] + 1;
			}
		};
		getPool()->at(7).addWork(f_7_m7FNq3TZTg);
		auto f_8_m7FNq3TZTg = [&] () {
			for (size_t INDEX_m7FNq3TZTg = 64; INDEX_m7FNq3TZTg < 64 + 8; ++INDEX_m7FNq3TZTg) {
				result_DCnv470eVc[(INDEX_m7FNq3TZTg)] = initial_CrNoAq512D[(INDEX_m7FNq3TZTg)] + 1;
			}
		};
		getPool()->at(8).addWork(f_8_m7FNq3TZTg);
		auto f_9_m7FNq3TZTg = [&] () {
			for (size_t INDEX_m7FNq3TZTg = 72; INDEX_m7FNq3TZTg < 72 + 8; ++INDEX_m7FNq3TZTg) {
				result_DCnv470eVc[(INDEX_m7FNq3TZTg)] = initial_CrNoAq512D[(INDEX_m7FNq3TZTg)] + 1;
			}
		};
		getPool()->at(9).addWork(f_9_m7FNq3TZTg);
		auto f_10_m7FNq3TZTg = [&] () {
			for (size_t INDEX_m7FNq3TZTg = 80; INDEX_m7FNq3TZTg < 80 + 8; ++INDEX_m7FNq3TZTg) {
				result_DCnv470eVc[(INDEX_m7FNq3TZTg)] = initial_CrNoAq512D[(INDEX_m7FNq3TZTg)] + 1;
			}
		};
		getPool()->at(10).addWork(f_10_m7FNq3TZTg);
		auto f_11_m7FNq3TZTg = [&] () {
			for (size_t INDEX_m7FNq3TZTg = 88; INDEX_m7FNq3TZTg < 88 + 8; ++INDEX_m7FNq3TZTg) {
				result_DCnv470eVc[(INDEX_m7FNq3TZTg)] = initial_CrNoAq512D[(INDEX_m7FNq3TZTg)] + 1;
			}
		};
		getPool()->at(11).addWork(f_11_m7FNq3TZTg);
		auto f_12_m7FNq3TZTg = [&] () {
			for (size_t INDEX_m7FNq3TZTg = 96; INDEX_m7FNq3TZTg < 96 + 8; ++INDEX_m7FNq3TZTg) {
				result_DCnv470eVc[(INDEX_m7FNq3TZTg)] = initial_CrNoAq512D[(INDEX_m7FNq3TZTg)] + 1;
			}
		};
		getPool()->at(12).addWork(f_12_m7FNq3TZTg);
		auto f_13_m7FNq3TZTg = [&] () {
			for (size_t INDEX_m7FNq3TZTg = 104; INDEX_m7FNq3TZTg < 104 + 8; ++INDEX_m7FNq3TZTg) {
				result_DCnv470eVc[(INDEX_m7FNq3TZTg)] = initial_CrNoAq512D[(INDEX_m7FNq3TZTg)] + 1;
			}
		};
		getPool()->at(13).addWork(f_13_m7FNq3TZTg);
		auto f_14_m7FNq3TZTg = [&] () {
			for (size_t INDEX_m7FNq3TZTg = 112; INDEX_m7FNq3TZTg < 112 + 8; ++INDEX_m7FNq3TZTg) {
				result_DCnv470eVc[(INDEX_m7FNq3TZTg)] = initial_CrNoAq512D[(INDEX_m7FNq3TZTg)] + 1;
			}
		};
		getPool()->at(14).addWork(f_14_m7FNq3TZTg);
		auto f_15_m7FNq3TZTg = [&] () {
			for (size_t INDEX_m7FNq3TZTg = 120; INDEX_m7FNq3TZTg < 120 + 8; ++INDEX_m7FNq3TZTg) {
				result_DCnv470eVc[(INDEX_m7FNq3TZTg)] = initial_CrNoAq512D[(INDEX_m7FNq3TZTg)] + 1;
			}
		};
		getPool()->at(15).addWork(f_15_m7FNq3TZTg);
		auto f_16_m7FNq3TZTg = [&] () {
			for (size_t INDEX_m7FNq3TZTg = 128; INDEX_m7FNq3TZTg < 128 + 8; ++INDEX_m7FNq3TZTg) {
				result_DCnv470eVc[(INDEX_m7FNq3TZTg)] = initial_CrNoAq512D[(INDEX_m7FNq3TZTg)] + 1;
			}
		};
		getPool()->at(16).addWork(f_16_m7FNq3TZTg);
		auto f_17_m7FNq3TZTg = [&] () {
			for (size_t INDEX_m7FNq3TZTg = 136; INDEX_m7FNq3TZTg < 136 + 8; ++INDEX_m7FNq3TZTg) {
				result_DCnv470eVc[(INDEX_m7FNq3TZTg)] = initial_CrNoAq512D[(INDEX_m7FNq3TZTg)] + 1;
			}
		};
		getPool()->at(17).addWork(f_17_m7FNq3TZTg);
		auto f_18_m7FNq3TZTg = [&] () {
			for (size_t INDEX_m7FNq3TZTg = 144; INDEX_m7FNq3TZTg < 144 + 8; ++INDEX_m7FNq3TZTg) {
				result_DCnv470eVc[(INDEX_m7FNq3TZTg)] = initial_CrNoAq512D[(INDEX_m7FNq3TZTg)] + 1;
			}
		};
		getPool()->at(18).addWork(f_18_m7FNq3TZTg);
		auto f_19_m7FNq3TZTg = [&] () {
			for (size_t INDEX_m7FNq3TZTg = 152; INDEX_m7FNq3TZTg < 152 + 8; ++INDEX_m7FNq3TZTg) {
				result_DCnv470eVc[(INDEX_m7FNq3TZTg)] = initial_CrNoAq512D[(INDEX_m7FNq3TZTg)] + 1;
			}
		};
		getPool()->at(19).addWork(f_19_m7FNq3TZTg);
		auto f_20_m7FNq3TZTg = [&] () {
			for (size_t INDEX_m7FNq3TZTg = 160; INDEX_m7FNq3TZTg < 160 + 8; ++INDEX_m7FNq3TZTg) {
				result_DCnv470eVc[(INDEX_m7FNq3TZTg)] = initial_CrNoAq512D[(INDEX_m7FNq3TZTg)] + 1;
			}
		};
		getPool()->at(20).addWork(f_20_m7FNq3TZTg);
		auto f_21_m7FNq3TZTg = [&] () {
			for (size_t INDEX_m7FNq3TZTg = 168; INDEX_m7FNq3TZTg < 168 + 8; ++INDEX_m7FNq3TZTg) {
				result_DCnv470eVc[(INDEX_m7FNq3TZTg)] = initial_CrNoAq512D[(INDEX_m7FNq3TZTg)] + 1;
			}
		};
		getPool()->at(21).addWork(f_21_m7FNq3TZTg);
		auto f_22_m7FNq3TZTg = [&] () {
			for (size_t INDEX_m7FNq3TZTg = 176; INDEX_m7FNq3TZTg < 176 + 8; ++INDEX_m7FNq3TZTg) {
				result_DCnv470eVc[(INDEX_m7FNq3TZTg)] = initial_CrNoAq512D[(INDEX_m7FNq3TZTg)] + 1;
			}
		};
		getPool()->at(22).addWork(f_22_m7FNq3TZTg);
		auto f_23_m7FNq3TZTg = [&] () {
			for (size_t INDEX_m7FNq3TZTg = 184; INDEX_m7FNq3TZTg < 184 + 16; ++INDEX_m7FNq3TZTg) {
				result_DCnv470eVc[(INDEX_m7FNq3TZTg)] = initial_CrNoAq512D[(INDEX_m7FNq3TZTg)] + 1;
			}
		};
		getPool()->at(23).addWork(f_23_m7FNq3TZTg);
	}
	if (rank == 0) {
		Bit_Mask * mask_ptr_2bxHfzWF56 = new Bit_Mask(48,true);
		for (size_t i_9Ck7NaGZhH = 0; i_9Ck7NaGZhH < 24; ++i_9Ck7NaGZhH) {
			mask_ptr_2bxHfzWF56->setBarrier(i_9Ck7NaGZhH);
		}
		boost::shared_ptr<Bit_Mask>boost_mask_ptr_2bxHfzWF56 (mask_ptr_2bxHfzWF56);
		self_barrier(boost_mask_ptr_2bxHfzWF56);
	}
	if (rank == 0) {
		for ( int32_t i = 0; i < 200; i++ ) {
			result_seq_8jCNPLFiCZ[(i)] = initial_CrNoAq512D[(i)] + 1;
			if (result_seq_8jCNPLFiCZ[(i)] != result_DCnv470eVc[(i)]) {
				print("value at element ", i, " is wrong!");
				MPI_Abort(MPI_COMM_WORLD, 1);
			}
		}
	}
	finishExecution();
	finishGPUExecution();
	std::free(result_DCnv470eVc);
	std::free(result_seq_8jCNPLFiCZ);
	std::free(initial_CrNoAq512D);
	MPI_Finalize();
	return 0;


}
