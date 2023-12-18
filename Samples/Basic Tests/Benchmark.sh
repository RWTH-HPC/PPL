#!/bin/bash


#SBATCH -A rwth1270
#SBATCH -c 48
#SBATCH -n 2
#SBATCH --gres=gpu:volta:2
#SBATCH -t 20:00:00
#SBATCH --switches=1

KERNEL="batch Jacobi Monte Multifilter nn"
ITER=40

source /home/as388453/software/.pplcpp_intel
make clean

mkdir -p bin obj cuda

make

for b in $KERNEL; do
	for (( i=0; i<$ITER; i++ )); do
		srun -n 2 bin/$b.exe >> res_$b.txt
	done
done
