#!/usr/local_rwth/bin/zsh

### Partition: c18m
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=2G
#SBATCH --job-name=Jacobi_Base
#SBATCH --output=output.jacobi_base.txt
#SBATCH --time=04:00:00 
#SBATCH -p=c18m

### beginning of executable commands

module unload intel
module load gcc/9

export OMP_NUM_THREADS=48

cd ~/PP/benchmark/c/jacobi
$CC -fopenmp -std=c99 -O2 jacobi_base.c -o jacobi_base.out -lm

for i in {1..30}
do
   ./jacobi_base.out
done
