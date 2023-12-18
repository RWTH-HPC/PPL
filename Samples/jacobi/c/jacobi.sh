#!/usr/local_rwth/bin/zsh

### Partition: c18m
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=2G
#SBATCH --job-name=Jacobi
#SBATCH --output=output.jacobi.second.txt
#SBATCH --time=04:00:00 
#SBATCH -p=c18m

### beginning of executable commands

export OMP_NUM_THREADS=48

#cd ~/PP/benchmark/c/jacobi
icx -fopenmp -std=c99 -O2 -fdump-rtl-loop2 jacobi.c -o jacobi.out -lm

for i in {1..30}
do
   ./jacobi.out
done
