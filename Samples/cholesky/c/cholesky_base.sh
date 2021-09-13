#!/usr/local_rwth/bin/zsh

### Partition: c18m
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=2G
#SBATCH --job-name=Cholseky_Base
#SBATCH --output=output.cholesky_base.second.txt
#SBATCH --time=04:00:00
#SBATCH -p=c18m

### beginning of executable commands

module switch intel gcc/9

export OMP_NUM_THREADS=48

cd ~/PP/benchmark/c/cholesky
$CC -fopenmp -std=c99 -O2 -lm cholesky_base.c -o cholesky_base.out

for i in {1..30}
do
   ./cholesky_base.out
done
