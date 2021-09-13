#!/usr/local_rwth/bin/zsh

### Partition: c18m
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=2G
#SBATCH --job-name=Classification
#SBATCH --output=output.classification.second.txt
#SBATCH --time=04:00:00 
#SBATCH -p=c18m

### beginning of executable commands

module unload intel
module load gcc/9

export OMP_NUM_THREADS=48

cd ~/PP/benchmark/c/classification
$CC -fopenmp -std=c99 -O2 classification.c -o classification.out

for i in {1..30}
do
   ./classification.out
done

