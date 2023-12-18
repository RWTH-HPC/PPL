#!/usr/local_rwth/bin/zsh

### Partition: c18m
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=2G
#SBATCH --job-name=NN_Base
#SBATCH --output=output.nn_base.txt
#SBATCH --time=04:00:00 
#SBATCH -p=c18m

### beginning of executable commands


export OMP_NUM_THREADS=48

#cd ~/PP/benchmark/c/nn
gcc -fopenmp -std=c99 -O2 neural_network_base.c -o nn_base.out -lm

for i in {1..40}
do
   ./nn_base.out
done

