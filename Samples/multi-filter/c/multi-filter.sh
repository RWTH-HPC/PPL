#!/usr/local_rwth/bin/zsh

### Partition: c18m
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=2G
#SBATCH --job-name=Multi-Filter
#SBATCH --output=output.multi-filter.second.txt
#SBATCH --time=04:00:00 
#SBATCH -p=c18m

### beginning of executable commands

export OMP_NUM_THREADS=48

#cd ~/PP/benchmark/c/multi-filter
icx -fopenmp -std=c99 multi-filter.c -o multi-filter.out

for i in {1..40}
do
   OMP_PROC_BIND=spread ./multi-filter.out
done
