#!/usr/local_rwth/bin/zsh

### Partition: c18m
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=2G
#SBATCH --job-name=Multi-Filter_Base
#SBATCH --output=output.multi-filter_base.second.txt
#SBATCH --time=04:00:00 
#SBATCH -p=c18m

### beginning of executable commands


export OMP_NUM_THREADS=48

#cd ~/PP/benchmark/c/multi-filter
icx -fopenmp -std=c99 multi-filter_base.c -o multi-filter_base.out

for i in {1..30}
do
   OMP_PROC_BIND=close ./multi-filter_base.out
done
