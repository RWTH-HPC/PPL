#!/usr/local_rwth/bin/zsh

### Partition: c18m
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=2G
#SBATCH --job-name=Multi-Filter-2
#SBATCH --output=output.multi-filter_2.second.txt
#SBATCH --time=04:00:00 
#SBATCH -p=c18m

### beginning of executable commands
module unload intel
module load gcc/9

export OMP_NUM_THREADS=48

cd ~/PP/benchmark/c/multi-filter
gcc -fopenmp -std=c99 multi-filter_2.c -o multi-filter_2.out

for i in {1..30}
do
   OMP_PROC_BIND=close ./multi-filter_2.out
done
