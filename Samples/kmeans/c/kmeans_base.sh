#!/usr/local_rwth/bin/zsh

### Partition: c18m
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=2G
#SBATCH --job-name=KMEANS_Base
#SBATCH --output=output.kmeans_base.second.txt
#SBATCH --time=04:00:00
#SBATCH -p=c18m

### beginning of executable commands

module switch intel gcc/9

export OMP_NUM_THREADS=48

cd ~/PP/benchmark/c/kmeans
$CC -fopenmp -std=c99 -O2 -lm kmeans_base.c -o kmeans_base.out

for i in {1..30}
do
   ./kmeans_base.out
done
