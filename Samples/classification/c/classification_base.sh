#!/usr/local_rwth/bin/zsh

### Partition: c18m
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=2G
#SBATCH --job-name=Classification_Base
#SBATCH --output=output.classification_base.second.txt
#SBATCH --time=04:00:00 
#SBATCH -p=c18m

### beginning of executable commands

export OMP_NUM_THREADS=48

#cd ~/PP/benchmark/c/classification
icx -fopenmp -std=c99 -O2 classification_base.c -o classification_base.out

for i in {1..40}
do
   ./classification_base.out
done
