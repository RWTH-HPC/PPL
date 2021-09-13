#!/usr/local_rwth/bin/zsh

#SBATCH --ntasks=48
 
#SBATCH --job-name=monte_carlo_base
#SBATCH --output=output.monte_carlo_base.txt
#SBATCH --time 06:00:00

module unload intel
module load gcc/9

module unload intelmpi
module load openmpi

cd ~/PP/benchmark/c/monte_carlo
mpicc -fopenmp -std=c99 monte_carlo.c -o monte_carlo_base.out

echo $FLAGS_MPI_BATCH
for i in {1..30}
do
   $MPIEXEC $FLAGS_MPI_BATCH ./monte_carlo_base.out
done
