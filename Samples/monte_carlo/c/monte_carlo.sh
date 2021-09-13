#!/usr/local_rwth/bin/zsh

#SBATCH --ntasks=96
 
#SBATCH --job-name=monte_carlo
#SBATCH --output=output.monte_carlo.txt
#SBATCH --time 06:00:00

module unload intel
module load gcc/9

module unload intelmpi
module load openmpi

cd ~/PP/benchmark/c/monte_carlo
mpicc -fopenmp -std=c99 monte_carlo.c -o monte_carlo.out

echo $FLAGS_MPI_BATCH
for i in {1..30}
do
   $MPIEXEC $FLAGS_MPI_BATCH ./monte_carlo.out
done
