#!/usr/local_rwth/bin/zsh

#SBATCH --ntasks=2
#SBATCH -c 48
 
#SBATCH --job-name=monte_carlo
#SBATCH --output=output.monte_carlo.txt
#SBATCH --time 06:00:00
ml purge
ml gompi

#cd ~/PP/benchmark/c/monte_carlo
mpicc -fopenmp -std=c99 monte_carlo.c -o monte_carlo.out

echo $FLAGS_MPI_BATCH
for i in {1..40}
do
   mpiexec -n 96 ./monte_carlo.out
done
