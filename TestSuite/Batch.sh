#!/usr/local_rwth/bin/zsh

### Partition: c18g
#SBATCH --cpus-per-task=48
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:volta:2
#SBATCH --mem-per-cpu=100G
#SBATCH --job-name=Testing
#SBATCH --output=output.txt
#SBATCH -p=c18g
#SBATCH --ntasks=2

module unload intel
module unload intelmpi
module load cuda/110
module load gcc/9
module load openmpi/4.0.3

#print some debug informations...
echo; export; echo;  nvidia-smi; echo

cd ~/Master/out
make runall
$MPIEXEC $FLAGS_MPI_BATCH ./bin/Pattern_Test.exe
