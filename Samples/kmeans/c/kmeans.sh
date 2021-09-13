#!/usr/local_rwth/bin/zsh

### Partition: c18g
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:volta:2
#SBATCH --mem-per-cpu=20G
#SBATCH --job-name=KMEANS
#SBATCH --output=output.kmeans.second.txt
#SBATCH -p=c18g

module switch intel gcc/9

module load cuda/112

#print some debug informations
echo; export; echo;  nvidia-smi; echo

### beginning of executable commands

cd ~/benchmark/c/kmeans
nvcc -Xcompiler -fopenmp -o kmeans.out kmeans.cu
for i in {1..30}
do
   ./kmeans.out
done

