#!/usr/local_rwth/bin/zsh

### Partition: c18g
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:volta:2
#SBATCH --mem-per-cpu=20G
#SBATCH --job-name=NN
#SBATCH --output=output.nn.second.txt
#SBATCH -p=c18g
 
ml CUDA
 
#print some debug informations
echo; export; echo;  nvidia-smi; echo

### beginning of executable commands

#cd ~/benchmark/c/nn
nvcc -Xcompiler -fopenmp -o nn.out neural_network.cu -lcurand
for i in {1..40}
do
   ./nn.out
done

