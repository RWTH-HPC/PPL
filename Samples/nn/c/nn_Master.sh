#!/bin/bash

# Batch setup
Setup="#!/usr/local_rwth/bin/zsh"

Batch = "#SBATCH --ntasks=2 \n
        #SBATCH --cpus-per-task=48 \n
        #SBATCH --mem-per-cpu=2G \n
        #SBATCH --job-name=NN_OpenMP \n
        #SBATCH --output=nn_OpenMP.txt \n
        #SBATCH --time=04:00:00 \n "
        
BatchOMP = "${Batch}#SBATCH -p=c18m\n"
BatchCUDA = "${Batch}#SBATCH -p=c18g\n"

mkdir scripts
mkdir mes

for i in {0..10}; do

    echo "$Setup" > ./scripts/