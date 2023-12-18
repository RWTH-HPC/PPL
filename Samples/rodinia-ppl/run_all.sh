#!/bin/bash

ITER=40
BENCH="Back-Propagation Heartwall Hotspot Hotspot3D Kmeans LavaMD nn particle srad" #"Back-Propagation Heartwall Hotspot Hotspot3D Kmeans LavaMD lud myocyte nn srad" # Breadth-First-Search CFD-Solver nw particle pathfinder stream"
GPU="g m"
NODES="1 2"
GPU_FLAG=""

for g in $GPU; do
    if [ $g = "g" ]; then
        GPU_FLAG="--gres=gpu:volta:2"
    else
        GPU_FLAG="--mem=0"
    fi
    for n in $NODES; do
        for b in $BENCH; do
            for i in {1..5}; do
                cd $b
                sbatch -n $n -A rwth1270 -J run_$b -o out_run_$g$n$i.txt -c 48 --mem=0 -t 48:00:00 --exclusive $GPU_FLAG run.sh $g $n $ITER $i
                cd ..
            done
        done
    done
done
