#!/bin/bash

ITER=40
BENCH="LavaMD" #"Back-Propagation Heartwall Hotspot Hotspot3D Kmeans LavaMD nn particle srad" #"Back-Propagation Breadth-First-Search CFD-Solver Heartwall Hotspot Hotspot3D Kmeans LavaMD lud myocyte nn nw particle pathfinder srad stream"
GPU="g m"
NODES="1 2"
GPU_FLAG=""

#rm -r **/out

for g in $GPU; do
    for n in $NODES; do
        for b in $BENCH; do
            for i in {1..5}; do
                cd $b
                sbatch -t 04:00:00 -A rwth1270 --exclusive -J opt_$b -o out_opt_$g$n$i.txt optimize.sh $g $n $ITER $b $i $i
                cd ..
            done
        done
    done
done
