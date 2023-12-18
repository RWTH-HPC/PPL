#!/usr/bin/env zsh

 module load cuda/11.5
 module switch intel gcc/10
 module switch intelmpi openmpi/4.0.3
 
 make -C Back-Propagation/out
 make -C Breadth-First-Search/out
 make -C CFD-Solver/out
 make -C Heartwall/out
 make -C Hotspot/out
 make -C Hotspot3D/out
 make -C Kmeans/out
 make -C LavaMD/out
 make -C leukocyte/out
 make -C leukocyte-preprocessing/out
 make -C lud/out
 make -C myocyte/out
 make -C nn/out
 make -C nw/out
 make -C particle/out
 make -C pathfinder/out
 make -C srad/out
 make -C stream/out
 