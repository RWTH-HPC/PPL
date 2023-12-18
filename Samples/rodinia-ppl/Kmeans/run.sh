#!/bin/bash

ITER=$3
rm done_run.txt

source /home/as388453/software/.pplcpp_intel

make -C $4/$1$2 clean
make -C $4/$1$2
for (( i=0; i<$ITER; i++ ))
do

    mpiexec -n $2 $4/$1$2/bin/Kmeans_$1_$2.exe >> c18$1_$2_$4.txt
done
touch done_run.txt
