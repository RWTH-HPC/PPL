#!/bin/bash

source /home/as388453/software/.pplcpp_rocky
source /home/as388453/software/.pplrc_rocky
rm c18$1_$2_$6.txt
touch c18$1_$2_$6.txt
rm done_opt.txt

mkdir -p $6
java --add-opens java.base/java.lang=ALL-UNNAMED -jar /home/as388453/parallel-pattern-dsl/Samples/rodinia-ppl/PPL_test.jar -n "../../clusters/cluster_c18$1_$2.json" -d 50 -s 50 -i "LavaMD.par" -o "$6/$1$2/LavaMD_$1_$2.cxx" >> c18$1_$2_$6.txt

mkdir -p $6/$1$2/bin $6/$1$2/obj $6/$1$2/cuda

make -C $6/$1$2
touch done_opt.txt
#sbatch  -n $2 -A rwth1270 -J run_$5 -o out_run_$1$2.txt -c 48 --mem=0 -t 02:00:00 --exclusive $4 run.sh $1 $2 $3
