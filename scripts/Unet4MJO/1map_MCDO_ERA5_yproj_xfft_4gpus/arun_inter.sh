#!/bin/bash

for exp_num in ''    # '' '1' '2' '3' '4' '5' '6' '7' '8' '9' '10' 
do 
    export exp_num
    ./arun_new_inter1.slurm
    ./arun_new_inter2.slurm
done

wait 