#!/bin/bash

dataflg='new'
export dataflg  

m=1
mflg='off'
wnx=1
wnxflg='off'
memlen=1

c=51

export m mflg wnx wnxflg memlen c 

for exp_num in '' # '' '1' '2' '3' '4' '5' '6' '7' '8' '9' '10'
do
    export exp_num
    sbatch arun_4gpus_a.slurm
    sbatch arun_4gpus_b.slurm
    sbatch arun_4gpus_c.slurm
    sbatch arun_4gpus_d.slurm
done


