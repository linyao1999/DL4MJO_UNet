#!/bin/bash

m=1
mflg='off'
wnx=1
wnxflg='off'
memlen=1
c=51
# exp_num=''

export c wnx mflg wnxflg memlen m
# {1..10} #

for exp_num in '' # 2 3 4 5 6
do 
    for varn in "olr" # "tcwv" "q200" "q500" "q850" "T200" "T500" "T850" "Z200" "Z500" "Z850" "v200" "v500" "v850" "u200" "u500" "u850" "prep" "sst" # 
    do 
        export exp_num
        export varn
        # export exp_num
        sbatch arun_4gpus_a.slurm
        sbatch arun_4gpus_b.slurm
        sbatch arun_4gpus_c.slurm
        sbatch arun_4gpus_d.slurm
    done
done 
