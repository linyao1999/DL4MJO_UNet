#!/bin/bash

m=1
mflg='off'
wnx=1
wnxflg='off'
memlen=1

mcdp=5  # Monte Carlo Dropout rate = mcdp * 1%

export m mflg wnx wnxflg memlen mcdp

for varn in "olr" # "tcwv" "q200" "q500" "q850" "T200" "T500" "T850" "Z200" "Z500" "Z850" "v200" "v500" "v850" "u200" "u500" "u850" "prep" "sst" # 
do 
    export varn
    sbatch arun_4gpus_a_reload.slurm
    sbatch arun_4gpus_b_reload.slurm
    sbatch arun_4gpus_c_reload.slurm
    sbatch arun_4gpus_d_reload.slurm
done

