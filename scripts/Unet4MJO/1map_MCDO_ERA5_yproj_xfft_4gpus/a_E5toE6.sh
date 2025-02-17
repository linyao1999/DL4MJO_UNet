#!/bin/bash

dataflg='new'
export dataflg  


varn='olr' 
memlen=1
c=51
export c varn memlen 

# E5
mflg='all'
wnxflg='resi'
export mflg wnxflg

for m in 3 # 3 5 10 #  m is the total number of meridional wave numbers 
do 
    for wnx in 2 4 9 14
    do
        for exp_num in '' '1' '2' '3' '4' '5' '6' '7' '8' '9' '10' # "tcwv" "q200" "q500" "q850" "T200" "T500" "T850" "Z200" "Z500" "Z850" "v200" "v500" "v850" "u200" "u500" "u850" "prep" "sst" # 
        do 
            export wnx 
            export exp_num
            export m
            # sbatch arun_4gpus_a.slurm
            # sbatch arun_4gpus_b.slurm
            sbatch arun_4gpus_c.slurm
            sbatch arun_4gpus_d.slurm
        done
    done
done 


# # E6
# mflg='off'
# wnxflg='resi'
# export mflg wnxflg

# for m in 1 #  m is the total number of meridional wave numbers 
# do 
#     for wnx in 2 4 9 14
#     do
#         for exp_num in '' '1' '2' '3' '4' '5' '6' '7' '8' '9' '10' # "tcwv" "q200" "q500" "q850" "T200" "T500" "T850" "Z200" "Z500" "Z850" "v200" "v500" "v850" "u200" "u500" "u850" "prep" "sst" # 
#         do 
#             export wnx 
#             export exp_num
#             export m
#             sbatch arun_4gpus_a.slurm
#             sbatch arun_4gpus_b.slurm
#             sbatch arun_4gpus_c.slurm
#             sbatch arun_4gpus_d.slurm
#         done
#     done
# done 

