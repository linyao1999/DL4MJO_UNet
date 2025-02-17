#!/bin/bash

dataflg='new'
export dataflg  

varn='olr' 
memlen=1
c=51
export c varn memlen 

# E4
mflg='resi'
wnxflg='resi'
export mflg wnxflg

for m in 3 5 10 #  m is the total number of meridional wave numbers 
do 
    for wnx in 2 4 9 14
    do
        for exp_num in '' '1' '2' '3' '4' '5' '6' '7' '8' '9' '10' # "tcwv" "q200" "q500" "q850" "T200" "T500" "T850" "Z200" "Z500" "Z850" "v200" "v500" "v850" "u200" "u500" "u850" "prep" "sst" # 
        do 
            export wnx 
            export exp_num
            export m
            sbatch arun_4gpus_a.slurm
            sbatch arun_4gpus_b.slurm
            sbatch arun_4gpus_c.slurm
            sbatch arun_4gpus_d.slurm
        done
    done
done 


# E3
mflg='all'
wnxflg='all'
export mflg wnxflg

for m in 3 5 10 #  m is the total number of meridional wave numbers 
do 
    for wnx in 2 4 9 14
    do
        for exp_num in '1' '2' '3' '4' '5' '6' '7' '8' '9' '10' # "tcwv" "q200" "q500" "q850" "T200" "T500" "T850" "Z200" "Z500" "Z850" "v200" "v500" "v850" "u200" "u500" "u850" "prep" "sst" # 
        do 
            export wnx 
            export exp_num
            export m
            sbatch arun_4gpus_a.slurm
            sbatch arun_4gpus_b.slurm
            sbatch arun_4gpus_c.slurm
            sbatch arun_4gpus_d.slurm
        done
    done
done 

for m in 3 5 10 #  m is the total number of meridional wave numbers 
do 
    for wnx in 2 
    do
        for exp_num in '' # "tcwv" "q200" "q500" "q850" "T200" "T500" "T850" "Z200" "Z500" "Z850" "v200" "v500" "v850" "u200" "u500" "u850" "prep" "sst" # 
        do 
            export wnx 
            export exp_num
            export m
            sbatch arun_4gpus_a.slurm
            sbatch arun_4gpus_b.slurm
            sbatch arun_4gpus_c.slurm
            sbatch arun_4gpus_d.slurm
        done
    done
done 



# E2
mflg='all'
wnxflg='off'
export mflg wnxflg

for m in 3 5 10 #  m is the total number of meridional wave numbers 
do 
    for wnx in 1
    do
        for exp_num in '1' '2' '3' '4' '5' '6' '7' '8' '9' '10' # "tcwv" "q200" "q500" "q850" "T200" "T500" "T850" "Z200" "Z500" "Z850" "v200" "v500" "v850" "u200" "u500" "u850" "prep" "sst" # 
        do 
            export wnx 
            export exp_num
            export m
            sbatch arun_4gpus_a.slurm
            sbatch arun_4gpus_b.slurm
            sbatch arun_4gpus_c.slurm
            sbatch arun_4gpus_d.slurm
        done
    done
done 
