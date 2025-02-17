#!/bin/bash
mflg='one'
wnx=1
wnxflg='off'
memlen=1
c=51
export c wnx mflg wnxflg memlen 

for exp_num in '' '1' '2' '3' '4' '5' '6' '7' '8' '9' '10'
do
    for varn in "olr" # "tcwv" "q200" "q500" "q850" "T200" "T500" "T850" "Z200" "Z500" "Z850" "v200" "v500" "v850" "u200" "u500" "u850" "prep" "sst" # 
    do 
        export varn
        export exp_num
        sbatch arun_4gpus_c1.slurm  # m=0 or 1; lead = 0 or 5
        sbatch arun_4gpus_c2.slurm  # m=2 or 3; lead = 0 or 5
        sbatch arun_4gpus_c3.slurm  # m=4 or 5; lead = 0 or 5
    done
done
