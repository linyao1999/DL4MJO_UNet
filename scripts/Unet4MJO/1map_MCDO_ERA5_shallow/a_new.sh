#!/bin/bash

m=10
mflg='all'
wnx=9
wnxflg='all'
export m mflg wnx wnxflg

for exp_num in '1' '2' '3' '4' '5' '6' '7' '8' '9' '10' 
do
    export exp_num
    sbatch run1.slurm
    sbatch run2.slurm
    sbatch run3.slurm
    sbatch run4.slurm
done

m=10
mflg='resi'
wnx=9
wnxflg='resi'
export m mflg wnx wnxflg

for exp_num in '1' '2' '3' '4' '5' '6' '7' '8' '9' '10' 
do
    export exp_num
    sbatch run1.slurm
    sbatch run2.slurm
    sbatch run3.slurm
    sbatch run4.slurm
done


# dataflg='new'
# export dataflg  

# m=10
# mflg='all'
# # varn='olr'
# # mflg='resi'
# # wnx=1
# wnxflg='all'
# memlen=1
# c=51
# export c m mflg wnxflg memlen 
# # {1..10} #

# for wnx in 9 # 0 1 2 3 4 5 # 2 3 4 5 6
# do 
#     for exp_num in '' '1' '2' '3' '4' '5' '6' '7' '8' '9' '10'
#     do
#         for varn in "olr" # "tcwv" "q200" "q500" "q850" "T200" "T500" "T850" "Z200" "Z500" "Z850" "v200" "v500" "v850" "u200" "u500" "u850" "prep" "sst" # 
#         do 
#             export wnx 
#             export varn
#             export exp_num
#             sbatch arun_4gpus_a.slurm
#             sbatch arun_4gpus_b.slurm
#             sbatch arun_4gpus_c.slurm
#             sbatch arun_4gpus_d.slurm
#         done
#     done
# done 

# dataflg='new'
# export dataflg  

# m=10
# mflg='resi'
# # varn='olr'
# # mflg='resi'
# # wnx=1
# wnxflg='resi'
# memlen=1
# c=51
# export c m mflg wnxflg memlen 
# # {1..10} #

# for wnx in 9 # 0 1 2 3 4 5 # 2 3 4 5 6
# do 
#     for exp_num in '' '1' '2' '3' '4' '5' '6' '7' '8' '9' '10'
#     do
#         for varn in "olr" # "tcwv" "q200" "q500" "q850" "T200" "T500" "T850" "Z200" "Z500" "Z850" "v200" "v500" "v850" "u200" "u500" "u850" "prep" "sst" # 
#         do 
#             export wnx 
#             export varn
#             export exp_num
#             sbatch arun_4gpus_a.slurm
#             sbatch arun_4gpus_b.slurm
#             sbatch arun_4gpus_c.slurm
#             sbatch arun_4gpus_d.slurm
#         done
#     done
# done 


# dataflg='new'
# export dataflg  

# # m=1
# mflg='one'
# # varn='olr'
# # mflg='resi'
# wnx=1
# wnxflg='off'
# memlen=1
# c=51
# export c wnx mflg wnxflg memlen 
# # {1..10} #

# for m in 1 2 3 4 5 6 # 2 3 4 5 6
# do 
#     for exp_num in '' '1' '2' '3' '4' '5' '6' '7' '8' '9' '10'
#     do
#         for varn in "olr" # "tcwv" "q200" "q500" "q850" "T200" "T500" "T850" "Z200" "Z500" "Z850" "v200" "v500" "v850" "u200" "u500" "u850" "prep" "sst" # 
#         do 
#             export m 
#             export varn
#             export exp_num
#             sbatch arun_4gpus_c.slurm
#             sbatch arun_4gpus_d.slurm
#         done
#     done
# done 

# # m=1
# mflg='even'
# # varn='olr'
# # mflg='resi'
# wnx=2
# wnxflg='no0'
# memlen=1
# c=51
# export c wnx mflg wnxflg memlen 
# # {1..10} #

# for m in 3
# do 
#     for exp_num in '' '1' '2' '3' '4' '5' '6' '7' '8' '9' '10'
#     do
#         for varn in "olr" # "tcwv" # "q200" "q500" "q850" "T200" "T500" "T850" "Z200" "Z500" "Z850" "v200" "v500" "v850" "u200" "u500" "u850" "prep" "sst" # 
#         do 
#             export m 
#             export varn
#             export exp_num
#             sbatch arun_4gpus_a.slurm
#         done
#     done
# done 

# m=1
# mflg='off'
# # varn='olr'
# # mflg='resi'
# # wnx=1
# wnxflg='one'
# memlen=1
# c=51
# export c m mflg wnxflg memlen 
# # {1..10} #

# for wnx in 0 1 2 3 4 5 6 7 8 9 10
# do 
#     for exp_num in '' '1' '2' '3' '4' '5' '6' '7' '8' '9' '10'
#     do
#         for varn in "olr" "tcwv" # "q200" "q500" "q850" "T200" "T500" "T850" "Z200" "Z500" "Z850" "v200" "v500" "v850" "u200" "u500" "u850" "prep" "sst" # 
#         do 
#             export wnx 
#             export varn
#             export exp_num
#             sbatch arun_4gpus_a.slurm
#         done
#     done
# done 

# m=1
# mflg='off'
# # varn='olr'
# # mflg='resi'
# # wnx=1
# wnxflg='resi'
# memlen=1
# c=51
# export c m mflg wnxflg memlen 
# # {1..10} #

# for wnx in 10
# do 
#     for exp_num in '' '1' '2' '3' '4' '5' '6' '7' '8' '9' '10'
#     do
#         for varn in "olr" "tcwv" # "q200" "q500" "q850" "T200" "T500" "T850" "Z200" "Z500" "Z850" "v200" "v500" "v850" "u200" "u500" "u850" "prep" "sst" # 
#         do 
#             export wnx 
#             export varn
#             export exp_num
#             sbatch arun_4gpus_a.slurm
#         done
#     done
# done 


# for varn in "olr" "tcwv" "q200" "q500" "q850" "T200" "T500" "T850" "Z200" "Z500" "Z850" "v200" "v500" "v850" "u200" "u500" "u850" "prep" "sst" # 
# do 
#     export varn
#     sbatch arun_4gpus_a1.slurm
#     sbatch arun_4gpus_a2.slurm
#     sbatch arun_4gpus_a3.slurm

# done


# for exp_num in '1' '2' '3' '4' '5' '6' '7' '8' '9' '10'
# do
#     for varn in "olr" "tcwv" "q200" "q500" "q850" "T200" "T500" "T850" "Z200" "Z500" "Z850" "v200" "v500" "v850" "u200" "u500" "u850" "prep" "sst" # 
#     do 
#         export varn
#         export exp_num
#         sbatch arun_4gpus_a.slurm
#         # sbatch arun_4gpus_b.slurm
#         # sbatch arun_4gpus_c.slurm
#         # sbatch arun_4gpus_d.slurm
#     done
# done

# m=1
# mflg='off'
# wnx=1
# wnxflg='off'
# memlen=1
# c=20
# export c m mflg wnx wnxflg memlen 

# for varn in "u850" "olr" # "olr" "tcwv" "q200" "q500" "q850" "T200" "T500" "T850" "Z200" "Z500" "Z850" "v200" "v500" "v850" "u200" "u500" "u850" "prep" "sst" # 
# do 
#     export varn
#     # sbatch arun_4gpus_a.slurm
#     # sbatch arun_4gpus_b.slurm
#     # sbatch arun_4gpus_c.slurm
#     sbatch arun_4gpus_d.slurm
# done


# m=8
# mflg='all'
# wnx=10
# wnxflg='all'
# memlen=1

# export m mflg wnx wnxflg memlen 

# for varn in "u850" # "olr" "tcwv" "q200" "q500" "q850" "T200" "T500" "T850" "Z200" "Z500" "Z850" "v200" "v500" "v850" "u200" "u500" "u850" "prep" "sst" # 
# do 
#     export varn
#     sbatch arun_4gpus_a.slurm
#     sbatch arun_4gpus_b.slurm
#     sbatch arun_4gpus_c.slurm
#     sbatch arun_4gpus_d.slurm
# done
