#!/bin/bash

# salloc --nodes 1 --qos interactive --time 02:00:00 --constraint gpu --gpus 4 --account=m4736_g


# Load the required module
module load pytorch/1.11.0

dataflg='new'
export dataflg

# Set environment variables
varn='olr'
export varn

mjo_ind='ROMI'
export mjo_ind 

m=1
mflg='off'
# varn='olr'
# mflg='resi'
wnx=1
wnxflg='off'
memlen=1
c=51
exp_num='1'
export c wnx m mflg wnxflg memlen exp_num

ysta_train=1979
yend_train=2015
ysta_test=2015
yend_test=2022
export ysta_train ysta_test yend_train yend_test

lat_lim=20
export lat_lim

# Run the Python scripts simultaneously with different 'lead30d' values
lead30d=0
export lead30d
logname="log${mjo_ind}_${varn}yproj_${lat_lim}deg_m${m}_${mflg}_wnx${wnx}_${wnxflg}_dailyinput_mem${memlen}d_lead${lead30d}_exp${exp_num}"
export logname
CUDA_VISIBLE_DEVICES=0 python3 Unet4MJO.py > /pscratch/sd/l/linyaoly/outlog/$logname.txt &

lead30d=25
export lead30d
logname="log${mjo_ind}_${varn}yproj_${lat_lim}deg_m${m}_${mflg}_wnx${wnx}_${wnxflg}_dailyinput_mem${memlen}d_lead${lead30d}_exp${exp_num}"
export logname
CUDA_VISIBLE_DEVICES=1 python3 Unet4MJO.py > /pscratch/sd/l/linyaoly/outlog/$logname.txt &

lead30d=30
export lead30d
logname="log${mjo_ind}_${varn}yproj_${lat_lim}deg_m${m}_${mflg}_wnx${wnx}_${wnxflg}_dailyinput_mem${memlen}d_lead${lead30d}_exp${exp_num}"
export logname
CUDA_VISIBLE_DEVICES=2 python3 Unet4MJO.py > /pscratch/sd/l/linyaoly/outlog/$logname.txt &

lead30d=35
export lead30d
logname="log${mjo_ind}_${varn}yproj_${lat_lim}deg_m${m}_${mflg}_wnx${wnx}_${wnxflg}_dailyinput_mem${memlen}d_lead${lead30d}_exp${exp_num}"
export logname
CUDA_VISIBLE_DEVICES=3 python3 Unet4MJO.py > /pscratch/sd/l/linyaoly/outlog/$logname.txt &

wait

# srun -u -n 1 --gpus-per-task 1 -c 32 --mem-per-gpu 60G python3 Unet4MJO_loop.py --lead30d 10 > ./outlog/log2.txt &
# srun -u -n 1 --gpus-per-task 1 -c 32 --mem-per-gpu 60G python3 Unet4MJO_loop.py --lead30d 15 > ./outlog/log3.txt &
# srun -u -n 1 --gpus-per-task 1 -c 32 --mem-per-gpu 60G python3 Unet4MJO_loop.py --lead30d 20 > ./outlog/log4.txt &

# Wait for all tasks to complete
# wait

# for lat_lim in 20 
# do 
#     for wnx in 5 # 50 100 # 1 2 10
#     do 
#         for memlen in 1 # 30  # 1 30 15 5
#         do 
#             for lead30d in 1 5 10 13 # 15 20 25 30  # {10..20}
#             do 
#                 export lead30d
#                 # echo $lead30d 
#                 export memlen
#                 # echo $memlen
#                 export wnx
#                 export lat_lim

#                 logname="logRMM_1mapyproj_${lat_lim}deg_m${m}_${mflg}_wnx${wnx}_${wnxflg}_dailyinput_mem${memlen}d_lead${lead30d}"
#                 export logname
#                 echo $logname
#                 # sbatch arunsub.slurm 
#                 srun --exact -u -n 1 --gpus-per-task 1 -c 32 --mem-per-gpu=60G python3 Unet4MJO_loop.py > ./outlog/$logname.txt &
#             done
#         done 
#     done 
# done

# wait