#!/bin/bash
#SBATCH -A dasrepo_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH -t 05:00:00
#SBATCH --output=outlog/%j.out

# salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 4 --account=dasrepo_g

# Set environment variables
# varn='olr'
# export varn

mjo_ind='RMM'
export mjo_ind 

ysta_train=1979
yend_train=2015
ysta_test=2015
yend_test=2024
export ysta_train ysta_test yend_train yend_test

lat_lim=20
export lat_lim

# # Enable debugging
# set -x

# Run the Python scripts simultaneously with different 'lead30d' values
lead30d=35
export lead30d
logname="log${mjo_ind}_${varn}yproj_${lat_lim}deg_m${m}_${mflg}_wnx${wnx}_${wnxflg}_dailyinput_mem${memlen}d_lead${lead30d}_exp${exp_num}"
export logname
CUDA_VISIBLE_DEVICES=0 python3 Unet4MJO.py > ./outlog/$logname.txt &

lead30d=20
export lead30d
logname="log${mjo_ind}_${varn}yproj_${lat_lim}deg_m${m}_${mflg}_wnx${wnx}_${wnxflg}_dailyinput_mem${memlen}d_lead${lead30d}_exp${exp_num}"
export logname
CUDA_VISIBLE_DEVICES=1 python3 Unet4MJO.py > ./outlog/$logname.txt &

lead30d=25
export lead30d
logname="log${mjo_ind}_${varn}yproj_${lat_lim}deg_m${m}_${mflg}_wnx${wnx}_${wnxflg}_dailyinput_mem${memlen}d_lead${lead30d}_exp${exp_num}"
export logname
CUDA_VISIBLE_DEVICES=2 python3 Unet4MJO.py > ./outlog/$logname.txt &

lead30d=30
export lead30d
logname="log${mjo_ind}_${varn}yproj_${lat_lim}deg_m${m}_${mflg}_wnx${wnx}_${wnxflg}_dailyinput_mem${memlen}d_lead${lead30d}_exp${exp_num}"
export logname
CUDA_VISIBLE_DEVICES=3 python3 Unet4MJO.py > ./outlog/$logname.txt &

wait
