#!/bin/bash
#SBATCH -A dasrepo_g
#SBATCH -C gpu
#SBATCH -q preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH -t 01:00:00
#SBATCH --output=outlog/%j.out

# salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 4 --account=dasrepo_g

# Load the required module
module load pytorch/1.11.0

# Set environment variables
# varn='olr'
# export varn

mjo_ind='ROMI'
export mjo_ind 

vertical_mode=1
ysta_train=1979
yend_train=2015
ysta_test=2015
yend_test=2020
export vertical_mode ysta_train ysta_test yend_train yend_test

lat_lim=20
export lat_lim

# # Enable debugging
# set -x

# Run the Python scripts simultaneously with different 'lead30d' values
lead30d=15
export lead30d
logname="reloadlog${mjo_ind}_${varn}yproj_${lat_lim}deg_m${m}_${mflg}_wnx${wnx}_${wnxflg}_dailyinput_mem${memlen}d_lead${lead30d}"
export logname
CUDA_VISIBLE_DEVICES=0 python3 Unet4MJO_reload.py > ./outlog/$logname.txt &
# CUDA_VISIBLE_DEVICES=0 python3 Unet4MJO_loop.py > ./outlog/$logname.txt &

lead30d=20
export lead30d
logname="reloadlog${mjo_ind}_${varn}yproj_${lat_lim}deg_m${m}_${mflg}_wnx${wnx}_${wnxflg}_dailyinput_mem${memlen}d_lead${lead30d}"
export logname
CUDA_VISIBLE_DEVICES=1 python3 Unet4MJO_reload.py > ./outlog/$logname.txt &
# CUDA_VISIBLE_DEVICES=1 python3 Unet4MJO_loop.py > ./outlog/$logname.txt &

lead30d=25
export lead30d
logname="reloadlog${mjo_ind}_${varn}yproj_${lat_lim}deg_m${m}_${mflg}_wnx${wnx}_${wnxflg}_dailyinput_mem${memlen}d_lead${lead30d}"
export logname
CUDA_VISIBLE_DEVICES=2 python3 Unet4MJO_reload.py > ./outlog/$logname.txt &
# CUDA_VISIBLE_DEVICES=2 python3 Unet4MJO_loop.py > ./outlog/$logname.txt &

lead30d=30
export lead30d
logname="reloadlog${mjo_ind}_${varn}yproj_${lat_lim}deg_m${m}_${mflg}_wnx${wnx}_${wnxflg}_dailyinput_mem${memlen}d_lead${lead30d}"
export logname
CUDA_VISIBLE_DEVICES=3 python3 Unet4MJO_reload.py > ./outlog/$logname.txt &
# CUDA_VISIBLE_DEVICES=3 python3 Unet4MJO_loop.py > ./outlog/$logname.txt &

wait
