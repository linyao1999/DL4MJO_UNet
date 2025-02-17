#!/bin/bash

# salloc --nodes 1 --qos interactive --time 02:00:00 --constraint gpu --gpus 4 --account=dasrepo_g


# Load the required module
# module load pytorch/1.11.0

dataflg='new'
export dataflg

# Set environment variables
varn='olr'
export varn

mjo_ind='RMM'
export mjo_ind 

m=1
mflg='off'
# varn='olr'
# mflg='resi'
wnx=3
wnxflg='resi'
memlen=1
c=51
exp_num=''
export c wnx m mflg wnxflg memlen exp_num

ysta_train=1979
yend_train=2015
ysta_test=2015
yend_test=2023
export ysta_train ysta_test yend_train yend_test

lat_lim=20
export lat_lim

# Run the Python scripts simultaneously with different 'lead30d' values
lead30d=0
export lead30d
logname="${mjo_ind}_${varn}yproj_${lat_lim}deg_m${m}_${mflg}_wnx${wnx}_${wnxflg}_dailyinput_mem${memlen}d_lead${lead30d}_exp${exp_num}"
export logname
CUDA_VISIBLE_DEVICES=0 python3 Unet4MJO_loopnew.py > /pscratch/sd/l/linyaoly/outlog/$logname.txt &

lead30d=10
export lead30d
logname="${mjo_ind}_${varn}yproj_${lat_lim}deg_m${m}_${mflg}_wnx${wnx}_${wnxflg}_dailyinput_mem${memlen}d_lead${lead30d}_exp${exp_num}"
export logname
CUDA_VISIBLE_DEVICES=1 python3 Unet4MJO_loopnew.py > /pscratch/sd/l/linyaoly/outlog/$logname.txt &

lead30d=15
export lead30d
logname="${mjo_ind}_${varn}yproj_${lat_lim}deg_m${m}_${mflg}_wnx${wnx}_${wnxflg}_dailyinput_mem${memlen}d_lead${lead30d}_exp${exp_num}"
export logname
CUDA_VISIBLE_DEVICES=2 python3 Unet4MJO_loopnew.py > /pscratch/sd/l/linyaoly/outlog/$logname.txt &

lead30d=5
export lead30d
logname="${mjo_ind}_${varn}yproj_${lat_lim}deg_m${m}_${mflg}_wnx${wnx}_${wnxflg}_dailyinput_mem${memlen}d_lead${lead30d}_exp${exp_num}"
export logname
CUDA_VISIBLE_DEVICES=3 python3 Unet4MJO_loopnew.py > /pscratch/sd/l/linyaoly/outlog/$logname.txt &

wait
