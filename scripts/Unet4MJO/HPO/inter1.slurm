#!/bin/bash

# salloc --nodes 1 --qos interactive --time 02:00:00 --constraint gpu --gpus 4 --account=m4736_g


# Load the required module
module load pytorch/1.11.0

# Set environment variables
batch_size=20
kernel_size=5
drop_prob=0.2
optimizer="SGD"
learning_rate=0.001

export batch_size
export kernel_size
export drop_prob
export optimizer_type=$optimizer
export learning_rate

mjo_ind='RMM'
export mjo_ind 
m=10
mflg='resi'
wnx=9
wnxflg='resi'
export m mflg wnx wnxflg

# # Enable debugging
# set -x

# Run the Python scripts simultaneously with different 'lead30d' values
lead30d=0
export lead30d
logname="logbs${batch_size}_ks${kernel_size}_dp${drop_prob}_${optimizer}_lr${learning_rate}_${mjo_ind}_${varn}yproj_${lat_lim}deg_m${m}_${mflg}_wnx${wnx}_${wnxflg}_dailyinput_mem${memlen}d_lead${lead30d}_exp${exp_num}"
export logname
# CUDA_VISIBLE_DEVICES=0 python3 Unet4MJO_reload.py > ./outlog/$logname.txt &
CUDA_VISIBLE_DEVICES=0 python3 Unet4MJO.py > /pscratch/sd/l/linyaoly/outlog/$logname.txt &
# CUDA_VISIBLE_DEVICES=0 python3 Unet4MJO_rmmasinput.py > /pscratch/sd/l/linyaoly/outlog/$logname.txt &

lead30d=5
export lead30d
logname="logbs${batch_size}_ks${kernel_size}_dp${drop_prob}_${optimizer}_lr${learning_rate}_${mjo_ind}_${varn}yproj_${lat_lim}deg_m${m}_${mflg}_wnx${wnx}_${wnxflg}_dailyinput_mem${memlen}d_lead${lead30d}_exp${exp_num}"
export logname
# CUDA_VISIBLE_DEVICES=1 python3 Unet4MJO_reload.py > ./outlog/$logname.txt &
# CUDA_VISIBLE_DEVICES=1 python3 Unet4MJO_amp1.py > /pscratch/sd/l/linyaoly/outlog/$logname.txt &
CUDA_VISIBLE_DEVICES=1 python3 Unet4MJO.py > /pscratch/sd/l/linyaoly/outlog/$logname.txt &


lead30d=10
export lead30d
logname="logbs${batch_size}_ks${kernel_size}_dp${drop_prob}_${optimizer}_lr${learning_rate}_${mjo_ind}_${varn}yproj_${lat_lim}deg_m${m}_${mflg}_wnx${wnx}_${wnxflg}_dailyinput_mem${memlen}d_lead${lead30d}_exp${exp_num}"
export logname
# CUDA_VISIBLE_DEVICES=2 python3 Unet4MJO_reload.py > ./outlog/$logname.txt &
CUDA_VISIBLE_DEVICES=2 python3 Unet4MJO.py > /pscratch/sd/l/linyaoly/outlog/$logname.txt &

lead30d=15
export lead30d
logname="logbs${batch_size}_ks${kernel_size}_dp${drop_prob}_${optimizer}_lr${learning_rate}_${mjo_ind}_${varn}yproj_${lat_lim}deg_m${m}_${mflg}_wnx${wnx}_${wnxflg}_dailyinput_mem${memlen}d_lead${lead30d}_exp${exp_num}"
export logname
# CUDA_VISIBLE_DEVICES=3 python3 Unet4MJO_reload.py > ./outlog/$logname.txt &
CUDA_VISIBLE_DEVICES=3 python3 Unet4MJO.py > /pscratch/sd/l/linyaoly/outlog/$logname.txt &

wait
