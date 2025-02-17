#!/bin/bash

# salloc --nodes 1 --qos interactive --time 04:00:00 --constraint gpu --gpus 4 --account=dasrepo_g

module load pytorch/1.11.0

module load parallel  # Load GNU Parallel module if available

# export lead=15

generate_commands() {
    for lead0 in {0..25..5}; do
        export leadmjo=$lead0
        for mjo_ind0 in 'RMM' 'ROMI'; do
            export mjo_ind=$mjo_ind0
            logname=lead$leadmjo\mjo_ind$mjo_ind
            echo "CUDA_VISIBLE_DEVICES=$(( ($exp - 1) / 4 )) leadmjo=$leadmjo mjo_ind=$mjo_ind python3 get_L2L6_std.py > /pscratch/sd/l/linyaoly/outlog/$logname.txt"
        done
    done
}

export -f generate_commands

generate_commands | parallel -j 4
