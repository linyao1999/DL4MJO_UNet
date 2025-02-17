#!/bin/bash

# salloc --nodes 1 --qos interactive --time 04:00:00 --constraint gpu --gpus 4 --account=dasrepo_g

# salloc --nodes 1 --qos interactive --time 04:00:00 --constraint gpu --gpus 4 --account=dasrepo_g
module load pytorch/1.11.0
module load parallel  # Load GNU Parallel module if available

# export lead=15

generate_commands() {
    for lead0 in {0..15..5}; do
        export lead=$lead0
        for mjo_ind0 in 'RMM' 'ROMI'; do
            export mjo_ind=$mjo_ind0
            for zero_num0 in 15 20; do
                export zero_num=$zero_num0
                logname=lead$lead\_channel$mjo_ind\_exp$zero_num
                echo "CUDA_VISIBLE_DEVICES=$(( ($exp - 1) / 4 )) lead=$lead mjo_ind=$mjo_ind zero_num=$zero_num python3 zero_out_unimportant.py > /pscratch/sd/l/linyaoly/outlog/$logname.txt"
            done
        done
    done
}

export -f generate_commands
# export -f contribution.py  # If your script relies on this file

generate_commands | parallel -j 4

