#!/bin/bash

# salloc --nodes 1 --qos interactive --time 04:00:00 --constraint gpu --gpus 4 --account=dasrepo_g

# module load pytorch/1.11.0
module load parallel  # Load GNU Parallel module if available

# export lead=15
export cut_m=1
export cut_k=9

generate_commands() {
    for exp_num in "" 1 2 3 4 5 6 7 8 9 10; do
        export exp=$exp_num
        for lead0 in {0..30..5}; do
            export lead=$lead0
            device_id=$(( lead % 4 ))
            for mjo_ind0 in 'RMM' 'ROMI'; do
                export mjo_ind=$mjo_ind0
                for flg in "off"; do 
                    export cut_m_flg=$flg
                    export cut_k_flg='resi'
                    logname=exp$exp\lead$lead\mjo_ind$mjo_ind\cut_m$cut_m_flg\cut_k$cut_k_flg
                    
                    echo "CUDA_VISIBLE_DEVICES=$device_id exp_num=$exp lead=$lead mjo_ind=$mjo_ind cut_m=$cut_m cut_k=$cut_k cut_m_flg=$cut_m_flg cut_k_flg=$cut_k_flg python3 filter_input_MD.py > /pscratch/sd/l/linyaoly/outlog/$logname.txt"
                done
            done
        done
    done 
}

export -f generate_commands

generate_commands | parallel -j 4