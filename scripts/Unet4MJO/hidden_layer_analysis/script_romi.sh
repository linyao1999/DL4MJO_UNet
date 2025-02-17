#!/bin/bash

# salloc --nodes 1 --qos interactive --time 04:00:00 --constraint gpu --gpus 4 --account=dasrepo_g
module load pytorch/1.11.0
module load parallel  # Load GNU Parallel module if available

# export lead=15

generate_commands() {
    for lead0 in {0..25..5}; do
        export lead=$lead0
        for exp0 in ''; do
            export exp=$exp0
            for channel0 in {0..191}; do
                export channel=$channel0
                logname=romi_lead$lead\_channel$channel\_exp$exp
                echo "CUDA_VISIBLE_DEVICES=$(( ($exp - 1) / 4 )) lead=$lead channel=$channel exp=$exp python3 contribution_small_romi.py > /pscratch/sd/l/linyaoly/outlog/$logname.txt"
            done
        done
    done
}

export -f generate_commands
# export -f contribution.py  # If your script relies on this file

generate_commands | parallel -j 4

# for lead0 in {0..30..5}
# do
#     export lead=$lead0
#     for channel0 in {0..191}
#     do
#         export channel=$channel0
#         python3 contribution.py
#     done
# done


# for lead0 in {0..30..5}
# do
#     export lead=$lead0
#     python3 contribution.py
# done
