#!/bin/bash
#SBATCH -A dasrepo_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH -t 05:20:00
#SBATCH --output=outlog/%j.out
# salloc --nodes 1 --qos interactive --time 04:00:00 --constraint gpu --gpus 4 --account=dasrepo_g

module load pytorch/1.11.0

export lead=15
for exp0 in {7..7}
do
    export exp=$exp0
    for channel0 in {0..191}
    do
        export channel=$channel0
        python3 contribution.py
    done
done


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
