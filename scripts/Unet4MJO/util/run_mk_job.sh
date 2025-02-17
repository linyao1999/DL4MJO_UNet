#!/bin/bash
#SBATCH -A dasrepo_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH -t 01:20:00
#SBATCH --output=outlog/%j.out

module load pytorch/1.11.0
python3 run_mk.py 