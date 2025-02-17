#!/bin/bash


export OMP_NUM_THREADS=1  # Ensures that Python does not use more threads than necessary

# Define a function to run the Python script
run_script() {
    mjo_ind=$1
    lead=$2
    export mjo_ind
    export lead
    srun --exclusive -N1 -n1 python3 scale_change_L2L6.py &
}

# Run for RMM
mjo_ind='RMM'
for lead in 15; do
    run_script $mjo_ind $lead
done

# Run for ROMI
mjo_ind='ROMI'
for lead in 25; do
    run_script $mjo_ind $lead
done

# Wait for all background jobs to finish
wait
