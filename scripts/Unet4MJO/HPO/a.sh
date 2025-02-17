#!/bin/bash

# Define the ranges for each hyperparameter
batch_sizes=(32 64 128)
kernel_sizes=(3 5 7)
drop_probs=(0.1 0.3 0.5)
optimizers=("Adam" "SGD")
learning_rates=(0.0005 0.001 0.005)

# Output directory to save results
output_dir="./outlog"
mkdir -p $output_dir

# Loop over all combinations of hyperparameters
for batch_size in "${batch_sizes[@]}"; do
  for kernel_size in "${kernel_sizes[@]}"; do
    for drop_prob in "${drop_probs[@]}"; do
      for optimizer in "${optimizers[@]}"; do
        for learning_rate in "${learning_rates[@]}"; do
          
          # Set environment variables
          export batch_size
          export kernel_size
          export drop_prob
          export optimizer_type=$optimizer
          export learning_rate

          # Define a unique identifier for this combination
          exp_id="bs${batch_size}_ks${kernel_size}_dp${drop_prob}_${optimizer}_lr${learning_rate}_lead${lead}_m${m}_wnx${wnx}"
          
          # Run your training script with the current hyperparameter combination
          echo "Running experiment $exp_id"
          sbatch arun_4gpus_d.slurm
          sbatch arun_4gpus_e.slurm

        done
      done
    done
  done
done

echo "Grid search completed."
