#!/bin/bash

#SBATCH --job-name=transcoder_finetune_array 
#SBATCH --partition=compsci-gpu                   
#SBATCH --mem=32G
#SBATCH --gres=gpu:a5000:1
#SBATCH --time=7-00:00:00                    
#SBATCH --output=slurm_logs/finetune_%A_%a.out # Standard output log, %A is the main task ID, and %a is the index of this task in the array.
#SBATCH --error=slurm_logs/finetune_%A_%a.err  # Error output log
#SBATCH --array=1-8  # Total number of tasks (26 * 3 * 3 = 234), and set the range of the job array

# --- parameter definition ---
# Define the list of parameters to iterate over
PERCENTILES=(0.25 0.5 0.75)
D_NEWS=(32 64 128)

# Calculate the length of each parameter list
NUM_PERCENTILES=${#PERCENTILES[@]}
NUM_D_NEWS=${#D_NEWS[@]}

# --- Calculate the current task's parameters from SLURM_ARRAY_TASK_ID ---
# This is a crucial step that maps a single integer index back to three separate parameters
task_id=${SLURM_ARRAY_TASK_ID}

# Calculate the index and value of d_new
d_new_idx=$((task_id % NUM_D_NEWS))
D_NEW=${D_NEWS[$d_new_idx]}

# Calculate the index and value of percentile
percentile_idx=$(((task_id / NUM_D_NEWS) % NUM_PERCENTILES))
PERCENTILE=${PERCENTILES[$percentile_idx]}



# --- Environment Setup ---
# Please modify according to your environment, such as loading conda or spack
echo "========================================================"
echo "Starting sbatch script..."
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "  --sae_selection_percentile: ${PERCENTILE}"
echo "  --d_new: ${D_NEW}"
echo "Running on host: $(hostname)"
echo "This task will train ALL layers (0-25) for one parameter combination."



# --- Print current task's parameters ---
echo "Current Parameters:"
echo "  --sae_layer: ${LAYER}"
echo "  --sae_selection_percentile: ${PERCENTILE}"
echo "  --d_new: ${D_NEW}"
echo "========================================================"

for layer in $(seq 0 25); do
    echo "--------------------------------------------------------"
    echo "Starting training for layer: ${layer}"
    echo "--------------------------------------------------------"

    python3 ./sae_kl_finetune/finetune_transcoder_Delta2.py \
        --sae_layer ${layer} \
        --sae_selection_percentile ${PERCENTILE} \
        --d_new ${D_NEW} \
        --train_data_path ./gen_dataset_SAE/sae_train.txt \
        --validation_data_path ./gen_dataset_SAE/sae_validation.txt \
        --output_dir /usr/project/xtmp/qc62/train_4/dnew_${D_NEW}_pctl_${PERCENTILE}

    # Check the exit status of the Python script, and terminate the entire job if it fails
    if [ $? -ne 0 ]; then
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "ERROR: Python script failed for layer ${layer}."
        echo "Aborting this job array task to prevent wasting more time."
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        exit 1 # Exit the script to stop training all subsequent layers
    fi
done
echo "========================================================"
echo "Task ${SLURM_ARRAY_TASK_ID} finished."
echo "========================================================"