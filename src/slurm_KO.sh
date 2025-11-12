#!/bin/bash

#SBATCH --job-name=inference
#SBATCH --output=log_inference%A_%a.out
#SBATCH --error=log_inference%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=5
#SBATCH --mem=8G
#SBATCH --partition="normal,parietal"
#SBATCH --array=[100-110]%5


# Command to run
python -m exp.fdr_control \
    --seeds $SLURM_ARRAY_TASK_ID
