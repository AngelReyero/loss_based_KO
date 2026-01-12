#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --output=log_inference%A_%a.out
#SBATCH --error=log_inference%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=5
#SBATCH --mem=8G
#SBATCH --partition="normal,parietal"
#SBATCH --array=0-749%10  # 3 settings × 5 models × 50 seeds = 750 jobs, 5 concurrent

# Define settings and models
settings=("adjacent" "spaced" "nongauss")
models=("lasso" "RF" "NN" "GB" "SL")

# Compute indices
n_settings=${#settings[@]}
n_models=${#models[@]}
n_seeds=50

setting_idx=$((SLURM_ARRAY_TASK_ID / (n_models * n_seeds)))
model_idx=$(( (SLURM_ARRAY_TASK_ID / n_seeds) % n_models ))
seed_idx=$((SLURM_ARRAY_TASK_ID % n_seeds))

setting=${settings[$setting_idx]}
model=${models[$model_idx]}
seed=$((seed_idx + 101))

echo "Running setting=${setting}, model=${model}, seed=${seed}"

python -m experiments.p_val_ridge \
    --seed $seed \
    --setting $setting \
    --model $model
