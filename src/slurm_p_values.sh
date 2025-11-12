#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --output=log_inference%A_%a.out
#SBATCH --error=log_inference%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=5
#SBATCH --mem=8G
#SBATCH --partition="normal,parietal"
#SBATCH --array=0-599%10  # 6 settings × 5 models × 20 seeds = 600 jobs, 5 concurrent

# Define settings and models
settings=("adjacent" "spaced" "sinusoidal" "highdim" "nongaussian" "poly")
models=("lasso" "RF" "NN" "GB" "SL")

# Compute indices
n_settings=${#settings[@]}
n_models=${#models[@]}
n_seeds=20

setting_idx=$((SLURM_ARRAY_TASK_ID / (n_models * n_seeds)))
model_idx=$(( (SLURM_ARRAY_TASK_ID / n_seeds) % n_models ))
seed_idx=$((SLURM_ARRAY_TASK_ID % n_seeds))

setting=${settings[$setting_idx]}
model=${models[$model_idx]}
seed=$((seed_idx + 1))

echo "Running setting=${setting}, model=${model}, seed=${seed}"

python -m experiments.p_values \
    --seed $seed \
    --setting $setting \
    --model $model
