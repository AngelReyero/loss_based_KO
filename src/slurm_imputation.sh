#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --output=log_inference%A_%a.out
#SBATCH --error=log_inference%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=5
#SBATCH --mem=8G
#SBATCH --partition="normal,parietal"
#SBATCH --array=0-479%10  

# Define settings and models
settings=("hidim" "cos")
models=("lasso" "SL")
imputers=('lasso' 'elasticnet' 'ridge' 'RF' 'GB' 'SL')

# Sizes
n_settings=${#settings[@]}
n_models=${#models[@]}
n_imputers=${#imputers[@]}
n_seeds=20

# Index computations
setting_idx=$(( SLURM_ARRAY_TASK_ID / (n_models * n_imputers * n_seeds) ))
model_idx=$(( (SLURM_ARRAY_TASK_ID / (n_imputers * n_seeds)) % n_models ))
imputer_idx=$(( (SLURM_ARRAY_TASK_ID / n_seeds) % n_imputers ))
seed_idx=$(( SLURM_ARRAY_TASK_ID % n_seeds ))

# Extract parameters
setting=${settings[$setting_idx]}
model=${models[$model_idx]}
imputer=${imputers[$imputer_idx]}
seed=$((seed_idx + 21))

echo "Running setting=${setting}, model=${model}, imputer=${imputer}, seed=${seed}"

python -m experiments.p_values \
    --seed $seed \
    --setting $setting \
    --model $model \
    --imputer $imputer
