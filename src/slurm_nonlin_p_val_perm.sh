#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --output=log_inference%A_%a.out
#SBATCH --error=log_inference%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=5
#SBATCH --mem=8G
#SBATCH --partition="normal,parietal"
#SBATCH --array=0-199%10  # 4 settings × 5 models × 10 seeds = 200 jobs, 10 concurrent

# Define settings and models
settings=('masked_corr' 'single_index_threshold' 'cond_var' 'label_noise_gate')
models=("lasso" "RF" "NN" "GB" "SL")

# Compute indices
n_settings=${#settings[@]}
n_models=${#models[@]}
n_seeds=10

setting_idx=$((SLURM_ARRAY_TASK_ID / (n_models * n_seeds)))
model_idx=$(( (SLURM_ARRAY_TASK_ID / n_seeds) % n_models ))
seed_idx=$((SLURM_ARRAY_TASK_ID % n_seeds))

setting=${settings[$setting_idx]}
model=${models[$model_idx]}
seed=$((seed_idx + 1))

echo "Running setting=${setting}, model=${model}, seed=${seed}"

python -m experiments.non_lin_p_val_perm \
    --seed $seed \
    --setting $setting \
    --model $model
