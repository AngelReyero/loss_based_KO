#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --output=log_inference%A_%a.out
#SBATCH --error=log_inference%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=5
#SBATCH --mem=8G
#SBATCH --partition="normal,parietal"
#SBATCH --array=0-479%10  

# Define settings, models, imputers, ns
settings=("poly" "interact_sin" "interact_pairwise" "interact_oscillatory")
models=("GB" "SL")
imputers=('elasticnet' 'GB' 'SL')
ns=(1000 3000)

# Sizes
n_settings=${#settings[@]}
n_models=${#models[@]}
n_imputers=${#imputers[@]}
n_ns=${#ns[@]}
n_seeds=10

# Index computations
idx=$SLURM_ARRAY_TASK_ID

setting_idx=$(( idx / (n_models * n_imputers * n_ns * n_seeds) ))
model_idx=$(( (idx / (n_imputers * n_ns * n_seeds)) % n_models ))
imputer_idx=$(( (idx / (n_ns * n_seeds)) % n_imputers ))
n_idx=$(( (idx / n_seeds) % n_ns ))
seed_idx=$(( idx % n_seeds ))

# Extract parameters
setting=${settings[$setting_idx]}
model=${models[$model_idx]}
imputer=${imputers[$imputer_idx]}
n=${ns[$n_idx]}
seed=$((seed_idx + 1))

echo "Running setting=${setting}, model=${model}, imputer=${imputer}, n=${n}, seed=${seed}"

python -m experiments.p_values \
    --seed $seed \
    --setting $setting \
    --model $model \
    --imputer $imputer \
    --n $n
