#!/bin/bash
#SBATCH --job-name=longt5_mnx
#SBATCH --partition=standard
#SBATCH --account=cdm8_cr_default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=/storage/group/cdm8/default/Somtirtha/ReactionT5v2/task_forward/longt5_mnx_%j.log
#SBATCH --error=/storage/group/cdm8/default/Somtirtha/ReactionT5v2/task_forward/longt5_mnx_%j.err

PYTHON=/storage/group/cdm8/default/Somtirtha/miniconda3/envs/reactiont5_env/bin/python

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/storage/group/cdm8/default/Somtirtha/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_DISABLED=true

echo "Job started on $(hostname) at $(date)"
$PYTHON --version

cd /storage/group/cdm8/default/Somtirtha/ReactionT5v2/task_forward

$PYTHON -u finetune.py \
  --model_name_or_path='/storage/group/cdm8/default/Somtirtha/longt5_from_reactiont5' \
  --epochs=50 \
  --lr=2e-5 \
  --batch_size=2 \
  --input_max_length=1024 \
  --target_max_length=512 \
  --disable_tqdm \
  --train_data_path='../../metanetx/metanetx_processed/metanetx_train_clean.csv' \
  --valid_data_path='../../metanetx/metanetx_processed/metanetx_val_clean.csv' \
  --output_dir='metanetx_longt5_finetuned' \
  --evaluation_strategy='epoch' \
  --save_strategy='epoch' \
  --logging_strategy='epoch'

echo "Finished at $(date)"
