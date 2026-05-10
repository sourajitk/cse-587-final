#!/bin/bash
#SBATCH --job-name=reactiont5_kg
#SBATCH --partition=standard
#SBATCH --account=cdm8_cr_default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:a100:1
#SBATCH --time=48:00:00
#SBATCH --output=/storage/group/cdm8/default/Somtirtha/ReactionT5v2/task_forward/reactiont5_kg_%j.log
#SBATCH --error=/storage/group/cdm8/default/Somtirtha/ReactionT5v2/task_forward/reactiont5_kg_%j.err

PYTHON=/storage/group/cdm8/default/Somtirtha/miniconda3/envs/reactiont5_env/bin/python

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/storage/group/cdm8/default/Somtirtha/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false

echo "============================================================"
echo "KG-Conditioned ReactionT5 Training"
echo "============================================================"
echo "Job started on $(hostname) at $(date)"
$PYTHON --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /storage/group/cdm8/default/Somtirtha/ReactionT5v2/task_forward

$PYTHON -u finetune_kg_simple.py \
  --model_name_or_path='/storage/group/cdm8/default/Somtirtha/ReactionT5v2/reactiont5v2_forward_pretrained' \
  --kg_embeddings='kg_embeddings/' \
  --epochs=50 \
  --lr=2e-5 \
  --batch_size=2 \
  --input_max_length=400 \
  --target_max_length=300 \
  --train_data_path='../../metanetx/metanetx_processed/metanetx_train_clean.csv' \
  --valid_data_path='../../metanetx/metanetx_processed/metanetx_val_clean.csv' \
  --output_dir='metanetx_forward_kg_simple' \
  --evaluation_strategy='epoch' \
  --save_strategy='epoch' \
  --logging_strategy='epoch'

echo "============================================================"
echo "Finished at $(date)"
