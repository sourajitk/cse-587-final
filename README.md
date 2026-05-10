# RECAP: Reaction Completion for Enzymatic Pathways

Fine-tuned ReactionT5v2 models for enzymatic reaction product prediction on MetaNetX.

## Models

| Model | Path | Description |
|-------|------|-------------|
| **Baseline** | `sagawa/ReactionT5v2-forward` | Pre-trained ReactionT5v2 |
| **Finetuned** | `./metanetx_forward_finetuned/checkpoint-153690` | MetaNetX fine-tuned |
| **KG-Simple** | `./metanetx_forward_kg_simple/checkpoint-127530-fixed` | + KG embeddings |
| **LongT5** | `./metanetx_longt5_finetuned/best_model` | LongT5-TGlobal |

## Setup

### Environment

```bash
conda create -n reactiont5_env python=3.10
conda activate reactiont5_env
pip install torch transformers datasets pandas numpy rdkit tqdm matplotlib seaborn
```

### Directory Structure

```
/storage/group/cdm8/default/Somtirtha/
├── ReactionT5v2/
│   ├── reactiont5v2_forward_pretrained/
│   └── task_forward/
│       ├── finetune.py
│       ├── weight_transfer.py
│       ├── fg_tokenizer.py
│       ├── fgkg_builder.py
│       ├── kg_embeddings.py
│       ├── kg_conditioned_model.py
│       ├── run_and_evaluate_all.py
│       ├── fgkg.pkl
│       ├── kg_embeddings/
│       ├── metanetx_forward_finetuned/
│       ├── metanetx_forward_kg_simple/
│       ├── metanetx_longt5_finetuned/
│       └── results/
├── longt5_from_reactiont5/
├── metanetx/
│   └── metanetx_processed/
│       ├── metanetx_train_clean.csv
│       ├── metanetx_val_clean.csv
│       └── metanetx_test_clean.csv
└── miniconda3/envs/reactiont5_env/
```

---

## 1. Fine-tune ReactionT5v2

### Script: `submit_finetune.sh`

```bash
#!/bin/bash
#SBATCH --job-name=reactiont5_mnx
#SBATCH --partition=standard
#SBATCH --account=cdm8_cr_default
#SBATCH --nodelist=p-ic-4012
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=/storage/group/cdm8/default/Somtirtha/ReactionT5v2/task_forward/reactiont5_mnx_%j.log
#SBATCH --error=/storage/group/cdm8/default/Somtirtha/ReactionT5v2/task_forward/reactiont5_mnx_%j.err

PYTHON=/storage/group/cdm8/default/Somtirtha/miniconda3/envs/reactiont5_env/bin/python

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/storage/group/cdm8/default/Somtirtha/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false

cd /storage/group/cdm8/default/Somtirtha/ReactionT5v2/task_forward

$PYTHON -u finetune.py \
  --model_name_or_path='/storage/group/cdm8/default/Somtirtha/ReactionT5v2/reactiont5v2_forward_pretrained' \
  --epochs=50 \
  --lr=2e-5 \
  --batch_size=2 \
  --input_max_length=400 \
  --target_max_length=300 \
  --train_data_path='../../metanetx/metanetx_processed/metanetx_train_clean.csv' \
  --valid_data_path='../../metanetx/metanetx_processed/metanetx_val_clean.csv' \
  --output_dir='metanetx_forward_finetuned' \
  --evaluation_strategy='epoch' \
  --save_strategy='epoch' \
  --logging_strategy='epoch'
```

---

## 2. LongT5 (Weight Transfer + Fine-tune)

### Weight Transfer

`weight_transfer.py` transfers ReactionT5v2 → LongT5-TGlobal:

- Matches ReactionT5 config (vocab=268, d_model=768, d_kv=64, d_ff=2048)
- Enables gated FFN (`is_gated_act=True`)
- Remaps `LocalSelfAttention ← SelfAttention`
- TGlobal: `local_radius=127`, `global_block_size=16`

```bash
python weight_transfer.py
# Output: /storage/group/cdm8/default/Somtirtha/longt5_from_reactiont5/
```

### Script: `submit_longt5.sh`

```bash
#!/bin/bash
#SBATCH --job-name=longt5_mnx
#SBATCH --partition=standard
#SBATCH --account=cdm8_cr_default
#SBATCH --nodelist=p-ic-4012
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=/storage/group/cdm8/default/Somtirtha/ReactionT5v2/task_forward/longt5_mnx_%j.log
#SBATCH --error=/storage/group/cdm8/default/Somtirtha/ReactionT5v2/task_forward/longt5_mnx_%j.err

PYTHON=/storage/group/cdm8/default/Somtirtha/miniconda3/envs/reactiont5_env/bin/python

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/storage/group/cdm8/default/Somtirtha/hf_cache
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false

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
```

---

## 3. FG-KG Conditioned Model

### Architecture

```
Substrates ──► fg_tokenizer.extract_fg_from_reaction()
                      │
                      ▼
            FGKG (transforms_to, co_consumed, co_formed)
                      │
                      ▼
            TransE Embeddings (128-d, margin loss)
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  ReactionT5v2 Encoder → Decoder + KGCrossAttention     │
│  (Q: decoder_hidden, K/V: kg_embeddings, 2 layers)     │
└─────────────────────────────────────────────────────────┘
                      │
                      ▼
                 LM Head → Products
```

### Script: `run_finetune_kg.sh`

```bash
#!/bin/bash
#SBATCH --job-name=kg_mnx
#SBATCH --partition=standard
#SBATCH --account=cdm8_cr_default
#SBATCH --nodelist=p-ic-4012
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=/storage/group/cdm8/default/Somtirtha/ReactionT5v2/task_forward/kg_mnx_%j.log
#SBATCH --error=/storage/group/cdm8/default/Somtirtha/ReactionT5v2/task_forward/kg_mnx_%j.err

PYTHON=/storage/group/cdm8/default/Somtirtha/miniconda3/envs/reactiont5_env/bin/python

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/storage/group/cdm8/default/Somtirtha/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false

cd /storage/group/cdm8/default/Somtirtha/ReactionT5v2/task_forward

# Step 1: Build FGKG
if [ ! -f "fgkg.pkl" ]; then
    $PYTHON -u fgkg_builder.py \
        -i '../../metanetx/metanetx_processed/metanetx_train_clean.csv' \
        -o 'fgkg.pkl' \
        --substrate-col 'substrates' \
        --product-col 'products'
fi

# Step 2: Train TransE embeddings
if [ ! -d "kg_embeddings" ]; then
    $PYTHON -u kg_embeddings.py \
        --fgkg 'fgkg.pkl' \
        -o 'kg_embeddings' \
        --dim 128 \
        --epochs 100 \
        --min-count 2
fi

# Step 3: Fine-tune KG model
$PYTHON -u finetune_kg.py \
  --base_model='/storage/group/cdm8/default/Somtirtha/ReactionT5v2/reactiont5v2_forward_pretrained' \
  --kg_embeddings='kg_embeddings' \
  --epochs=40 \
  --lr=2e-5 \
  --batch_size=2 \
  --n_kg_layers=2 \
  --train_data_path='../../metanetx/metanetx_processed/metanetx_train_clean.csv' \
  --valid_data_path='../../metanetx/metanetx_processed/metanetx_val_clean.csv' \
  --output_dir='metanetx_forward_kg_simple'
```

---

## 4. Evaluate All Models

### Usage

```bash
python run_and_evaluate_all.py \
  --input_data '../../metanetx/metanetx_processed/metanetx_test_clean.csv' \
  --input_col 'REACTANT' \
  --target_col 'PRODUCT' \
  --models baseline finetuned kg_simple longt5 \
  --num_beams 5 \
  --batch_size 16 \
  --output_dir './results'
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input_data` | required | Test CSV path |
| `--input_col` | `REACTANT` | Input column name |
| `--target_col` | `PRODUCT` | Target column name |
| `--models` | all 4 | Models to evaluate |
| `--num_beams` | 5 | Beam search width |
| `--batch_size` | 16 | Inference batch size |
| `--output_dir` | `./results` | Output directory |
| `--skip_prediction` | False | Skip prediction, use existing |
| `--skip_existing` | False | Skip if predictions exist |
| `--debug` | False | Run on first 100 samples |

### Script: `submit_evaluate.sh`

```bash
#!/bin/bash
#SBATCH --job-name=eval_mnx
#SBATCH --partition=standard
#SBATCH --account=cdm8_cr_default
#SBATCH --nodelist=p-ic-4012
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --output=/storage/group/cdm8/default/Somtirtha/ReactionT5v2/task_forward/eval_mnx_%j.log
#SBATCH --error=/storage/group/cdm8/default/Somtirtha/ReactionT5v2/task_forward/eval_mnx_%j.err

PYTHON=/storage/group/cdm8/default/Somtirtha/miniconda3/envs/reactiont5_env/bin/python

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/storage/group/cdm8/default/Somtirtha/hf_cache
export TRANSFORMERS_OFFLINE=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false

cd /storage/group/cdm8/default/Somtirtha/ReactionT5v2/task_forward

$PYTHON -u run_and_evaluate_all.py \
  --input_data '../../metanetx/metanetx_processed/metanetx_test_clean.csv' \
  --input_col 'REACTANT' \
  --target_col 'PRODUCT' \
  --models baseline finetuned kg_simple longt5 \
  --num_beams 5 \
  --batch_size 16 \
  --output_dir './results'
```

### Outputs

```
results/
├── comparison.csv          # Side-by-side metrics
├── comparison.png          # Visualization
├── baseline/
│   ├── predictions.csv     # Raw predictions (0th, 1th, ...)
│   ├── detailed.csv        # Per-sample metrics
│   └── summary.csv         # Aggregate metrics
├── finetuned/
│   └── ...
├── kg_simple/
│   └── ...
└── longt5/
    └── ...
```

---

## Run Order

```bash
# 1. Basic finetuning
sbatch submit_finetune.sh

# 2. Weight transfer (no SLURM)
python weight_transfer.py

# 3. LongT5 finetuning
sbatch submit_longt5.sh

# 4. FG-KG pipeline
sbatch run_finetune_kg.sh

# 5. Evaluate all
sbatch submit_evaluate.sh
```

---

## Metrics

| Metric | Description |
|--------|-------------|
| All-Products | All predicted products exactly match target |
| Any-Product | At least one product matches |
| Primary-Product | Largest product (by heavy atoms) matches |
| Validity | Chemically valid SMILES |
| Tanimoto | Greedy-matched fingerprint similarity |
| Stoichiometric Balance | Atom counts (C, H, O, N) match |
| Top-K | Correct in top K beam results |
| Missing | Avg target products not in prediction |
| Hallucinated | Avg predicted products not in target |

---

## Results

| Model | All-Prod | Any-Prod | Primary | Valid | Tanimoto | Balance | Top-5 |
|-------|----------|----------|---------|-------|----------|---------|-------|
| Baseline | 0.30% | 2.51% | 0.77% | 85.86% | 0.156 | 3.31% | 0.50% |
| Finetuned | 17.30% | 58.62% | 43.30% | 81.82% | 0.634 | 12.54% | 32.14% |
| KG-Simple | 16.79% | 58.32% | 43.06% | 81.46% | 0.631 | 12.54% | 31.88% |
| LongT5 | 15.53% | 56.29% | 40.88% | 79.78% | 0.610 | 9.50% | 28.78% |

**Key findings:**
- Fine-tuning: 57× improvement over baseline
- KG embeddings: negligible benefit (-0.5%)
- LongT5: worse than standard T5 (-1.8%), shorter context sufficient
- Stoichiometric balance: poor across all (best 12.5%)

---

## Files

| File | Description |
|------|-------------|
| `finetune.py` | ReactionT5v2 / LongT5 finetuning |
| `weight_transfer.py` | ReactionT5v2 → LongT5-TGlobal |
| `fg_tokenizer.py` | FG detection (FARM-based) |
| `fgkg_builder.py` | Build FGKG from reactions |
| `kg_embeddings.py` | TransE embedding training |
| `kg_conditioned_model.py` | ReactionT5 + KG cross-attention |
| `run_and_evaluate_all.py` | Combined prediction + evaluation |

---

## License

MIT
