"""
Weight Transfer: ReactionT5v2 → LongT5 (TGlobal)
==================================================
Transfers chemistry-aware weights from ReactionT5v2 into a LongT5
architecture with TGlobal attention for handling longer cofactor SMILES.

Usage:
    python weight_transfer.py

Output:
    /storage/group/cdm8/default/Somtirtha/longt5_from_reactiont5/
"""

from transformers import (
    LongT5ForConditionalGeneration,
    LongT5Config,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
import torch

# ── Paths ─────────────────────────────────────────────────────────────────────
REACTION_T5_PATH = (
    "/storage/group/cdm8/default/Somtirtha/ReactionT5v2/"
    "reactiont5v2_forward_pretrained"
)
SAVE_PATH = "/storage/group/cdm8/default/Somtirtha/longt5_from_reactiont5"

# ── Step 1: Load ReactionT5 ───────────────────────────────────────────────────
print("=" * 60)
print("Step 1: Loading ReactionT5v2...")
print("=" * 60)

reactiont5 = AutoModelForSeq2SeqLM.from_pretrained(REACTION_T5_PATH)
tokenizer  = AutoTokenizer.from_pretrained(REACTION_T5_PATH)

print(f"  vocab_size (config):    {reactiont5.config.vocab_size}")
print(f"  vocab_size (tokenizer): {len(tokenizer)}")
print(f"  d_model:                {reactiont5.config.d_model}")
print(f"  d_kv:                   {reactiont5.config.d_kv}")
print(f"  d_ff:                   {reactiont5.config.d_ff}")
print(f"  num_heads:              {reactiont5.config.num_heads}")
print(f"  num_layers:             {reactiont5.config.num_layers}")
print(f"  dense_act_fn:           {reactiont5.config.dense_act_fn}")

# ── Step 2: Build LongT5 Config ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 2: Building LongT5 config matching ReactionT5 dimensions...")
print("=" * 60)

# dense_act_fn='gelu_new' forces T5DenseGatedActDense (wi_0, wi_1, wo)
# matching ReactionT5's gated FFN structure exactly
config = LongT5Config(
    vocab_size=reactiont5.config.vocab_size,         # 268
    d_model=reactiont5.config.d_model,               # 768
    d_kv=reactiont5.config.d_kv,                     # 64
    d_ff=reactiont5.config.d_ff,                     # 2048
    num_heads=reactiont5.config.num_heads,            # 12
    num_layers=reactiont5.config.num_layers,          # 12
    num_decoder_layers=reactiont5.config.num_decoder_layers,  # 12
    dense_act_fn="gelu_new",                         # activation function
    is_gated_act=True,                               # ← enables wi_0/wi_1 gated FFN
    attention_type="transient-global",               # TGlobal attention
    local_radius=127,                                # local window size
    global_block_size=16,                            # global token block size
    decoder_start_token_id=reactiont5.config.decoder_start_token_id,
    eos_token_id=reactiont5.config.eos_token_id,
    pad_token_id=reactiont5.config.pad_token_id,
    tie_word_embeddings=reactiont5.config.tie_word_embeddings,
)

longt5 = LongT5ForConditionalGeneration(config)

# Verify gated FFN was applied correctly
ffn_type = type(longt5.encoder.block[0].layer[1].DenseReluDense).__name__
print(f"  LongT5 FFN type: {ffn_type}")
assert "Gated" in ffn_type or "Gated" in ffn_type or ffn_type in (
    "LongT5DenseGatedActDense", "T5DenseGatedActDense"
), (
    f"ERROR: Expected gated FFN but got {ffn_type}. "
    "Try setting is_gated_act=True in LongT5Config."
)
print(f"  Gated FFN confirmed ✓ ({ffn_type})")

# ── Step 3: Transfer Weights ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 3: Transferring weights...")
print("=" * 60)

rt5_state = reactiont5.state_dict()
lt5_state = longt5.state_dict()

transferred = []  # direct name match
remapped    = []  # manual key remapping (attention name difference)
skipped     = []  # new TGlobal-specific params (stay random)

for lt5_key in lt5_state:

    # ── Direct match ──────────────────────────────────────────────
    if lt5_key in rt5_state:
        if lt5_state[lt5_key].shape == rt5_state[lt5_key].shape:
            lt5_state[lt5_key] = rt5_state[lt5_key].clone()
            transferred.append(lt5_key)
            continue
        else:
            skipped.append(
                f"{lt5_key}: shape mismatch "
                f"lt5={lt5_state[lt5_key].shape} "
                f"rt5={rt5_state[lt5_key].shape}"
            )
            continue

    # ── Encoder attention name remapping ──────────────────────────
    # LongT5 uses LocalSelfAttention, ReactionT5 uses SelfAttention
    # The weights are identical in structure — just different class names
    if "LocalSelfAttention" in lt5_key:
        rt5_key = lt5_key.replace("LocalSelfAttention", "SelfAttention")
        if rt5_key in rt5_state:
            if lt5_state[lt5_key].shape == rt5_state[rt5_key].shape:
                lt5_state[lt5_key] = rt5_state[rt5_key].clone()
                remapped.append(f"  {lt5_key}\n    ← {rt5_key}")
                continue
            else:
                skipped.append(
                    f"{lt5_key}: shape mismatch after remap "
                    f"lt5={lt5_state[lt5_key].shape} "
                    f"rt5={rt5_state[rt5_key].shape}"
                )
                continue

    # ── Not found — new TGlobal-specific param ────────────────────
    skipped.append(f"{lt5_key}: not in ReactionT5 (new TGlobal param)")

longt5.load_state_dict(lt5_state)

total = len(transferred) + len(remapped) + len(skipped)
pct   = 100 * (len(transferred) + len(remapped)) / total

print(f"  Direct transfer:   {len(transferred):3d} tensors")
print(f"  Remapped:          {len(remapped):3d} tensors  "
      f"(LocalSelfAttention ← SelfAttention)")
print(f"  Skipped (random):  {len(skipped):3d} tensors  "
      f"(new TGlobal params)")
print(f"  Total:             {total:3d} tensors")
print(f"  Transfer rate:     {pct:.1f}%")

if skipped:
    print(f"\n  Skipped layers (first 10):")
    for s in skipped[:10]:
        print(f"    {s}")
    if len(skipped) > 10:
        print(f"    ... and {len(skipped) - 10} more")

# ── Step 4: Sanity Test ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 4: Running sanity test...")
print("=" * 60)

test_inputs = [
    "REACTANT:CC(=O)O.OCC REAGENT:",
    "REACTANT:CC(=O)SCCNC(=O)CCNC(=O) REAGENT:",
]

longt5.eval()
for inp in test_inputs:
    tokens = tokenizer(
        inp,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
    )
    with torch.no_grad():
        out = longt5.generate(
            **tokens,
            max_new_tokens=80,
            num_beams=1,
        )
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"  Input:  {inp[:50]}...")
    print(f"  Output: '{decoded}'")
    print()

# Check output is non-empty (basic validity check)
tokens = tokenizer(
    "REACTANT:CC(=O)O.OCC REAGENT:",
    return_tensors="pt",
    max_length=1024,
    truncation=True,
)
with torch.no_grad():
    out = longt5.generate(**tokens, max_new_tokens=80, num_beams=1)
decoded = tokenizer.decode(out[0], skip_special_tokens=True)

if decoded and len(decoded) > 0:
    print("  Sanity check PASSED ✓ — model produces non-empty output")
else:
    print("  WARNING: model produces empty output — check transfer")

# ── Step 5: Save ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"Step 5: Saving to {SAVE_PATH}...")
print("=" * 60)

longt5.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

print("  Saved files:")
import os
for f in sorted(os.listdir(SAVE_PATH)):
    size = os.path.getsize(os.path.join(SAVE_PATH, f))
    print(f"    {f}  ({size/1e6:.1f} MB)")

print("\n" + "=" * 60)
print("Transfer complete!")
print(f"Hybrid model saved to: {SAVE_PATH}")
print()
print("Next step — fine-tune on MetaNetX:")
print(f"""
python -u finetune.py \\
  --model_name_or_path='{SAVE_PATH}' \\
  --epochs=50 \\
  --lr=2e-5 \\
  --batch_size=2 \\
  --input_max_length=1024 \\
  --target_max_length=512 \\
  --disable_tqdm \\
  --train_data_path='../../metanetx/metanetx_processed/metanetx_train_clean.csv' \\
  --valid_data_path='../../metanetx/metanetx_processed/metanetx_val_clean.csv' \\
  --output_dir='metanetx_longt5_finetuned' \\
  --evaluation_strategy='epoch' \\
  --save_strategy='epoch' \\
  --logging_strategy='epoch'
""")
print("=" * 60)
