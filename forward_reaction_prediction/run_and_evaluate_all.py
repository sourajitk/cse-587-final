#!/usr/bin/env python
"""
run_and_evaluate_all.py

Combined prediction and evaluation script for RECAP models.
Fixed for: REACTANT/PRODUCT columns, checkpoint paths, model loading.
"""

import argparse
import gc
import os
import sys
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration

warnings.filterwarnings("ignore")

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Configuration
# =============================================================================

MODEL_CONFIGS = {
    'baseline': {
        'path': 'sagawa/ReactionT5v2-forward',
        'description': 'Baseline (ReactionT5v2)',
    },
    'finetuned': {
        'path': './metanetx_forward_finetuned/checkpoint-153690',
        'description': 'MetaNetX Finetuned',
    },
    'kg_simple': {
        'path': './metanetx_forward_kg_simple/checkpoint-127530-fixed',
        'description': 'MetaNetX + KG Simple',
    },
    'longt5': {
        'path': './metanetx_longt5_finetuned/best_model',
        'description': 'LongT5 Finetuned',
    }
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--input_col", type=str, default="REACTANT")
    parser.add_argument("--target_col", type=str, default="PRODUCT")
    parser.add_argument("--models", type=str, nargs='+',
                        default=['baseline', 'finetuned', 'kg_simple', 'longt5'])
    parser.add_argument("--input_max_length", type=int, default=400)
    parser.add_argument("--output_max_length", type=int, default=300)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--num_return_sequences", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_prediction", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


# =============================================================================
# Dataset
# =============================================================================

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, input_col, max_length):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.input_col = input_col
        self.max_length = max_length
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        input_text = str(self.df.iloc[idx][self.input_col])
        encoded = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
        }


# =============================================================================
# Model Loading
# =============================================================================

def load_model(model_name: str, device: torch.device):
    """Load model and tokenizer."""
    
    model_path = MODEL_CONFIGS[model_name]['path']
    print(f"  Loading from: {model_path}")
    
    # Check path exists for local models
    if model_name != 'baseline' and not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    if model_name == 'baseline':
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    elif model_name == 'longt5':
        # LongT5 needs special handling
        try:
            from transformers import LongT5ForConditionalGeneration
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = LongT5ForConditionalGeneration.from_pretrained(model_path)
        except Exception as e:
            print(f"  LongT5 specific load failed: {e}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    else:
        # Local T5 checkpoints (finetuned, kg_simple)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
    
    model = model.to(device)
    model.eval()
    print(f"  Loaded successfully!")
    return model, tokenizer


# =============================================================================
# Prediction
# =============================================================================

def run_predictions(model, tokenizer, df, input_col, device, args) -> pd.DataFrame:
    """Run predictions."""
    
    dataset = SimpleDataset(df, tokenizer, input_col, args.input_max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    all_predictions = [[] for _ in range(args.num_return_sequences)]
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=args.output_max_length,
                num_beams=args.num_beams,
                num_return_sequences=args.num_return_sequences,
                early_stopping=True
            )
            
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            bs = input_ids.shape[0]
            for i in range(bs):
                for j in range(args.num_return_sequences):
                    pred = decoded[i * args.num_return_sequences + j].replace(" ", "")
                    all_predictions[j].append(pred)
    
    results = df[[input_col]].copy()
    for i in range(args.num_return_sequences):
        results[f'{i}th'] = all_predictions[i]
    
    return results


# =============================================================================
# Evaluation Functions
# =============================================================================

def canonicalize_smiles(smiles: str) -> Optional[str]:
    if not smiles or pd.isna(smiles):
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles).strip().replace(" ", ""))
        if mol:
            return Chem.MolToSmiles(mol, isomericSmiles=True)
    except:
        pass
    return None


def parse_product_set(smiles_str: str) -> Set[str]:
    if not smiles_str or pd.isna(smiles_str):
        return set()
    products = set()
    for smi in str(smiles_str).split('.'):
        canon = canonicalize_smiles(smi.strip())
        if canon:
            products.add(canon)
    return products


def get_atom_counts(smiles_str: str) -> Counter:
    atom_counts = Counter()
    if not smiles_str or pd.isna(smiles_str):
        return atom_counts
    for smi in str(smiles_str).split('.'):
        try:
            mol = Chem.MolFromSmiles(smi.strip())
            if mol:
                mol = Chem.AddHs(mol)
                for atom in mol.GetAtoms():
                    atom_counts[atom.GetSymbol()] += 1
        except:
            pass
    return atom_counts


def compute_tanimoto(pred_smiles: str, target_smiles: str) -> float:
    """Compute Tanimoto similarity with greedy matching."""
    pred_list = [s.strip() for s in str(pred_smiles).split('.') if s.strip()]
    target_list = [s.strip() for s in str(target_smiles).split('.') if s.strip()]
    
    if not pred_list or not target_list:
        return 0.0
    
    # Get fingerprints
    pred_fps = []
    for smi in pred_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                pred_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
        except:
            pass
    
    target_fps = []
    for smi in target_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                target_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
        except:
            pass
    
    if not pred_fps or not target_fps:
        return 0.0
    
    # Greedy matching
    sims = []
    used = set()
    for t_fp in target_fps:
        best_sim = 0.0
        best_idx = -1
        for i, p_fp in enumerate(pred_fps):
            if i in used:
                continue
            sim = DataStructs.TanimotoSimilarity(t_fp, p_fp)
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        sims.append(best_sim)
        if best_idx >= 0:
            used.add(best_idx)
    
    return np.mean(sims) if sims else 0.0


def is_valid_smiles(smiles_str: str) -> bool:
    if not smiles_str or pd.isna(smiles_str):
        return False
    for smi in str(smiles_str).split('.'):
        try:
            if Chem.MolFromSmiles(smi.strip()) is None:
                return False
        except:
            return False
    return True


def check_balance(reactant: str, product: str) -> Dict:
    """Check stoichiometric balance."""
    r_atoms = get_atom_counts(reactant)
    p_atoms = get_atom_counts(product)
    
    all_elems = set(r_atoms.keys()) | set(p_atoms.keys())
    delta = {}
    elem_balanced = {}
    
    for elem in all_elems:
        r = r_atoms.get(elem, 0)
        p = p_atoms.get(elem, 0)
        delta[elem] = p - r
        elem_balanced[elem] = (r == p)
    
    return {
        'balanced': all(d == 0 for d in delta.values()),
        'delta': delta,
        'elem_balanced': elem_balanced
    }


def evaluate_single(reactant: str, prediction: str, target: str) -> Dict:
    """Evaluate single prediction."""
    
    result = {
        'valid': False,
        'all_products': False,
        'any_product': False,
        'primary_product': False,
        'tanimoto': 0.0,
        'balanced': False,
        'missing': 0,
        'hallucinated': 0,
    }
    
    result['valid'] = is_valid_smiles(prediction)
    if not result['valid']:
        return result
    
    pred_set = parse_product_set(prediction)
    target_set = parse_product_set(target)
    
    if not target_set:
        return result
    
    # Accuracy metrics
    result['all_products'] = (pred_set == target_set)
    result['any_product'] = len(pred_set & target_set) > 0
    result['missing'] = len(target_set - pred_set)
    result['hallucinated'] = len(pred_set - target_set)
    
    # Primary product (largest by heavy atoms)
    def get_primary(pset):
        best, max_heavy = None, -1
        for s in pset:
            mol = Chem.MolFromSmiles(s)
            if mol and mol.GetNumHeavyAtoms() > max_heavy:
                max_heavy = mol.GetNumHeavyAtoms()
                best = s
        return best
    
    pred_primary = get_primary(pred_set)
    target_primary = get_primary(target_set)
    result['primary_product'] = (pred_primary == target_primary) if (pred_primary and target_primary) else False
    
    # Tanimoto
    result['tanimoto'] = compute_tanimoto(prediction, target)
    
    # Balance
    balance = check_balance(reactant, prediction)
    result['balanced'] = balance['balanced']
    result['elem_balanced'] = balance['elem_balanced']
    
    return result


def evaluate_beams(reactant: str, predictions: List[str], target: str) -> Dict:
    """Evaluate beam predictions."""
    
    results = [evaluate_single(reactant, p, target) for p in predictions if p]
    
    if not results:
        return evaluate_single(reactant, "", target)
    
    return {
        # Top-1 metrics
        'top1_valid': results[0]['valid'],
        'top1_all_products': results[0]['all_products'],
        'top1_any_product': results[0]['any_product'],
        'top1_primary_product': results[0]['primary_product'],
        'top1_tanimoto': results[0]['tanimoto'],
        'top1_balanced': results[0]['balanced'],
        'top1_missing': results[0]['missing'],
        'top1_hallucinated': results[0]['hallucinated'],
        
        # Top-K
        'top3_correct': any(r['all_products'] for r in results[:3]),
        'top5_correct': any(r['all_products'] for r in results[:5]),
        
        # Best beam
        'best_all_products': any(r['all_products'] for r in results),
        'best_tanimoto': max(r['tanimoto'] for r in results),
        'best_balanced': any(r['balanced'] for r in results),
        
        # Element balance details
        'elem_balanced': results[0].get('elem_balanced', {}),
    }


def evaluate_all(pred_df: pd.DataFrame, target_df: pd.DataFrame, 
                 input_col: str, target_col: str, num_beams: int) -> Tuple[pd.DataFrame, Dict]:
    """Evaluate all predictions."""
    
    pred_cols = [f'{i}th' for i in range(num_beams)]
    
    df = pred_df.copy()
    df['target'] = target_df[target_col].values
    df['reactant'] = target_df[input_col].values
    
    all_results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Evaluating"):
        preds = [str(row[c]) if c in row and pd.notna(row[c]) else "" for c in pred_cols]
        result = evaluate_beams(str(row['reactant']), preds, str(row['target']))
        result['idx'] = idx
        all_results.append(result)
    
    results_df = pd.DataFrame(all_results)
    n = len(results_df)
    
    # Aggregate
    metrics = {
        'all_products_accuracy': results_df['top1_all_products'].mean() * 100,
        'any_product_accuracy': results_df['top1_any_product'].mean() * 100,
        'primary_product_accuracy': results_df['top1_primary_product'].mean() * 100,
        'validity_rate': results_df['top1_valid'].mean() * 100,
        'invalidity_rate': (1 - results_df['top1_valid'].mean()) * 100,
        'mean_tanimoto': results_df['top1_tanimoto'].mean(),
        'stoichiometric_balance': results_df['top1_balanced'].mean() * 100,
        'top1_accuracy': results_df['top1_all_products'].mean() * 100,
        'top3_accuracy': results_df['top3_correct'].mean() * 100,
        'top5_accuracy': results_df['top5_correct'].mean() * 100,
        'best_beam_accuracy': results_df['best_all_products'].mean() * 100,
        'best_beam_tanimoto': results_df['best_tanimoto'].mean(),
        'avg_missing': results_df['top1_missing'].mean(),
        'avg_hallucinated': results_df['top1_hallucinated'].mean(),
        'n_samples': n,
    }
    
    # Element balance rates
    elem_stats = defaultdict(lambda: {'balanced': 0, 'total': 0})
    for _, row in results_df.iterrows():
        elem_bal = row.get('elem_balanced', None)
        if row['top1_valid'] and isinstance(elem_bal, dict):
            for elem, is_bal in elem_bal.items():
                elem_stats[elem]['total'] += 1
                if is_bal:
                    elem_stats[elem]['balanced'] += 1
    
    metrics['element_balance'] = {
        e: s['balanced']/s['total']*100 if s['total'] > 0 else 0 
        for e, s in elem_stats.items()
    }
    
    return results_df, metrics


def print_results(metrics: Dict, name: str):
    """Print formatted results."""
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {name}")
    print("=" * 60)
    
    print("\n=? PRIMARY METRICS")
    print("-" * 40)
    print(f"  All-Products Accuracy:    {metrics['all_products_accuracy']:6.2f}%")
    print(f"  Any-Product Accuracy:     {metrics['any_product_accuracy']:6.2f}%")
    print(f"  Primary-Product Accuracy: {metrics['primary_product_accuracy']:6.2f}%")
    print(f"  Validity Rate:            {metrics['validity_rate']:6.2f}%")
    print(f"  Mean Tanimoto:            {metrics['mean_tanimoto']:6.4f}")
    print(f"  Stoichiometric Balance:   {metrics['stoichiometric_balance']:6.2f}%")
    
    print("\n=? TOP-K ACCURACY")
    print("-" * 40)
    print(f"  Top-1: {metrics['top1_accuracy']:6.2f}%")
    print(f"  Top-3: {metrics['top3_accuracy']:6.2f}%")
    print(f"  Top-5: {metrics['top5_accuracy']:6.2f}%")
    
    print("\n  ERRORS")
    print("-" * 40)
    print(f"  Invalidity:      {metrics['invalidity_rate']:6.2f}%")
    print(f"  Avg Missing:     {metrics['avg_missing']:6.2f}")
    print(f"  Avg Hallucinated:{metrics['avg_hallucinated']:6.2f}")
    
    if 'element_balance' in metrics and metrics['element_balance']:
        print("\n ELEMENT BALANCE")
        print("-" * 40)
        for elem in ['C', 'H', 'O', 'N', 'S', 'P']:
            if elem in metrics['element_balance']:
                print(f"  {elem}: {metrics['element_balance'][elem]:6.2f}%")


def plot_comparison(all_metrics: Dict, output_dir: str):
    """Generate comparison plots."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not available, skipping plots")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    models = list(all_metrics.keys())
    labels = [MODEL_CONFIGS.get(m, {}).get('description', m) for m in models]
    colors = ['#95a5a6', '#3498db', '#f39c12', '#2ecc71'][:len(models)]
    
    # 1. Main accuracy
    ax = axes[0, 0]
    metrics_list = ['all_products_accuracy', 'any_product_accuracy', 'primary_product_accuracy']
    x = np.arange(3)
    w = 0.2
    for i, m in enumerate(models):
        vals = [all_metrics[m][k] for k in metrics_list]
        ax.bar(x + i*w, vals, w, label=labels[i], color=colors[i])
    ax.set_xticks(x + w*1.5)
    ax.set_xticklabels(['All-Prod', 'Any-Prod', 'Primary'])
    ax.set_ylabel('%')
    ax.set_title('Product Accuracy')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 100)
    
    # 2. Validity & Balance
    ax = axes[0, 1]
    metrics_list = ['validity_rate', 'stoichiometric_balance']
    x = np.arange(2)
    for i, m in enumerate(models):
        vals = [all_metrics[m][k] for k in metrics_list]
        ax.bar(x + i*w, vals, w, color=colors[i])
    ax.set_xticks(x + w*1.5)
    ax.set_xticklabels(['Validity', 'Balance'])
    ax.set_ylabel('%')
    ax.set_title('Validity & Balance')
    ax.set_ylim(0, 100)
    
    # 3. Tanimoto
    ax = axes[0, 2]
    vals = [all_metrics[m]['mean_tanimoto'] for m in models]
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylabel('Tanimoto')
    ax.set_title('Mean Tanimoto')
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', rotation=15)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
    
    # 4. Top-K
    ax = axes[1, 0]
    metrics_list = ['top1_accuracy', 'top3_accuracy', 'top5_accuracy']
    x = np.arange(3)
    for i, m in enumerate(models):
        vals = [all_metrics[m][k] for k in metrics_list]
        ax.bar(x + i*w, vals, w, label=labels[i], color=colors[i])
    ax.set_xticks(x + w*1.5)
    ax.set_xticklabels(['Top-1', 'Top-3', 'Top-5'])
    ax.set_ylabel('%')
    ax.set_title('Top-K Accuracy')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 100)
    
    # 5. Errors
    ax = axes[1, 1]
    metrics_list = ['avg_missing', 'avg_hallucinated']
    x = np.arange(2)
    for i, m in enumerate(models):
        vals = [all_metrics[m][k] for k in metrics_list]
        ax.bar(x + i*w, vals, w, color=colors[i])
    ax.set_xticks(x + w*1.5)
    ax.set_xticklabels(['Missing', 'Hallucinated'])
    ax.set_ylabel('Count')
    ax.set_title('Avg Errors')
    
    # 6. Heatmap
    ax = axes[1, 2]
    hm_metrics = ['all_products_accuracy', 'any_product_accuracy', 'validity_rate', 'stoichiometric_balance']
    hm_data = [[all_metrics[m][k] for k in hm_metrics] for m in models]
    sns.heatmap(hm_data, annot=True, fmt='.1f', cmap='RdYlGn',
                xticklabels=['All', 'Any', 'Valid', 'Bal'],
                yticklabels=labels, ax=ax, vmin=0, vmax=100)
    ax.set_title('Summary')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to {output_dir}/comparison.png")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("RECAP: Prediction and Evaluation Pipeline")
    print("=" * 60)
    print(f"Start: {datetime.now()}")
    print(f"Device: {device}")
    print(f"Input: {args.input_data}")
    print(f"Columns: {args.input_col} -> {args.target_col}")
    print(f"Models: {args.models}")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(args.input_data)
    if args.debug:
        df = df.head(100)
    print(f"\nLoaded {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")
    
    all_metrics = {}
    
    for i, model_name in enumerate(args.models, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(args.models)}] {MODEL_CONFIGS[model_name]['description']}")
        print("=" * 60)
        
        model_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        pred_path = os.path.join(model_dir, 'predictions.csv')
        
        # Prediction
        if args.skip_prediction:
            print("  Skipping prediction")
        elif args.skip_existing and os.path.exists(pred_path):
            print(f"  Using existing predictions: {pred_path}")
        else:
            try:
                model, tokenizer = load_model(model_name, device)
                pred_df = run_predictions(model, tokenizer, df, args.input_col, device, args)
                pred_df.to_csv(pred_path, index=False)
                print(f"  Saved: {pred_path}")
                del model
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Evaluation
        if not os.path.exists(pred_path):
            print(f"  No predictions found, skipping evaluation")
            continue
        
        try:
            pred_df = pd.read_csv(pred_path)
            results_df, metrics = evaluate_all(pred_df, df, args.input_col, args.target_col, args.num_beams)
            
            all_metrics[model_name] = metrics
            
            results_df.to_csv(os.path.join(model_dir, 'detailed.csv'), index=False)
            pd.DataFrame([metrics]).to_csv(os.path.join(model_dir, 'summary.csv'), index=False)
            
            print_results(metrics, MODEL_CONFIGS[model_name]['description'])
            
        except Exception as e:
            print(f"  Evaluation ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Comparison
    if len(all_metrics) > 1:
        print(f"\n{'='*60}")
        print("COMPARISON")
        print("=" * 60)
        
        comp_df = pd.DataFrame({
            'Model': [MODEL_CONFIGS[m]['description'] for m in all_metrics],
            'All-Prod (%)': [all_metrics[m]['all_products_accuracy'] for m in all_metrics],
            'Any-Prod (%)': [all_metrics[m]['any_product_accuracy'] for m in all_metrics],
            'Valid (%)': [all_metrics[m]['validity_rate'] for m in all_metrics],
            'Tanimoto': [all_metrics[m]['mean_tanimoto'] for m in all_metrics],
            'Balance (%)': [all_metrics[m]['stoichiometric_balance'] for m in all_metrics],
            'Top-1 (%)': [all_metrics[m]['top1_accuracy'] for m in all_metrics],
            'Top-5 (%)': [all_metrics[m]['top5_accuracy'] for m in all_metrics],
        })
        print(comp_df.to_string(index=False))
        comp_df.to_csv(os.path.join(args.output_dir, 'comparison.csv'), index=False)
        
        plot_comparison(all_metrics, args.output_dir)
    
    print(f"\nEnd: {datetime.now()}")
    print(f"Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
