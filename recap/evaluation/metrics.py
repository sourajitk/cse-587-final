"""
Evaluation metrics for reaction completion.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from collections import defaultdict

from recap.utils.chemistry import (
    validate_smiles,
    canonicalize_smiles,
    check_reaction_balance,
    compute_similarity,
)


def top_k_accuracy(
    predictions: List[List[str]],
    targets: List[str],
    k: int = 1,
) -> float:
    """
    Calculate top-k accuracy.
    
    Args:
        predictions: List of prediction lists (one per sample)
        targets: List of target SMILES
        k: Number of top predictions to consider
        
    Returns:
        Top-k accuracy (0-1)
    """
    correct = 0
    total = len(targets)
    
    for preds, target in zip(predictions, targets):
        # Canonicalize for fair comparison
        target_canonical = canonicalize_smiles(target)
        pred_canonical = [canonicalize_smiles(p) for p in preds[:k]]
        
        if target_canonical in pred_canonical:
            correct += 1
    
    return correct / total if total > 0 else 0.0


def smiles_validity_rate(predictions: List[str]) -> float:
    """
    Calculate percentage of valid SMILES predictions.
    
    Args:
        predictions: List of predicted SMILES
        
    Returns:
        Validity rate (0-1)
    """
    if not predictions:
        return 0.0
    
    valid_count = sum(1 for p in predictions if validate_smiles(p))
    return valid_count / len(predictions)


def reaction_balance_rate(
    predictions: List[str],
    substrates: List[List[str]],
) -> float:
    """
    Calculate percentage of atom-balanced reactions.
    
    Args:
        predictions: List of predicted product SMILES
        substrates: List of substrate SMILES lists
        
    Returns:
        Balance rate (0-1)
    """
    if not predictions:
        return 0.0
    
    balanced_count = 0
    
    for pred, subs in zip(predictions, substrates):
        products = [p.strip() for p in pred.split(".") if p.strip()]
        is_balanced, _ = check_reaction_balance(subs, products)
        if is_balanced:
            balanced_count += 1
    
    return balanced_count / len(predictions)


def exact_match_accuracy(
    predictions: List[str],
    targets: List[str],
) -> float:
    """
    Calculate exact match accuracy (canonical SMILES).
    
    Args:
        predictions: List of predicted SMILES
        targets: List of target SMILES
        
    Returns:
        Exact match accuracy (0-1)
    """
    if not predictions:
        return 0.0
    
    matches = 0
    
    for pred, target in zip(predictions, targets):
        pred_canonical = canonicalize_smiles(pred)
        target_canonical = canonicalize_smiles(target)
        
        if pred_canonical == target_canonical:
            matches += 1
    
    return matches / len(predictions)


def max_similarity(
    predictions: List[List[str]],
    targets: List[str],
) -> Tuple[float, float]:
    """
    Calculate maximum similarity between predictions and targets.
    
    Args:
        predictions: List of prediction lists
        targets: List of target SMILES
        
    Returns:
        Tuple of (mean_max_similarity, std_max_similarity)
    """
    max_sims = []
    
    for preds, target in zip(predictions, targets):
        sims = [compute_similarity(p, target) for p in preds]
        max_sims.append(max(sims) if sims else 0.0)
    
    return np.mean(max_sims), np.std(max_sims)


class ReactionEvaluator:
    """
    Comprehensive evaluator for reaction completion models.
    """
    
    def __init__(self, k_values: List[int] = [1, 3, 5, 10]):
        """
        Args:
            k_values: List of k values for top-k accuracy
        """
        self.k_values = k_values
        self.results = defaultdict(list)
    
    def evaluate(
        self,
        predictions: List[List[str]],
        targets: List[str],
        substrates: Optional[List[List[str]]] = None,
    ) -> Dict[str, float]:
        """
        Run full evaluation.
        
        Args:
            predictions: List of prediction lists
            targets: List of target SMILES
            substrates: Optional list of substrate lists (for balance check)
            
        Returns:
            Dictionary of metric names to values
        """
        results = {}
        
        # Top-k accuracy
        for k in self.k_values:
            results[f"top{k}_accuracy"] = top_k_accuracy(predictions, targets, k)
        
        # Flatten predictions for other metrics
        top1_preds = [p[0] if p else "" for p in predictions]
        
        # Validity
        results["smiles_validity"] = smiles_validity_rate(top1_preds)
        
        # Exact match
        results["exact_match"] = exact_match_accuracy(top1_preds, targets)
        
        # Similarity
        mean_sim, std_sim = max_similarity(predictions, targets)
        results["max_similarity_mean"] = mean_sim
        results["max_similarity_std"] = std_sim
        
        # Reaction balance
        if substrates:
            results["reaction_balance"] = reaction_balance_rate(top1_preds, substrates)
        
        return results
    
    def evaluate_by_ec(
        self,
        predictions: List[List[str]],
        targets: List[str],
        ec_numbers: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate stratified by EC number class.
        
        Args:
            predictions: List of prediction lists
            targets: List of target SMILES
            ec_numbers: List of EC numbers
            
        Returns:
            Dictionary of EC class to metrics
        """
        # Group by EC class (first digit)
        ec_groups = defaultdict(lambda: {"preds": [], "targets": []})
        
        for pred, target, ec in zip(predictions, targets, ec_numbers):
            ec_class = ec.split(".")[0] if ec else "unknown"
            ec_groups[ec_class]["preds"].append(pred)
            ec_groups[ec_class]["targets"].append(target)
        
        # Evaluate each group
        results = {}
        for ec_class, data in ec_groups.items():
            results[ec_class] = self.evaluate(data["preds"], data["targets"])
        
        return results


def evaluate_model(
    checkpoint_path: str,
    test_data_path: str,
    batch_size: int = 32,
) -> Dict[str, float]:
    """
    Evaluate a trained model on test data.
    
    Args:
        checkpoint_path: Path to model checkpoint
        test_data_path: Path to test data CSV
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary of metrics
    """
    import pandas as pd
    from tqdm import tqdm
    from recap import ReactionCompleter
    
    # Load model
    completer = ReactionCompleter.from_pretrained(checkpoint_path)
    
    # Load test data
    df = pd.read_csv(test_data_path)
    
    # Generate predictions
    all_predictions = []
    all_targets = []
    all_substrates = []
    
    for i in tqdm(range(0, len(df), batch_size), desc="Evaluating"):
        batch = df.iloc[i:i + batch_size]
        
        for _, row in batch.iterrows():
            substrates = row["substrates"].split(".")
            target = row["products"]
            
            result = completer.complete(
                substrates=substrates,
                mode="forward",
                num_predictions=10,
            )
            
            all_predictions.append(result.predictions)
            all_targets.append(target)
            all_substrates.append(substrates)
    
    # Evaluate
    evaluator = ReactionEvaluator()
    metrics = evaluator.evaluate(
        predictions=all_predictions,
        targets=all_targets,
        substrates=all_substrates,
    )
    
    return metrics
