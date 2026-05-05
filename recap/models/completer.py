"""
Main reaction completion interface.
"""

from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
import torch
from transformers import AutoTokenizer

from .reaction_t5 import ReactionT5Model, ReactionT5Config
from .kg import BiochemicalKG


@dataclass
class CompletionResult:
    """Result from reaction completion."""
    
    # Input
    input_reaction: str
    mode: str
    
    # Predictions
    predictions: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    
    # Parsed outputs
    substrates: Optional[List[str]] = None
    products: Optional[List[str]] = None
    cofactors: Optional[List[str]] = None
    
    # Metadata
    complete_reaction: Optional[str] = None
    confidence: float = 0.0
    is_valid: bool = True
    
    @property
    def top_prediction(self) -> str:
        """Get top prediction."""
        return self.predictions[0] if self.predictions else ""
    
    @property
    def top_score(self) -> float:
        """Get top score."""
        return self.scores[0] if self.scores else 0.0


class ReactionCompleter:
    """
    High-level interface for biochemical reaction completion.
    
    Supports multiple completion modes:
    - forward: Predict products from substrates
    - retro: Predict substrates from products (retrosynthesis)
    - fill: Fill in missing components in partial reactions
    
    Example:
        >>> completer = ReactionCompleter.from_pretrained("recap-base")
        >>> result = completer.complete(
        ...     substrates=["ATP", "glucose"],
        ...     mode="forward"
        ... )
        >>> print(result.products)
    """
    
    # Model registry
    MODEL_REGISTRY = {
        "recap-base": {
            "forward": "path/to/recap-base-forward",
            "retro": "path/to/recap-base-retro",
        },
        "recap-large": {
            "forward": "path/to/recap-large-forward",
            "retro": "path/to/recap-large-retro",
        },
    }
    
    def __init__(
        self,
        model: ReactionT5Model,
        tokenizer: AutoTokenizer,
        device: Optional[str] = None,
        kg: Optional[BiochemicalKG] = None,
    ):
        """
        Initialize completer.
        
        Args:
            model: ReactionT5 model
            tokenizer: Tokenizer
            device: Device to use (auto-detected if None)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.kg = kg
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "recap-base",
        mode: str = "forward",
        device: Optional[str] = None,
        kg_path: Optional[str] = None,
        **kwargs,
    ) -> "ReactionCompleter":
        """
        Load pretrained model.
        
        Args:
            model_name: Model identifier (recap-base, recap-large, or path)
            mode: Default completion mode
            device: Device to use
            **kwargs: Additional arguments
            
        Returns:
            Initialized completer
        """
        # Resolve model path
        if model_name in cls.MODEL_REGISTRY:
            model_path = cls.MODEL_REGISTRY[model_name].get(mode)
        else:
            model_path = model_name
        
        # Load model and tokenizer
        # For now, use ReactionT5v2 as base
        if model_path is None or model_path.startswith("path/"):
            model_path = "sagawa/ReactionT5v2-forward" if mode == "forward" else "sagawa/ReactionT5v2-retrosynthesis"
        
        model = ReactionT5Model.from_pretrained(model_path, mode=mode, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        kg = None
        if kg_path:
            kg = BiochemicalKG()
            kg.load_from_tsv(kg_path)
            
        return cls(model=model, tokenizer=tokenizer, device=device, kg=kg)
    
    def complete(
        self,
        substrates: Optional[List[str]] = None,
        products: Optional[List[str]] = None,
        partial_reaction: Optional[str] = None,
        mode: str = "forward",
        ec_number: Optional[str] = None,
        num_predictions: int = 5,
        **kwargs,
    ) -> CompletionResult:
        """
        Complete a reaction.
        
        Args:
            substrates: List of substrate SMILES (for forward mode)
            products: List of product SMILES (for retro mode)
            partial_reaction: Partial reaction string (for fill mode)
            mode: Completion mode (forward, retro, fill)
            ec_number: Optional EC number for context
            num_predictions: Number of predictions to return
            **kwargs: Additional generation arguments
            
        Returns:
            CompletionResult with predictions
        """
        # Format input based on mode
        input_str = self._format_input(
            substrates=substrates,
            products=products,
            partial_reaction=partial_reaction,
            mode=mode,
            ec_number=ec_number,
        )
        
        # Tokenize
        inputs = self.tokenizer(
            input_str,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                num_return_sequences=num_predictions,
                **kwargs,
            )
        
        # Decode predictions
        predictions = self.tokenizer.batch_decode(
            outputs, 
            skip_special_tokens=True
        )
        
        # Calculate confidence scores (placeholder - implement properly)
        scores = [1.0 / (i + 1) for i in range(len(predictions))]
        
        # Re-rank predictions using Knowledge Graph if available
        if self.kg is not None:
            # Extract actual SMILES part from input_str (e.g., 'REACTANT:A.B' -> 'A.B')
            input_smiles = input_str.split(':', 1)[-1].strip()
            # Remove EC number if present (e.g., 'EC:1.2.3.4 REACTANT:A.B' -> 'REACTANT:A.B')
            if ' ' in input_smiles and input_smiles.startswith('EC:'):
                input_smiles = input_smiles.split(' ', 1)[1]
                if ':' in input_smiles:
                    input_smiles = input_smiles.split(':', 1)[-1].strip()
            
            input_mols = self._parse_molecules(input_smiles)
            
            for i, pred in enumerate(predictions):
                pred_mols = self._parse_molecules(pred)
                if mode == "forward":
                    boost = self.kg.check_plausibility(input_mols, pred_mols)
                elif mode == "retro":
                    boost = self.kg.check_plausibility(pred_mols, input_mols)
                else:
                    boost = 0.0
                scores[i] += boost
                
            # Sort by new scores
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            predictions = [predictions[i] for i in sorted_indices]
            scores = [scores[i] for i in sorted_indices]
        
        # Parse result
        result = self._parse_result(
            input_str=input_str,
            predictions=predictions,
            scores=scores,
            mode=mode,
        )
        
        return result
    
    def _format_input(
        self,
        substrates: Optional[List[str]] = None,
        products: Optional[List[str]] = None,
        partial_reaction: Optional[str] = None,
        mode: str = "forward",
        ec_number: Optional[str] = None,
    ) -> str:
        """Format input string based on mode."""
        
        if mode == "forward":
            if substrates is None:
                raise ValueError("substrates required for forward mode")
            input_str = ".".join(substrates)
            prefix = "REACTANT:"
        
        elif mode == "retro":
            if products is None:
                raise ValueError("products required for retro mode")
            input_str = ".".join(products)
            prefix = "PRODUCT:"
        
        elif mode == "fill":
            if partial_reaction is None:
                raise ValueError("partial_reaction required for fill mode")
            input_str = partial_reaction
            prefix = "PARTIAL:"
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Add EC number context if provided
        if ec_number:
            input_str = f"EC:{ec_number} {input_str}"
        
        return f"{prefix}{input_str}"
    
    def _parse_result(
        self,
        input_str: str,
        predictions: List[str],
        scores: List[float],
        mode: str,
    ) -> CompletionResult:
        """Parse raw predictions into structured result."""
        
        result = CompletionResult(
            input_reaction=input_str,
            mode=mode,
            predictions=predictions,
            scores=scores,
        )
        
        # Parse top prediction
        if predictions:
            top_pred = predictions[0]
            
            if mode == "forward":
                result.products = self._parse_molecules(top_pred)
            elif mode == "retro":
                result.substrates = self._parse_molecules(top_pred)
            elif mode == "fill":
                result.complete_reaction = top_pred
            
            result.confidence = scores[0] if scores else 0.0
            result.is_valid = self._validate_smiles(top_pred)
        
        return result
    
    def _parse_molecules(self, smiles_str: str) -> List[str]:
        """Parse dot-separated SMILES string into list."""
        return [s.strip() for s in smiles_str.split(".") if s.strip()]
    
    def _validate_smiles(self, smiles: str) -> bool:
        """Validate SMILES string using RDKit."""
        try:
            from rdkit import Chem
            
            for s in smiles.split("."):
                mol = Chem.MolFromSmiles(s.strip())
                if mol is None:
                    return False
            return True
        except Exception:
            return False
    
    def batch_complete(
        self,
        inputs: List[Dict[str, Any]],
        mode: str = "forward",
        batch_size: int = 32,
        **kwargs,
    ) -> List[CompletionResult]:
        """
        Complete multiple reactions in batch.
        
        Args:
            inputs: List of input dicts with substrates/products
            mode: Completion mode
            batch_size: Batch size for processing
            **kwargs: Additional arguments
            
        Returns:
            List of CompletionResults
        """
        results = []
        
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            
            # Process each item (could be optimized for true batching)
            for item in batch:
                result = self.complete(mode=mode, **item, **kwargs)
                results.append(result)
        
        return results
    
    def score_reaction(
        self,
        substrates: List[str],
        products: List[str],
    ) -> float:
        """
        Score a complete reaction.
        
        Args:
            substrates: Substrate SMILES
            products: Product SMILES
            
        Returns:
            Plausibility score (0-1)
        """
        # Forward prediction
        forward_result = self.complete(substrates=substrates, mode="forward")
        
        # Check if actual products match predictions
        actual_products = set(products)
        
        score = 0.0
        for pred, pred_score in zip(forward_result.predictions, forward_result.scores):
            pred_products = set(self._parse_molecules(pred))
            overlap = len(actual_products & pred_products) / max(len(actual_products), 1)
            score = max(score, overlap * pred_score)
        
        return score
