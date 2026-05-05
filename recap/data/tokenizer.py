"""
Tokenizer for biochemical reactions.
"""

from typing import Optional, List, Dict, Union
from pathlib import Path
import json
import re
from transformers import PreTrainedTokenizer, T5Tokenizer


class ReactionTokenizer:
    """
    Tokenizer wrapper for biochemical reaction SMILES.
    
    Extends base T5 tokenizer with:
    - Special tokens for reaction components
    - SMILES-aware tokenization
    - EC number encoding
    
    Args:
        base_tokenizer: Base tokenizer (T5)
        add_special_tokens: Whether to add reaction-specific tokens
    """
    
    SPECIAL_TOKENS = [
        "[REACTANT]",
        "[PRODUCT]",
        "[COFACTOR]",
        "[ENZYME]",
        "[EC]",
        "[PARTIAL]",
        "[MASK]",
        "[SEP]",
    ]
    
    # Common SMILES tokens
    SMILES_TOKENS = [
        "Cl", "Br", "Si", "Se", "se", "@@", "++", "--",
        "[C@@H]", "[C@H]", "[nH]", "[NH]", "[O-]", "[N+]",
    ]
    
    def __init__(
        self,
        base_tokenizer: Optional[PreTrainedTokenizer] = None,
        add_special_tokens: bool = True,
    ):
        if base_tokenizer is None:
            base_tokenizer = T5Tokenizer.from_pretrained(
                "sagawa/ReactionT5v2-forward"
            )
        
        self.tokenizer = base_tokenizer
        
        if add_special_tokens:
            self._add_special_tokens()
    
    def _add_special_tokens(self):
        """Add reaction-specific special tokens."""
        special_tokens = {
            "additional_special_tokens": self.SPECIAL_TOKENS
        }
        self.tokenizer.add_special_tokens(special_tokens)
    
    def __call__(
        self,
        text: Union[str, List[str]],
        **kwargs,
    ):
        """Tokenize input."""
        return self.tokenizer(text, **kwargs)
    
    def encode(self, text: str, **kwargs) -> List[int]:
        """Encode text to token IDs."""
        return self.tokenizer.encode(text, **kwargs)
    
    def decode(self, token_ids: List[int], **kwargs) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids, **kwargs)
    
    def batch_decode(self, token_ids_batch: List[List[int]], **kwargs) -> List[str]:
        """Decode batch of token IDs."""
        return self.tokenizer.batch_decode(token_ids_batch, **kwargs)
    
    @classmethod
    def from_pretrained(cls, path: str) -> "ReactionTokenizer":
        """Load tokenizer from path."""
        base_tokenizer = T5Tokenizer.from_pretrained(path)
        return cls(base_tokenizer)
    
    def save_pretrained(self, path: str):
        """Save tokenizer to path."""
        self.tokenizer.save_pretrained(path)
    
    def format_reaction(
        self,
        substrates: List[str],
        products: Optional[List[str]] = None,
        cofactors: Optional[List[str]] = None,
        ec_number: Optional[str] = None,
    ) -> str:
        """
        Format a reaction into tokenizable string.
        
        Args:
            substrates: List of substrate SMILES
            products: List of product SMILES (optional)
            cofactors: List of cofactor SMILES (optional)
            ec_number: EC number (optional)
            
        Returns:
            Formatted reaction string
        """
        parts = []
        
        # Add EC number
        if ec_number:
            parts.append(f"[EC]{ec_number}")
        
        # Add substrates
        parts.append(f"[REACTANT]{'.'.join(substrates)}")
        
        # Add cofactors
        if cofactors:
            parts.append(f"[COFACTOR]{'.'.join(cofactors)}")
        
        # Add products
        if products:
            parts.append(f"[PRODUCT]{'.'.join(products)}")
        
        return "[SEP]".join(parts)
    
    def parse_reaction(self, text: str) -> Dict[str, List[str]]:
        """
        Parse a formatted reaction string.
        
        Args:
            text: Formatted reaction string
            
        Returns:
            Dictionary with substrates, products, cofactors, ec_number
        """
        result = {
            "substrates": [],
            "products": [],
            "cofactors": [],
            "ec_number": None,
        }
        
        # Extract EC number
        ec_match = re.search(r"\[EC\]([0-9.]+)", text)
        if ec_match:
            result["ec_number"] = ec_match.group(1)
        
        # Extract components
        for component in ["REACTANT", "PRODUCT", "COFACTOR"]:
            match = re.search(rf"\[{component}\]([^\[]+)", text)
            if match:
                smiles = match.group(1).strip()
                key = {
                    "REACTANT": "substrates",
                    "PRODUCT": "products",
                    "COFACTOR": "cofactors",
                }[component]
                result[key] = [s.strip() for s in smiles.split(".") if s.strip()]
        
        return result
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.tokenizer)
    
    @property
    def pad_token_id(self) -> int:
        """Get pad token ID."""
        return self.tokenizer.pad_token_id
    
    @property
    def eos_token_id(self) -> int:
        """Get EOS token ID."""
        return self.tokenizer.eos_token_id
