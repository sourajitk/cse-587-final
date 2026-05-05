"""
Dataset classes for reaction completion training.
"""

from typing import Optional, List, Dict, Any, Union, Callable
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from tqdm import tqdm


class ReactionDataset(Dataset):
    """
    PyTorch Dataset for biochemical reactions.
    
    Supports multiple data formats:
    - CSV with columns: substrates, products, (optional) ec_number
    - MetaNetX format
    - SMILES reaction format (reactants>>products)
    
    Args:
        data_path: Path to data file or directory
        tokenizer: Tokenizer for encoding reactions
        mode: Completion mode (forward, retro, fill)
        max_length: Maximum sequence length
        augment: Whether to apply data augmentation
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: PreTrainedTokenizer,
        mode: str = "forward",
        max_length: int = 512,
        augment: bool = False,
        transform: Optional[Callable] = None,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = max_length
        self.augment = augment
        self.transform = transform
        
        # Load data
        self.data = self._load_data()
        
    def _load_data(self) -> pd.DataFrame:
        """Load data from file."""
        if self.data_path.suffix == ".csv":
            return self._load_csv()
        elif self.data_path.suffix == ".tsv":
            return pd.read_csv(self.data_path, sep="\t")
        elif self.data_path.suffix == ".parquet":
            return pd.read_parquet(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
    
    def _load_csv(self) -> pd.DataFrame:
        """Load CSV data."""
        df = pd.read_csv(self.data_path)
        
        # Normalize column names
        column_mapping = {
            "SMILES": "substrates",
            "smiles": "substrates",
            "reactants": "substrates",
            "Reactants": "substrates",
            "Products": "products",
            "product": "products",
            "EC": "ec_number",
            "ec": "ec_number",
        }
        
        df = df.rename(columns=column_mapping)
        
        # Handle reaction SMILES format (reactants>>products)
        if "reaction" in df.columns:
            df[["substrates", "products"]] = df["reaction"].str.split(
                ">>", expand=True
            )
        
        return df
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example."""
        row = self.data.iloc[idx]
        
        # Format input/output based on mode
        if self.mode == "forward":
            input_text = f"REACTANT:{row['substrates']}"
            target_text = row["products"]
        elif self.mode == "retro":
            input_text = f"PRODUCT:{row['products']}"
            target_text = row["substrates"]
        elif self.mode == "fill":
            # Random masking for fill mode
            input_text, target_text = self._create_fill_example(row)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # Add EC number if available
        if "ec_number" in row and pd.notna(row["ec_number"]):
            input_text = f"EC:{row['ec_number']} {input_text}"
        
        # Apply augmentation
        if self.augment:
            input_text = self._augment(input_text)
        
        # Apply custom transform
        if self.transform:
            input_text, target_text = self.transform(input_text, target_text)
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        targets = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": targets["input_ids"].squeeze(),
            "idx": idx,
        }
    
    def _create_fill_example(self, row: pd.Series) -> tuple:
        """Create a fill-in-the-blank example."""
        import random
        
        substrates = row["substrates"].split(".")
        products = row["products"].split(".")
        
        # Randomly mask some components
        if random.random() < 0.5 and len(substrates) > 1:
            # Mask a substrate
            mask_idx = random.randint(0, len(substrates) - 1)
            masked = substrates.copy()
            masked[mask_idx] = "?"
            input_text = f"PARTIAL:{'.'.join(masked)}>>{row['products']}"
            target_text = substrates[mask_idx]
        else:
            # Mask a product
            mask_idx = random.randint(0, len(products) - 1)
            masked = products.copy()
            masked[mask_idx] = "?"
            input_text = f"PARTIAL:{row['substrates']}>>{'.'.join(masked)}"
            target_text = products[mask_idx]
        
        return input_text, target_text
    
    def _augment(self, text: str) -> str:
        """Apply data augmentation."""
        import random
        from rdkit import Chem
        
        # SMILES randomization (different valid SMILES for same molecule)
        def randomize_smiles(smi: str) -> str:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    return smi
                return Chem.MolToSmiles(mol, doRandom=True)
            except Exception:
                return smi
        
        # Extract SMILES portion and randomize
        if ":" in text:
            prefix, smiles_part = text.split(":", 1)
            molecules = smiles_part.split(".")
            randomized = [randomize_smiles(m.strip()) for m in molecules]
            
            # Shuffle order (reaction invariant to order within reactants/products)
            if random.random() < 0.5:
                random.shuffle(randomized)
            
            text = f"{prefix}:{'.'.join(randomized)}"
        
        return text


class MetaNetXDataset(ReactionDataset):
    """
    Dataset specifically for MetaNetX biochemical reactions.
    
    Handles MetaNetX-specific format and filtering.
    """
    
    # Common cofactors to potentially exclude
    COFACTORS = {
        "MNXM2",  # ATP
        "MNXM3",  # ADP
        "MNXM7",  # NAD+
        "MNXM8",  # NADH
        "MNXM5",  # H2O
        "MNXM1",  # H+
        "MNXM9",  # CO2
    }
    
    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: PreTrainedTokenizer,
        mode: str = "forward",
        include_cofactors: bool = True,
        balance_threshold: float = 0.1,
        **kwargs,
    ):
        self.include_cofactors = include_cofactors
        self.balance_threshold = balance_threshold
        
        super().__init__(data_path, tokenizer, mode, **kwargs)
    
    def _load_data(self) -> pd.DataFrame:
        """Load and process MetaNetX data."""
        df = super()._load_data()
        
        # Filter by balance if available
        if "is_balanced" in df.columns:
            df = df[df["is_balanced"] == True]
        
        # Optionally filter cofactors
        if not self.include_cofactors and "cofactors" in df.columns:
            df = df[df["cofactors"].isna() | (df["cofactors"] == "")]
        
        return df


def create_dataloader(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    mode: str = "forward",
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader for reaction data.
    
    Args:
        data_path: Path to data file
        tokenizer: Tokenizer
        mode: Completion mode
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers
        **kwargs: Additional dataset arguments
        
    Returns:
        DataLoader
    """
    dataset = ReactionDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        mode=mode,
        **kwargs,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
