"""
Functional Group Knowledge Graph (FGKG) Builder
===============================================

Builds a reaction-centric knowledge graph from reaction data.
Uses FARM's exact detect_functional_group() implementation.

Graph Structure:
- Nodes: Functional group types (e.g., 'hydroxyl', 'ketone')
- Edges: 
  - transforms_to: FG A consumed, FG B formed in same reaction
  - co_consumed: FGs consumed together
  - co_formed: FGs formed together
"""

import pickle
import argparse
from typing import Dict, List, Set, Tuple
from collections import defaultdict, Counter
import pandas as pd
from tqdm import tqdm

from fg_tokenizer import extract_fg_from_reaction, get_fg_set


class FGKG:
    """Functional Group Knowledge Graph."""
    
    def __init__(self):
        # Node info
        self.fg_nodes: Set[str] = set()
        self.fg_counts: Counter = Counter()  # How often each FG appears
        
        # Edge info (head, relation, tail) -> count
        self.transforms_to: Dict[Tuple[str, str], int] = defaultdict(int)
        self.co_consumed: Dict[Tuple[str, str], int] = defaultdict(int)
        self.co_formed: Dict[Tuple[str, str], int] = defaultdict(int)
        
        # Statistics
        self.n_reactions = 0
    
    def add_reaction(self, substrates: List[str], products: List[str]):
        """
        Add a reaction to the knowledge graph.
        
        Args:
            substrates: List of substrate SMILES
            products: List of product SMILES
        """
        # Extract FG changes
        changes = extract_fg_from_reaction(substrates, products)
        
        consumed_fgs = set(changes['consumed_fgs'].keys())
        formed_fgs = set(changes['formed_fgs'].keys())
        all_fgs = set(changes['substrate_fgs'].keys()) | set(changes['product_fgs'].keys())
        
        # Update node counts
        for fg, count in changes['substrate_fgs'].items():
            self.fg_nodes.add(fg)
            self.fg_counts[fg] += count
        for fg, count in changes['product_fgs'].items():
            self.fg_nodes.add(fg)
            self.fg_counts[fg] += count
        
        # transforms_to edges: consumed FG -> formed FG
        for consumed in consumed_fgs:
            for formed in formed_fgs:
                self.transforms_to[(consumed, formed)] += 1
        
        # co_consumed edges
        consumed_list = list(consumed_fgs)
        for i in range(len(consumed_list)):
            for j in range(i + 1, len(consumed_list)):
                pair = tuple(sorted([consumed_list[i], consumed_list[j]]))
                self.co_consumed[pair] += 1
        
        # co_formed edges
        formed_list = list(formed_fgs)
        for i in range(len(formed_list)):
            for j in range(i + 1, len(formed_list)):
                pair = tuple(sorted([formed_list[i], formed_list[j]]))
                self.co_formed[pair] += 1
        
        self.n_reactions += 1
    
    def get_triples(self, min_count: int = 1) -> List[Tuple[str, str, str]]:
        """
        Get KG triples for embedding training.
        
        Args:
            min_count: Minimum edge count to include
            
        Returns:
            List of (head, relation, tail) triples
        """
        triples = []
        
        for (head, tail), count in self.transforms_to.items():
            if count >= min_count:
                triples.append((head, 'transforms_to', tail))
        
        for (fg1, fg2), count in self.co_consumed.items():
            if count >= min_count:
                triples.append((fg1, 'co_consumed', fg2))
                triples.append((fg2, 'co_consumed', fg1))  # Symmetric
        
        for (fg1, fg2), count in self.co_formed.items():
            if count >= min_count:
                triples.append((fg1, 'co_formed', fg2))
                triples.append((fg2, 'co_formed', fg1))  # Symmetric
        
        return triples
    
    def summary(self) -> str:
        """Print summary statistics."""
        lines = [
            "=" * 60,
            "FGKG Summary",
            "=" * 60,
            f"Reactions processed: {self.n_reactions}",
            f"Unique FG types: {len(self.fg_nodes)}",
            f"transforms_to edges: {len(self.transforms_to)}",
            f"co_consumed edges: {len(self.co_consumed)}",
            f"co_formed edges: {len(self.co_formed)}",
            "",
            "Top 20 FGs by occurrence:",
        ]
        for fg, count in self.fg_counts.most_common(20):
            lines.append(f"  {fg}: {count}")
        
        lines.append("")
        lines.append("Top 20 transforms_to edges:")
        sorted_transforms = sorted(self.transforms_to.items(), key=lambda x: -x[1])[:20]
        for (head, tail), count in sorted_transforms:
            lines.append(f"  {head} -> {tail}: {count}")
        
        return "\n".join(lines)
    
    def save(self, path: str):
        """Save FGKG to pickle file."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Saved FGKG to {path}")
    
    @staticmethod
    def load(path: str) -> 'FGKG':
        """Load FGKG from pickle file."""
        with open(path, 'rb') as f:
            return pickle.load(f)


def build_fgkg_from_csv(
    csv_path: str,
    substrate_col: str = 'substrates',
    product_col: str = 'products',
    sep: str = '.',
) -> FGKG:
    """
    Build FGKG from a CSV file of reactions.
    
    Args:
        csv_path: Path to CSV file
        substrate_col: Column name for substrates (SMILES, dot-separated)
        product_col: Column name for products (SMILES, dot-separated)
        sep: Separator for multiple molecules
        
    Returns:
        FGKG instance
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} reactions from {csv_path}")
    
    fgkg = FGKG()
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building FGKG"):
        try:
            substrates = str(row[substrate_col]).split(sep)
            products = str(row[product_col]).split(sep)
            
            # Filter empty strings
            substrates = [s.strip() for s in substrates if s.strip()]
            products = [p.strip() for p in products if p.strip()]
            
            if substrates and products:
                fgkg.add_reaction(substrates, products)
        except Exception as e:
            continue  # Skip malformed reactions
    
    return fgkg


def main():
    parser = argparse.ArgumentParser(description="Build FGKG from reaction data")
    parser.add_argument('--input', '-i', required=True, help='Input CSV file')
    parser.add_argument('--output', '-o', required=True, help='Output pickle file')
    parser.add_argument('--substrate-col', default='substrates', help='Substrate column name')
    parser.add_argument('--product-col', default='products', help='Product column name')
    parser.add_argument('--sep', default='.', help='SMILES separator')
    
    args = parser.parse_args()
    
    fgkg = build_fgkg_from_csv(
        args.input,
        substrate_col=args.substrate_col,
        product_col=args.product_col,
        sep=args.sep,
    )
    
    print(fgkg.summary())
    fgkg.save(args.output)


if __name__ == '__main__':
    main()
