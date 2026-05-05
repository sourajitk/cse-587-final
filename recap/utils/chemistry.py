"""
Chemistry utilities for reaction processing.
"""

from typing import Optional, List, Dict, Tuple
from collections import Counter


def validate_smiles(smiles: str) -> bool:
    """
    Validate a SMILES string.
    
    Args:
        smiles: SMILES string to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """
    Convert SMILES to canonical form.
    
    Args:
        smiles: Input SMILES string
        
    Returns:
        Canonical SMILES or None if invalid
    """
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def get_atom_counts(smiles: str) -> Dict[str, int]:
    """
    Get atom counts from SMILES.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Dictionary of element -> count
    """
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        
        counts = Counter()
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            counts[symbol] += 1
            # Add implicit hydrogens
            counts['H'] += atom.GetTotalNumHs()
        
        return dict(counts)
    except Exception:
        return {}


def check_reaction_balance(
    substrates: List[str],
    products: List[str],
    tolerance: float = 0.0,
) -> Tuple[bool, Dict[str, int]]:
    """
    Check if a reaction is atom-balanced.
    
    Args:
        substrates: List of substrate SMILES
        products: List of product SMILES
        tolerance: Allowed difference (for approximate balance)
        
    Returns:
        Tuple of (is_balanced, difference_dict)
    """
    # Count atoms in substrates
    substrate_counts = Counter()
    for smi in substrates:
        counts = get_atom_counts(smi)
        substrate_counts.update(counts)
    
    # Count atoms in products
    product_counts = Counter()
    for smi in products:
        counts = get_atom_counts(smi)
        product_counts.update(counts)
    
    # Calculate difference
    all_elements = set(substrate_counts.keys()) | set(product_counts.keys())
    difference = {}
    
    for element in all_elements:
        diff = substrate_counts.get(element, 0) - product_counts.get(element, 0)
        if diff != 0:
            difference[element] = diff
    
    # Check balance
    is_balanced = len(difference) == 0
    
    if tolerance > 0 and not is_balanced:
        # Check if within tolerance
        total_diff = sum(abs(v) for v in difference.values())
        total_atoms = sum(substrate_counts.values()) + sum(product_counts.values())
        is_balanced = (total_diff / total_atoms) <= tolerance
    
    return is_balanced, difference


def reaction_to_smiles(
    substrates: List[str],
    products: List[str],
    agents: Optional[List[str]] = None,
) -> str:
    """
    Convert reaction components to reaction SMILES.
    
    Args:
        substrates: List of substrate SMILES
        products: List of product SMILES
        agents: Optional list of agent/catalyst SMILES
        
    Returns:
        Reaction SMILES string
    """
    substrate_str = ".".join(substrates)
    product_str = ".".join(products)
    
    if agents:
        agent_str = ".".join(agents)
        return f"{substrate_str}>{agent_str}>{product_str}"
    else:
        return f"{substrate_str}>>{product_str}"


def parse_reaction_smiles(reaction: str) -> Dict[str, List[str]]:
    """
    Parse reaction SMILES into components.
    
    Args:
        reaction: Reaction SMILES string
        
    Returns:
        Dictionary with substrates, products, and agents
    """
    result = {
        "substrates": [],
        "products": [],
        "agents": [],
    }
    
    if ">>" in reaction:
        # No agents
        parts = reaction.split(">>")
        if len(parts) == 2:
            result["substrates"] = [s.strip() for s in parts[0].split(".") if s.strip()]
            result["products"] = [s.strip() for s in parts[1].split(".") if s.strip()]
    elif ">" in reaction:
        # With agents
        parts = reaction.split(">")
        if len(parts) == 3:
            result["substrates"] = [s.strip() for s in parts[0].split(".") if s.strip()]
            result["agents"] = [s.strip() for s in parts[1].split(".") if s.strip()]
            result["products"] = [s.strip() for s in parts[2].split(".") if s.strip()]
    
    return result


def compute_similarity(smiles1: str, smiles2: str, method: str = "tanimoto") -> float:
    """
    Compute molecular similarity between two molecules.
    
    Args:
        smiles1: First SMILES
        smiles2: Second SMILES
        method: Similarity method (tanimoto, dice)
        
    Returns:
        Similarity score (0-1)
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs
        
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            return 0.0
        
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        
        if method == "tanimoto":
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        elif method == "dice":
            return DataStructs.DiceSimilarity(fp1, fp2)
        else:
            return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception:
        return 0.0
