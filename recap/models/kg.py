"""
Knowledge Graph implementation for biochemical reactions.
"""

import os
import csv
from typing import List, Set, Optional
import networkx as nx

class BiochemicalKG:
    """
    In-memory knowledge graph using NetworkX.
    Nodes are chemicals (SMILES/IDs), edges are reactions.
    """
    def __init__(self):
        self.graph = nx.DiGraph()
    
    def load_from_tsv(self, file_path: str):
        """
        Load graph edges from a TSV file (e.g. MetaNetX reac_prop.tsv or custom edges).
        Expected format: reactant_smiles \t product_smiles
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Graph file not found: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) >= 2:
                    reactant = row[0].strip()
                    product = row[1].strip()
                    if reactant and product and reactant != product:
                        self.graph.add_edge(reactant, product)

    def add_reaction(self, substrates: List[str], products: List[str]):
        """Add a reaction to the graph manually."""
        for s in substrates:
            for p in products:
                if s and p and s != p:
                    self.graph.add_edge(s, p)
                    
    def check_plausibility(self, substrates: List[str], products: List[str], max_hops: int = 2) -> float:
        """
        Check if a predicted reaction is plausible based on the graph.
        Returns a confidence boost score [0.0 - 1.0].
        """
        if len(self.graph) == 0:
            return 0.0
            
        found_path = False
        for s in substrates:
            for p in products:
                if s in self.graph and p in self.graph:
                    try:
                        # Check if product is reachable from substrate within max_hops
                        length = nx.shortest_path_length(self.graph, source=s, target=p)
                        if length <= max_hops:
                            found_path = True
                            break
                    except nx.NetworkXNoPath:
                        continue
            if found_path:
                break
                
        if found_path:
            return 0.5  # Boost score if path exists
        return 0.0
