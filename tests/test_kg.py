import pytest
from recap.models.kg import BiochemicalKG
import tempfile
import os

def test_kg_add_reaction():
    kg = BiochemicalKG()
    kg.add_reaction(["A", "B"], ["C", "D"])
    assert kg.graph.has_edge("A", "C")
    assert kg.graph.has_edge("B", "D")
    assert not kg.graph.has_edge("A", "B")
    
def test_kg_check_plausibility():
    kg = BiochemicalKG()
    kg.add_reaction(["A"], ["B"])
    kg.add_reaction(["B"], ["C"])
    
    # 1 hop
    assert kg.check_plausibility(["A"], ["B"]) > 0
    # 2 hops
    assert kg.check_plausibility(["A"], ["C"]) > 0
    # No path
    assert kg.check_plausibility(["A"], ["D"]) == 0.0
    
def test_kg_load_tsv():
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.tsv') as f:
        f.write("reactant1\tproduct1\nreactant2\tproduct2\n")
        temp_path = f.name
        
    try:
        kg = BiochemicalKG()
        kg.load_from_tsv(temp_path)
        assert kg.graph.has_edge("reactant1", "product1")
        assert kg.graph.has_edge("reactant2", "product2")
    finally:
        os.unlink(temp_path)
