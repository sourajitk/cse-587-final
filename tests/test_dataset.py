"""
Tests for dataset classes.
"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock
import torch


class TestReactionDataset:
    """Tests for ReactionDataset."""
    
    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Create a sample CSV file for testing."""
        csv_path = tmp_path / "test_reactions.csv"
        
        data = pd.DataFrame({
            "substrates": ["CC=O.O", "CCO", "ATP.glucose"],
            "products": ["CCO", "CC=O.O", "ADP.glucose-6-phosphate"],
            "ec_number": ["1.1.1.1", "1.1.1.1", "2.7.1.1"],
        })
        
        data.to_csv(csv_path, index=False)
        return csv_path
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0, 0]]),
        }
        return tokenizer
    
    def test_load_csv(self, sample_csv, mock_tokenizer):
        from recap.data.dataset import ReactionDataset
        
        dataset = ReactionDataset(
            data_path=sample_csv,
            tokenizer=mock_tokenizer,
            mode="forward",
        )
        
        assert len(dataset) == 3
    
    def test_getitem_forward(self, sample_csv, mock_tokenizer):
        from recap.data.dataset import ReactionDataset
        
        dataset = ReactionDataset(
            data_path=sample_csv,
            tokenizer=mock_tokenizer,
            mode="forward",
        )
        
        item = dataset[0]
        
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
    
    def test_getitem_retro(self, sample_csv, mock_tokenizer):
        from recap.data.dataset import ReactionDataset
        
        dataset = ReactionDataset(
            data_path=sample_csv,
            tokenizer=mock_tokenizer,
            mode="retro",
        )
        
        item = dataset[0]
        
        assert "input_ids" in item
        assert "labels" in item


class TestReactionTokenizer:
    """Tests for ReactionTokenizer."""
    
    def test_format_reaction_simple(self):
        from recap.data.tokenizer import ReactionTokenizer
        
        # Create tokenizer with mock base
        tokenizer = MagicMock()
        reaction_tokenizer = ReactionTokenizer.__new__(ReactionTokenizer)
        reaction_tokenizer.tokenizer = tokenizer
        
        # Test format_reaction
        formatted = reaction_tokenizer.format_reaction(
            substrates=["CC=O", "O"],
            products=["CCO"],
        )
        
        assert "[REACTANT]" in formatted
        assert "[PRODUCT]" in formatted
        assert "CC=O" in formatted
    
    def test_format_reaction_with_ec(self):
        from recap.data.tokenizer import ReactionTokenizer
        
        tokenizer = MagicMock()
        reaction_tokenizer = ReactionTokenizer.__new__(ReactionTokenizer)
        reaction_tokenizer.tokenizer = tokenizer
        
        formatted = reaction_tokenizer.format_reaction(
            substrates=["ATP", "glucose"],
            ec_number="2.7.1.1",
        )
        
        assert "[EC]2.7.1.1" in formatted
    
    def test_parse_reaction(self):
        from recap.data.tokenizer import ReactionTokenizer
        
        tokenizer = MagicMock()
        reaction_tokenizer = ReactionTokenizer.__new__(ReactionTokenizer)
        reaction_tokenizer.tokenizer = tokenizer
        
        parsed = reaction_tokenizer.parse_reaction(
            "[EC]1.1.1.1[SEP][REACTANT]CC=O.O[SEP][PRODUCT]CCO"
        )
        
        assert parsed["ec_number"] == "1.1.1.1"
        assert "CC=O" in parsed["substrates"]
        assert "CCO" in parsed["products"]


class TestCreateDataloader:
    """Tests for create_dataloader function."""
    
    def test_create_dataloader(self, tmp_path):
        from recap.data.dataset import create_dataloader
        
        # Create sample data
        csv_path = tmp_path / "test.csv"
        pd.DataFrame({
            "substrates": ["CC=O", "CCO"],
            "products": ["CCO", "CC=O"],
        }).to_csv(csv_path, index=False)
        
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        
        dataloader = create_dataloader(
            data_path=str(csv_path),
            tokenizer=mock_tokenizer,
            batch_size=2,
            shuffle=False,
            num_workers=0,
        )
        
        assert dataloader is not None
        assert len(dataloader) == 1  # 2 samples, batch_size=2
