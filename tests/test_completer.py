"""
Tests for ReactionCompleter.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import torch


class TestCompletionResult:
    """Tests for CompletionResult dataclass."""
    
    def test_top_prediction(self):
        from recap.models.completer import CompletionResult
        
        result = CompletionResult(
            input_reaction="test",
            mode="forward",
            predictions=["CC=O", "CCO", "C=O"],
            scores=[0.9, 0.5, 0.3],
        )
        
        assert result.top_prediction == "CC=O"
        assert result.top_score == 0.9
    
    def test_empty_predictions(self):
        from recap.models.completer import CompletionResult
        
        result = CompletionResult(
            input_reaction="test",
            mode="forward",
            predictions=[],
            scores=[],
        )
        
        assert result.top_prediction == ""
        assert result.top_score == 0.0


class TestReactionCompleter:
    """Tests for ReactionCompleter."""
    
    @pytest.fixture
    def mock_completer(self):
        """Create a mock completer for testing."""
        from recap.models.completer import ReactionCompleter
        
        # Mock model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        # Setup tokenizer mock
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_tokenizer.batch_decode = Mock(return_value=["CCO"])
        
        # Setup model mock
        mock_model.generate = Mock(return_value=torch.tensor([[1, 2, 3]]))
        mock_model.to = Mock(return_value=mock_model)
        mock_model.eval = Mock()
        
        completer = ReactionCompleter(
            model=mock_model,
            tokenizer=mock_tokenizer,
            device="cpu",
        )
        
        return completer
    
    def test_format_input_forward(self, mock_completer):
        """Test input formatting for forward mode."""
        input_str = mock_completer._format_input(
            substrates=["CC=O", "O"],
            mode="forward",
        )
        
        assert "REACTANT:" in input_str
        assert "CC=O" in input_str
    
    def test_format_input_retro(self, mock_completer):
        """Test input formatting for retro mode."""
        input_str = mock_completer._format_input(
            products=["CCO"],
            mode="retro",
        )
        
        assert "PRODUCT:" in input_str
        assert "CCO" in input_str
    
    def test_format_input_with_ec(self, mock_completer):
        """Test input formatting with EC number."""
        input_str = mock_completer._format_input(
            substrates=["ATP"],
            mode="forward",
            ec_number="2.7.1.1",
        )
        
        assert "EC:2.7.1.1" in input_str
    
    def test_parse_molecules(self, mock_completer):
        """Test SMILES parsing."""
        molecules = mock_completer._parse_molecules("CC=O.O.CCO")
        
        assert len(molecules) == 3
        assert "CC=O" in molecules
        assert "CCO" in molecules
    
    def test_validate_smiles_valid(self, mock_completer):
        """Test SMILES validation with valid SMILES."""
        # Mock RDKit
        with patch("recap.models.completer.Chem") as mock_chem:
            mock_chem.MolFromSmiles = Mock(return_value=Mock())
            
            assert mock_completer._validate_smiles("CCO") is True
    
    def test_validate_smiles_invalid(self, mock_completer):
        """Test SMILES validation with invalid SMILES."""
        with patch("recap.models.completer.Chem") as mock_chem:
            mock_chem.MolFromSmiles = Mock(return_value=None)
            
            assert mock_completer._validate_smiles("invalid") is False
    
    def test_complete_forward(self, mock_completer):
        """Test forward completion."""
        result = mock_completer.complete(
            substrates=["CC=O", "O"],
            mode="forward",
        )
        
        assert result.mode == "forward"
        assert len(result.predictions) > 0
    
    def test_complete_missing_substrates(self, mock_completer):
        """Test error handling for missing substrates."""
        with pytest.raises(ValueError):
            mock_completer.complete(mode="forward")
    
    def test_complete_missing_products(self, mock_completer):
        """Test error handling for missing products."""
        with pytest.raises(ValueError):
            mock_completer.complete(mode="retro")


class TestReactionT5Model:
    """Tests for ReactionT5Model."""
    
    def test_config_defaults(self):
        from recap.models.reaction_t5 import ReactionT5Config
        
        config = ReactionT5Config()
        
        assert config.max_length == 512
        assert config.num_beams == 5
        assert config.temperature == 1.0
    
    def test_supported_modes(self):
        from recap.models.reaction_t5 import ReactionT5Model
        
        assert "forward" in ReactionT5Model.SUPPORTED_MODES
        assert "retro" in ReactionT5Model.SUPPORTED_MODES
        assert "fill" in ReactionT5Model.SUPPORTED_MODES
