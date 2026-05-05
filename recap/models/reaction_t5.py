"""
ReactionT5v2 model wrapper for reaction completion.
"""

from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Config, PreTrainedModel


@dataclass
class ReactionT5Config:
    """Configuration for ReactionT5 model."""
    
    model_name: str = "sagawa/ReactionT5v2-forward"
    max_length: int = 512
    num_beams: int = 5
    num_return_sequences: int = 5
    temperature: float = 1.0
    do_sample: bool = False
    top_k: int = 50
    top_p: float = 0.95
    
    # Fine-tuning settings
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    
    # Model architecture
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    dropout: float = 0.1


class ReactionT5Model(nn.Module):
    """
    Wrapper around ReactionT5v2 for biochemical reaction completion.
    
    Supports multiple completion modes:
    - forward: substrates → products
    - retro: products → substrates
    - fill: partial reaction → complete reaction
    
    Args:
        config: Model configuration
        pretrained: Load pretrained weights
    """
    
    SUPPORTED_MODES = ["forward", "retro", "fill"]
    
    # Pretrained model mappings
    PRETRAINED_MODELS = {
        "forward": "sagawa/ReactionT5v2-forward",
        "retro": "sagawa/ReactionT5v2-retrosynthesis",
    }
    
    def __init__(
        self,
        config: Optional[ReactionT5Config] = None,
        pretrained: bool = True,
    ):
        super().__init__()
        
        self.config = config or ReactionT5Config()
        
        if pretrained:
            self.model = T5ForConditionalGeneration.from_pretrained(
                self.config.model_name
            )
        else:
            t5_config = T5Config(
                d_model=self.config.hidden_size,
                num_layers=self.config.num_layers,
                num_heads=self.config.num_heads,
                dropout_rate=self.config.dropout,
            )
            self.model = T5ForConditionalGeneration(t5_config)
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        mode: str = "forward",
        **kwargs,
    ) -> "ReactionT5Model":
        """
        Load pretrained model.
        
        Args:
            model_name_or_path: Model identifier or path
            mode: Completion mode (forward, retro)
            **kwargs: Additional arguments
            
        Returns:
            Loaded model
        """
        config = ReactionT5Config(**kwargs)
        
        # Map mode to pretrained model
        if model_name_or_path in cls.PRETRAINED_MODELS:
            config.model_name = cls.PRETRAINED_MODELS[model_name_or_path]
        elif mode in cls.PRETRAINED_MODELS:
            config.model_name = cls.PRETRAINED_MODELS[mode]
        else:
            config.model_name = model_name_or_path
        
        return cls(config=config, pretrained=True)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target token IDs for training
            
        Returns:
            Model outputs including loss if labels provided
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
        
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "encoder_last_hidden_state": outputs.encoder_last_hidden_state,
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate completions.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            **kwargs: Generation arguments
            
        Returns:
            Generated token IDs
        """
        gen_kwargs = {
            "max_length": kwargs.pop("max_length", self.config.max_length),
            "num_beams": kwargs.pop("num_beams", self.config.num_beams),
            "num_return_sequences": kwargs.pop(
                "num_return_sequences", 
                self.config.num_return_sequences
            ),
            "do_sample": kwargs.pop("do_sample", self.config.do_sample),
            "temperature": kwargs.pop("temperature", self.config.temperature),
            "top_k": kwargs.pop("top_k", self.config.top_k),
            "top_p": kwargs.pop("top_p", self.config.top_p),
        }
        
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
            **kwargs,
        )
    
    def save_pretrained(self, save_directory: str):
        """Save model to directory."""
        self.model.save_pretrained(save_directory)
    
    def get_encoder(self) -> nn.Module:
        """Get the encoder for embedding extraction."""
        return self.model.encoder
    
    def get_decoder(self) -> nn.Module:
        """Get the decoder."""
        return self.model.decoder
