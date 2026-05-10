"""
KG-Conditioned ReactionT5
=========================

ReactionT5 with cross-attention to FGKG embeddings.
The decoder attends to both:
1. Encoder outputs (substrate representation)
2. KG embeddings (functional group knowledge)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from transformers import T5ForConditionalGeneration, T5Config
import numpy as np

from fg_tokenizer import get_fg_set
from kg_embeddings import FGKGEmbedder


class KGCrossAttention(nn.Module):
    """
    Cross-attention layer for attending to KG embeddings.
    
    Query: decoder hidden states
    Key/Value: projected KG embeddings
    """
    
    def __init__(
        self,
        hidden_dim: int,
        kg_dim: int,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.kg_dim = kg_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        
        # Query projection (from decoder hidden states)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Key/Value projections (from KG embeddings)
        self.k_proj = nn.Linear(kg_dim, hidden_dim)
        self.v_proj = nn.Linear(kg_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Store attention weights for interpretability
        self.last_attention_weights = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,      # (batch, seq_len, hidden_dim)
        kg_embeddings: torch.Tensor,       # (batch, n_fgs, kg_dim)
        kg_mask: Optional[torch.Tensor] = None,  # (batch, n_fgs) - 1 for valid, 0 for padding
    ) -> torch.Tensor:
        """
        Apply cross-attention to KG embeddings.
        
        Args:
            hidden_states: Decoder hidden states
            kg_embeddings: FG embeddings from KG
            kg_mask: Mask for padding FGs
            
        Returns:
            Updated hidden states with residual connection
        """
        batch_size, seq_len, _ = hidden_states.shape
        n_fgs = kg_embeddings.shape[1]
        
        # Project queries, keys, values
        Q = self.q_proj(hidden_states)  # (batch, seq_len, hidden_dim)
        K = self.k_proj(kg_embeddings)  # (batch, n_fgs, hidden_dim)
        V = self.v_proj(kg_embeddings)  # (batch, n_fgs, hidden_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, n_fgs, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, n_fgs, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply mask if provided
        if kg_mask is not None:
            # Expand mask for heads and sequence
            mask = kg_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, n_fgs)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Store for interpretability
        self.last_attention_weights = attn_weights.detach()
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # (batch, n_heads, seq_len, head_dim)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # Output projection
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)
        
        # Residual connection and layer norm
        output = self.layer_norm(hidden_states + attn_output)
        
        return output


class KGConditionedReactionT5(nn.Module):
    """
    ReactionT5 with KG conditioning via cross-attention.
    
    Architecture:
    - Base: ReactionT5v2 (T5 fine-tuned on reactions)
    - Added: KG cross-attention layers after decoder layers
    """
    
    def __init__(
        self,
        model_name: str = "sagawa/ReactionT5v2-forward",
        kg_embedder: Optional[FGKGEmbedder] = None,
        n_kg_layers: int = 2,
        kg_heads: int = 8,
        freeze_base: bool = False,
    ):
        super().__init__()
        
        # Load base model
        self.base_model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.hidden_dim = self.base_model.config.d_model
        
        # KG embedder
        self.kg_embedder = kg_embedder
        self.kg_dim = kg_embedder.embedding_dim if kg_embedder else 128
        
        # KG cross-attention layers
        self.kg_attn_layers = nn.ModuleList([
            KGCrossAttention(
                hidden_dim=self.hidden_dim,
                kg_dim=self.kg_dim,
                n_heads=kg_heads,
            )
            for _ in range(n_kg_layers)
        ])
        
        # Freeze base model if specified
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # Store attention weights for interpretability
        self.last_kg_attention_weights = None
    
    def get_kg_embeddings(
        self,
        substrate_smiles_list: List[str],
        max_fgs: int = 20,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract FG embeddings for a batch of substrates.
        
        Args:
            substrate_smiles_list: List of substrate SMILES strings
            max_fgs: Maximum number of FGs to include
            
        Returns:
            kg_embeddings: (batch, max_fgs, kg_dim)
            kg_mask: (batch, max_fgs)
        """
        batch_size = len(substrate_smiles_list)
        
        if self.kg_embedder is None:
            # Return zeros if no embedder
            return (
                torch.zeros(batch_size, max_fgs, self.kg_dim),
                torch.zeros(batch_size, max_fgs),
            )
        
        all_embeddings = []
        all_masks = []
        
        for smiles in substrate_smiles_list:
            # Get FGs in substrate
            fgs = list(get_fg_set(smiles))
            
            # Get embeddings
            if fgs:
                embeddings = self.kg_embedder.get_embeddings(fgs)
            else:
                embeddings = np.zeros((0, self.kg_dim))
            
            # Pad or truncate
            n_fgs = min(len(fgs), max_fgs)
            padded_emb = np.zeros((max_fgs, self.kg_dim))
            mask = np.zeros(max_fgs)
            
            if n_fgs > 0:
                padded_emb[:n_fgs] = embeddings[:n_fgs]
                mask[:n_fgs] = 1
            
            all_embeddings.append(padded_emb)
            all_masks.append(mask)
        
        kg_embeddings = torch.tensor(np.array(all_embeddings), dtype=torch.float32)
        kg_mask = torch.tensor(np.array(all_masks), dtype=torch.float32)
        
        return kg_embeddings, kg_mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        kg_embeddings: Optional[torch.Tensor] = None,
        kg_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass with KG conditioning.
        """
        device = input_ids.device
        
        # Encode
        encoder_outputs = self.base_model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Prepare decoder inputs
        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = self.base_model._shift_right(labels)
        
        # Decode with base model
        decoder_outputs = self.base_model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
        )
        
        hidden_states = decoder_outputs.last_hidden_state
        
        # Apply KG cross-attention
        if kg_embeddings is not None:
            kg_embeddings = kg_embeddings.to(device)
            if kg_mask is not None:
                kg_mask = kg_mask.to(device)
            
            for kg_attn in self.kg_attn_layers:
                hidden_states = kg_attn(hidden_states, kg_embeddings, kg_mask)
            
            # Store last attention weights
            self.last_kg_attention_weights = self.kg_attn_layers[-1].last_attention_weights
        
        # LM head
        lm_logits = self.base_model.lm_head(hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': lm_logits,
            'hidden_states': hidden_states,
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        substrate_smiles: Optional[str] = None,
        kg_embeddings: Optional[torch.Tensor] = None,
        kg_mask: Optional[torch.Tensor] = None,
        max_length: int = 512,
        num_beams: int = 5,
        **kwargs,
    ):
        """
        Generate products with KG conditioning.
        
        Note: For simplicity, this uses the base model's generate.
        Full KG conditioning during generation requires custom decoding.
        """
        # Get KG embeddings if substrate provided
        if substrate_smiles is not None and kg_embeddings is None:
            kg_embeddings, kg_mask = self.get_kg_embeddings([substrate_smiles])
            kg_embeddings = kg_embeddings.to(input_ids.device)
            kg_mask = kg_mask.to(input_ids.device)
        
        # For now, use base model generation
        # TODO: Implement custom generation with KG attention
        outputs = self.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            **kwargs,
        )
        
        return outputs
    
    def save_pretrained(self, save_dir: str):
        """Save model to directory."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Save base model
        self.base_model.save_pretrained(save_dir)
        
        # Save KG attention layers
        torch.save(
            {
                'kg_attn_layers': self.kg_attn_layers.state_dict(),
                'kg_dim': self.kg_dim,
                'n_kg_layers': len(self.kg_attn_layers),
            },
            os.path.join(save_dir, 'kg_layers.pt')
        )
    
    @classmethod
    def from_pretrained(
        cls,
        load_dir: str,
        kg_embedder: Optional[FGKGEmbedder] = None,
    ):
        """Load model from directory."""
        import os
        
        # Load KG layer config
        kg_config = torch.load(os.path.join(load_dir, 'kg_layers.pt'))
        
        # Create model
        model = cls(
            model_name=load_dir,
            kg_embedder=kg_embedder,
            n_kg_layers=kg_config['n_kg_layers'],
        )
        
        # Load KG attention weights
        model.kg_attn_layers.load_state_dict(kg_config['kg_attn_layers'])
        
        return model


if __name__ == '__main__':
    # Test model creation
    print("Testing KGConditionedReactionT5...")
    
    # Create dummy embedder
    embedder = FGKGEmbedder(embedding_dim=128)
    embedder.entity2idx = {'hydroxyl': 0, 'ketone': 1, 'ester': 2}
    embedder.model = None  # No trained model for test
    
    # Create model (will download ReactionT5v2)
    try:
        model = KGConditionedReactionT5(
            model_name="sagawa/ReactionT5v2-forward",
            kg_embedder=embedder,
            n_kg_layers=2,
        )
        print("Model created successfully!")
        print(f"Hidden dim: {model.hidden_dim}")
        print(f"KG dim: {model.kg_dim}")
        print(f"KG attention layers: {len(model.kg_attn_layers)}")
    except Exception as e:
        print(f"Model creation failed (expected if offline): {e}")
