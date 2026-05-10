from fgkg_builder import FGKG
"""
KG Embeddings for FGKG
======================

Trains TransE embeddings on the Functional Group Knowledge Graph.
Embeddings can be used to condition ReactionT5 via cross-attention.
"""

import os
import json
import argparse
import pickle
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class TransE(nn.Module):
    """
    TransE knowledge graph embedding model.
    
    For a triple (h, r, t), TransE enforces: h + r ≈ t
    """
    
    def __init__(
        self,
        n_entities: int,
        n_relations: int,
        embedding_dim: int = 128,
        margin: float = 1.0,
        norm: int = 2,
    ):
        super().__init__()
        
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.norm = norm
        
        # Initialize embeddings
        self.entity_embeddings = nn.Embedding(n_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(n_relations, embedding_dim)
        
        # Initialize with uniform distribution
        nn.init.uniform_(self.entity_embeddings.weight, -6/np.sqrt(embedding_dim), 6/np.sqrt(embedding_dim))
        nn.init.uniform_(self.relation_embeddings.weight, -6/np.sqrt(embedding_dim), 6/np.sqrt(embedding_dim))
        
        # Normalize relation embeddings
        with torch.no_grad():
            self.relation_embeddings.weight.data = F.normalize(self.relation_embeddings.weight.data, p=2, dim=1)
    
    def forward(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute TransE distance: ||h + r - t||
        
        Args:
            heads: (batch,) entity indices
            relations: (batch,) relation indices
            tails: (batch,) entity indices
            
        Returns:
            distances: (batch,) TransE distances
        """
        h = self.entity_embeddings(heads)
        r = self.relation_embeddings(relations)
        t = self.entity_embeddings(tails)
        
        # Normalize entity embeddings during forward pass
        h = F.normalize(h, p=2, dim=1)
        t = F.normalize(t, p=2, dim=1)
        
        # TransE score: ||h + r - t||
        score = torch.norm(h + r - t, p=self.norm, dim=1)
        return score
    
    def margin_loss(
        self,
        pos_heads: torch.Tensor,
        pos_relations: torch.Tensor,
        pos_tails: torch.Tensor,
        neg_heads: torch.Tensor,
        neg_relations: torch.Tensor,
        neg_tails: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute margin-based ranking loss.
        
        L = max(0, margin + d(h,r,t) - d(h',r,t'))
        """
        pos_score = self.forward(pos_heads, pos_relations, pos_tails)
        neg_score = self.forward(neg_heads, neg_relations, neg_tails)
        
        loss = F.relu(self.margin + pos_score - neg_score)
        return loss.mean()


class TripleDataset(Dataset):
    """Dataset of KG triples with negative sampling."""
    
    def __init__(
        self,
        triples: List[Tuple[int, int, int]],
        n_entities: int,
        n_neg_samples: int = 1,
    ):
        self.triples = triples
        self.n_entities = n_entities
        self.n_neg_samples = n_neg_samples
    
    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, idx):
        h, r, t = self.triples[idx]
        
        # Negative sampling: corrupt head or tail
        if np.random.rand() < 0.5:
            # Corrupt head
            neg_h = np.random.randint(0, self.n_entities)
            while neg_h == h:
                neg_h = np.random.randint(0, self.n_entities)
            neg_t = t
        else:
            # Corrupt tail
            neg_h = h
            neg_t = np.random.randint(0, self.n_entities)
            while neg_t == t:
                neg_t = np.random.randint(0, self.n_entities)
        
        return {
            'pos_h': h,
            'pos_r': r,
            'pos_t': t,
            'neg_h': neg_h,
            'neg_r': r,
            'neg_t': neg_t,
        }


class FGKGEmbedder:
    """
    Manages FGKG embeddings.
    
    Handles training, saving, loading, and lookup of FG embeddings.
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.embedding_dim = embedding_dim
        self.device = device
        
        self.entity2idx: Dict[str, int] = {}
        self.idx2entity: Dict[int, str] = {}
        self.relation2idx: Dict[str, int] = {}
        self.idx2relation: Dict[int, str] = {}
        
        self.model: Optional[TransE] = None
    
    def build_vocab(self, triples: List[Tuple[str, str, str]]):
        """Build entity and relation vocabularies from triples."""
        entities = set()
        relations = set()
        
        for h, r, t in triples:
            entities.add(h)
            entities.add(t)
            relations.add(r)
        
        self.entity2idx = {e: i for i, e in enumerate(sorted(entities))}
        self.idx2entity = {i: e for e, i in self.entity2idx.items()}
        self.relation2idx = {r: i for i, r in enumerate(sorted(relations))}
        self.idx2relation = {i: r for r, i in self.relation2idx.items()}
        
        print(f"Vocabulary: {len(self.entity2idx)} entities, {len(self.relation2idx)} relations")
    
    def encode_triples(self, triples: List[Tuple[str, str, str]]) -> List[Tuple[int, int, int]]:
        """Convert string triples to index triples."""
        encoded = []
        for h, r, t in triples:
            if h in self.entity2idx and r in self.relation2idx and t in self.entity2idx:
                encoded.append((self.entity2idx[h], self.relation2idx[r], self.entity2idx[t]))
        return encoded
    
    def train(
        self,
        triples: List[Tuple[str, str, str]],
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 0.01,
        margin: float = 1.0,
    ):
        """
        Train TransE embeddings on triples.
        
        Args:
            triples: List of (head, relation, tail) string triples
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            margin: Margin for ranking loss
        """
        # Build vocabulary
        self.build_vocab(triples)
        
        # Encode triples
        encoded_triples = self.encode_triples(triples)
        print(f"Training on {len(encoded_triples)} triples")
        
        # Initialize model
        self.model = TransE(
            n_entities=len(self.entity2idx),
            n_relations=len(self.relation2idx),
            embedding_dim=self.embedding_dim,
            margin=margin,
        ).to(self.device)
        
        # Dataset and dataloader
        dataset = TripleDataset(encoded_triples, len(self.entity2idx))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            n_batches = 0
            
            for batch in dataloader:
                pos_h = batch['pos_h'].to(self.device)
                pos_r = batch['pos_r'].to(self.device)
                pos_t = batch['pos_t'].to(self.device)
                neg_h = batch['neg_h'].to(self.device)
                neg_r = batch['neg_r'].to(self.device)
                neg_t = batch['neg_t'].to(self.device)
                
                optimizer.zero_grad()
                loss = self.model.margin_loss(pos_h, pos_r, pos_t, neg_h, neg_r, neg_t)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / n_batches
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def get_embedding(self, fg_name: str) -> Optional[np.ndarray]:
        """Get embedding for a functional group."""
        if self.model is None:
            return None
        if fg_name not in self.entity2idx:
            return None
        
        idx = self.entity2idx[fg_name]
        with torch.no_grad():
            emb = self.model.entity_embeddings.weight[idx].cpu().numpy()
        return emb
    
    def get_embeddings(self, fg_names: List[str]) -> np.ndarray:
        """
        Get embeddings for multiple FGs.
        
        Returns zero vector for unknown FGs.
        """
        embeddings = []
        for fg in fg_names:
            emb = self.get_embedding(fg)
            if emb is None:
                emb = np.zeros(self.embedding_dim)
            embeddings.append(emb)
        return np.array(embeddings)
    
    def save(self, output_dir: str):
        """Save embedder to directory."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        if self.model is not None:
            torch.save(self.model.state_dict(), os.path.join(output_dir, 'model.pt'))
        
        # Save vocabularies
        vocab = {
            'entity2idx': self.entity2idx,
            'relation2idx': self.relation2idx,
            'embedding_dim': self.embedding_dim,
        }
        with open(os.path.join(output_dir, 'vocab.json'), 'w') as f:
            json.dump(vocab, f, indent=2)
        
        # Save embeddings as numpy for easy loading
        if self.model is not None:
            with torch.no_grad():
                entity_embs = self.model.entity_embeddings.weight.cpu().numpy()
                relation_embs = self.model.relation_embeddings.weight.cpu().numpy()
            np.save(os.path.join(output_dir, 'entity_embeddings.npy'), entity_embs)
            np.save(os.path.join(output_dir, 'relation_embeddings.npy'), relation_embs)
        
        print(f"Saved embedder to {output_dir}")
    
    @classmethod
    def load(cls, input_dir: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> 'FGKGEmbedder':
        """Load embedder from directory."""
        # Load vocabulary
        with open(os.path.join(input_dir, 'vocab.json'), 'r') as f:
            vocab = json.load(f)
        
        embedder = cls(embedding_dim=vocab['embedding_dim'], device=device)
        embedder.entity2idx = vocab['entity2idx']
        embedder.idx2entity = {int(i): e for e, i in embedder.entity2idx.items()}
        embedder.relation2idx = vocab['relation2idx']
        embedder.idx2relation = {int(i): r for r, i in embedder.relation2idx.items()}
        
        # Load model
        embedder.model = TransE(
            n_entities=len(embedder.entity2idx),
            n_relations=len(embedder.relation2idx),
            embedding_dim=embedder.embedding_dim,
        ).to(device)
        
        model_path = os.path.join(input_dir, 'model.pt')
        if os.path.exists(model_path):
            embedder.model.load_state_dict(torch.load(model_path, map_location=device))
        
        print(f"Loaded embedder from {input_dir}")
        return embedder


def main():
    parser = argparse.ArgumentParser(description="Train FGKG embeddings")
    parser.add_argument('--fgkg', required=True, help='Path to FGKG pickle file')
    parser.add_argument('--output', '-o', required=True, help='Output directory for embeddings')
    parser.add_argument('--dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--min-count', type=int, default=2, help='Minimum edge count')
    
    args = parser.parse_args()
    
    # Load FGKG
    from fgkg_builder import FGKG
    fgkg = FGKG.load(args.fgkg)
    
    # Get triples
    triples = fgkg.get_triples(min_count=args.min_count)
    print(f"Got {len(triples)} triples from FGKG")
    
    # Train embeddings
    embedder = FGKGEmbedder(embedding_dim=args.dim)
    embedder.train(
        triples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    
    # Save
    embedder.save(args.output)


if __name__ == '__main__':
    main()
