"""User Encoder Module: Multi-interest User Modeling"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiInterestExtractor(nn.Module):
    """Extract multiple interest vectors from user history"""
    
    def __init__(self, input_dim: int, num_interests: int, dropout: float = 0.2):
        super().__init__()
        self.num_interests = num_interests
        self.input_dim = input_dim
        
        # Multi-head attention to extract interests
        self.interest_queries = nn.Parameter(
            torch.randn(num_interests, input_dim)
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Projection
        self.projection = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, history_embeddings: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract multiple interest vectors
        
        Args:
            history_embeddings: (batch_size, history_len, input_dim)
            mask: (batch_size, history_len) attention mask
            
        Returns:
            Interest vectors: (batch_size, num_interests, input_dim)
        """
        batch_size = history_embeddings.size(0)
        
        # Expand interest queries
        queries = self.interest_queries.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # (batch, num_interests, input_dim)
        
        # Multi-head attention
        interests, _ = self.attention(
            queries,  # query
            history_embeddings,  # key
            history_embeddings,  # value
            key_padding_mask=mask if mask is not None else None
        )  # (batch, num_interests, input_dim)
        
        # Projection
        interests = self.projection(interests)
        
        return interests


class ContextAwareAttention(nn.Module):
    """Context-aware attention to select relevant interest"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.attention = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Tanh(),
            nn.Linear(dim, 1)
        )
    
    def forward(self, interests: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        """
        Select most relevant interest for candidate news
        
        Args:
            interests: (batch_size, num_interests, dim)
            candidate: (batch_size, dim)
            
        Returns:
            User representation: (batch_size, dim)
        """
        batch_size, num_interests, dim = interests.shape
        
        # Expand candidate
        candidate_expanded = candidate.unsqueeze(1).expand(
            -1, num_interests, -1
        )  # (batch, num_interests, dim)
        
        # Concatenate
        combined = torch.cat([interests, candidate_expanded], dim=-1)  # (batch, num_interests, 2*dim)
        
        # Attention weights
        weights = self.attention(combined).squeeze(-1)  # (batch, num_interests)
        weights = F.softmax(weights, dim=1)  # (batch, num_interests)
        
        # Weighted sum
        user_emb = (weights.unsqueeze(-1) * interests).sum(dim=1)  # (batch, dim)
        
        return user_emb


class UserEncoder(nn.Module):
    """User Encoder: Encodes user history into user representation"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.max_history_len = config.max_history_len
        
        # Multi-interest extractor
        self.multi_interest = MultiInterestExtractor(
            input_dim=config.user_encoder_dim,
            num_interests=config.num_interests,
            dropout=config.user_encoder_dropout
        )
        
        # Context-aware attention
        self.context_attention = ContextAwareAttention(
            dim=config.user_encoder_dim
        )
        
        # Optional: Sequential modeling (GRU)
        self.gru = nn.GRU(
            input_size=config.user_encoder_dim,
            hidden_size=config.user_encoder_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
    
    def forward(self, history_embeddings: torch.Tensor,
                candidate_embedding: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode user based on history and candidate news
        
        Args:
            history_embeddings: (batch_size, history_len, user_encoder_dim)
            candidate_embedding: (batch_size, user_encoder_dim)
            mask: (batch_size, history_len) padding mask
            
        Returns:
            User representation: (batch_size, user_encoder_dim)
        """
        # Optional: Apply GRU for sequential modeling
        if hasattr(self, 'gru'):
            history_embeddings, _ = self.gru(history_embeddings)
        
        # Extract multiple interests
        interests = self.multi_interest(history_embeddings, mask)  # (batch, num_interests, dim)
        
        # Context-aware attention
        user_emb = self.context_attention(interests, candidate_embedding)  # (batch, dim)
        
        return user_emb

