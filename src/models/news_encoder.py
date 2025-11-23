"""News Encoder Module"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FastAttention(nn.Module):
    """Fastformer: Additive Attention (O(n) complexity)"""
    
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, dim)
        Returns:
            (batch_size, dim)
        """
        batch_size, seq_len, dim = x.shape
        
        # Query: global query vector
        q = self.query(x.mean(dim=1, keepdim=True))  # (batch, 1, dim)
        
        # Key: element-wise interaction
        k = self.key(x)  # (batch, seq_len, dim)
        alpha = torch.softmax((q * k).sum(dim=-1, keepdim=True), dim=1)  # (batch, seq_len, 1)
        
        # Value: weighted aggregation
        v = self.value(x)  # (batch, seq_len, dim)
        out = (alpha * v).sum(dim=1)  # (batch, dim)
        
        return self.dropout(out)


class NewsEncoder(nn.Module):
    """News Encoder: Encodes news articles into vectors"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Text encoder (imported from data module)
        from src.data.text_encoder import create_text_encoder
        self.text_encoder = create_text_encoder(config)
        
        # Category embeddings
        self.category_embedding = nn.Embedding(1000, config.category_embedding_dim)  # Assume max 1000 categories
        self.subcategory_embedding = nn.Embedding(1000, config.subcategory_embedding_dim)
        
        # Entity embeddings (if used)
        if config.use_entities:
            self.entity_embedding = nn.Embedding(10000, config.entity_embedding_dim)  # Assume max 10000 entities
            # Attention for entity fusion
            self.entity_attention = nn.MultiheadAttention(
                embed_dim=config.news_encoder_dim,
                num_heads=4,
                dropout=config.news_encoder_dropout,
                batch_first=True
            )
        
        # Fastformer layers
        self.fast_attention = FastAttention(
            dim=config.news_encoder_dim,
            dropout=config.news_encoder_dropout
        )
        
        # Projection layers
        text_dim = config.news_encoder_dim
        cat_dim = config.category_embedding_dim + config.subcategory_embedding_dim
        
        # Combine text + category
        self.fusion = nn.Sequential(
            nn.Linear(text_dim + cat_dim, config.news_encoder_dim),
            nn.LayerNorm(config.news_encoder_dim),
            nn.ReLU(),
            nn.Dropout(config.news_encoder_dropout)
        )
        
        # Final projection
        self.output_proj = nn.Linear(config.news_encoder_dim, config.final_dim)
    
    def forward(self, title: list, abstract: list, 
                category: Optional[torch.Tensor] = None,
                subcategory: Optional[torch.Tensor] = None,
                entities: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode news article
        
        Args:
            title: List of title strings
            abstract: List of abstract strings
            category: (batch_size,) category indices
            subcategory: (batch_size,) subcategory indices
            entities: (batch_size, num_entities) entity indices
            
        Returns:
            News embedding: (batch_size, final_dim)
        """
        # Text encoding
        text_emb = self.text_encoder(
            title, 
            abstract,
            max_title_len=self.config.max_title_len,
            max_abstract_len=self.config.max_abstract_len
        )  # (batch, news_encoder_dim)
        
        # Category embeddings
        if category is not None and subcategory is not None:
            cat_emb = self.category_embedding(category)  # (batch, cat_dim)
            subcat_emb = self.subcategory_embedding(subcategory)  # (batch, subcat_dim)
            cat_combined = torch.cat([cat_emb, subcat_emb], dim=1)  # (batch, cat_dim + subcat_dim)
        else:
            batch_size = text_emb.size(0)
            cat_dim = self.config.category_embedding_dim + self.config.subcategory_embedding_dim
            cat_combined = torch.zeros(batch_size, cat_dim, device=text_emb.device)
        
        # Combine text + category
        combined = torch.cat([text_emb, cat_combined], dim=1)  # (batch, text_dim + cat_dim)
        combined = self.fusion(combined)  # (batch, news_encoder_dim)
        
        # Entity fusion (if used)
        if self.config.use_entities and entities is not None:
            # Encode entities
            entity_emb = self.entity_embedding(entities)  # (batch, num_entities, entity_dim)
            
            # Project entity to match text dimension
            entity_proj = nn.Linear(
                self.config.entity_embedding_dim,
                self.config.news_encoder_dim
            ).to(combined.device)
            entity_emb = entity_proj(entity_emb)  # (batch, num_entities, news_encoder_dim)
            
            # Attention: text as query, entities as key/value
            text_query = combined.unsqueeze(1)  # (batch, 1, news_encoder_dim)
            entity_attended, _ = self.entity_attention(text_query, entity_emb, entity_emb)
            combined = combined + entity_attended.squeeze(1)  # Residual connection
        
        # Apply Fastformer-style attention
        combined = combined.unsqueeze(1)  # (batch, 1, dim) for FastAttention
        combined = self.fast_attention(combined)  # (batch, dim)
        
        # Final projection
        output = self.output_proj(combined)  # (batch, final_dim)
        
        return output

