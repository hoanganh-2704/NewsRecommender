"""Hybrid Model: Complete News Recommendation Model"""
import torch
import torch.nn as nn
from src.models.news_encoder import NewsEncoder
from src.models.user_encoder import UserEncoder


class HybridNewsRecommendationModel(nn.Module):
    """Complete Hybrid Model for News Recommendation"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # News Encoder
        self.news_encoder = NewsEncoder(config)
        
        # User Encoder
        self.user_encoder = UserEncoder(config)
        
        # Prediction layer (dot product + sigmoid)
        # No additional parameters needed for dot product
    
    def encode_news(self, title: list, abstract: list,
                   category: torch.Tensor = None,
                   subcategory: torch.Tensor = None,
                   entities: torch.Tensor = None) -> torch.Tensor:
        """
        Encode news articles
        
        Args:
            title: List of title strings
            abstract: List of abstract strings
            category: (batch_size,) category indices
            subcategory: (batch_size,) subcategory indices
            entities: (batch_size, num_entities) entity indices
            
        Returns:
            News embeddings: (batch_size, final_dim)
        """
        return self.news_encoder(title, abstract, category, subcategory, entities)
    
    def encode_user(self, history_embeddings: torch.Tensor,
                   candidate_embedding: torch.Tensor,
                   mask: torch.Tensor = None) -> torch.Tensor:
        """
        Encode user based on history
        
        Args:
            history_embeddings: (batch_size, history_len, final_dim)
            candidate_embedding: (batch_size, final_dim)
            mask: (batch_size, history_len) padding mask
            
        Returns:
            User representation: (batch_size, final_dim)
        """
        return self.user_encoder(history_embeddings, candidate_embedding, mask)
    
    def forward(self, 
                # News inputs
                candidate_title: list,
                candidate_abstract: list,
                candidate_category: torch.Tensor = None,
                candidate_subcategory: torch.Tensor = None,
                candidate_entities: torch.Tensor = None,
                # User inputs
                history_titles: list,
                history_abstracts: list,
                history_categories: torch.Tensor = None,
                history_subcategories: torch.Tensor = None,
                history_entities: torch.Tensor = None,
                history_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass: predict click probability
        
        Args:
            candidate_title: List of candidate news titles
            candidate_abstract: List of candidate news abstracts
            candidate_category: (batch_size,) category indices
            candidate_subcategory: (batch_size,) subcategory indices
            candidate_entities: (batch_size, num_entities) entity indices
            history_titles: List of lists, history news titles
            history_abstracts: List of lists, history news abstracts
            history_categories: (batch_size, history_len) category indices
            history_subcategories: (batch_size, history_len) subcategory indices
            history_entities: (batch_size, history_len, num_entities) entity indices
            history_mask: (batch_size, history_len) padding mask
            
        Returns:
            Click probabilities: (batch_size,)
        """
        # Encode candidate news
        candidate_emb = self.news_encoder(
            candidate_title,
            candidate_abstract,
            candidate_category,
            candidate_subcategory,
            candidate_entities
        )  # (batch, final_dim)
        
        # Encode history news
        batch_size = len(history_titles)
        history_len = self.config.max_history_len
        history_embs = []
        
        for i in range(batch_size):
            # Encode each news in history
            hist_titles = history_titles[i] if isinstance(history_titles[i], list) else [history_titles[i]]
            hist_abstracts = history_abstracts[i] if isinstance(history_abstracts[i], list) else [history_abstracts[i]]
            
            # Pad or truncate
            if len(hist_titles) < history_len:
                hist_titles = hist_titles + [""] * (history_len - len(hist_titles))
                hist_abstracts = hist_abstracts + [""] * (history_len - len(hist_abstracts))
            else:
                hist_titles = hist_titles[:history_len]
                hist_abstracts = hist_abstracts[:history_len]
            
            # Encode
            hist_emb = self.news_encoder(
                hist_titles,
                hist_abstracts,
                history_categories[i] if history_categories is not None else None,
                history_subcategories[i] if history_subcategories is not None else None,
                history_entities[i] if history_entities is not None else None
            )  # (history_len, final_dim)
            history_embs.append(hist_emb)
        
        history_embeddings = torch.stack(history_embs, dim=0)  # (batch, history_len, final_dim)
        
        # Encode user
        user_emb = self.user_encoder(
            history_embeddings,
            candidate_emb,
            history_mask
        )  # (batch, final_dim)
        
        # Prediction: dot product + sigmoid
        scores = (user_emb * candidate_emb).sum(dim=1)  # (batch,)
        probabilities = torch.sigmoid(scores)  # (batch,)
        
        return probabilities

