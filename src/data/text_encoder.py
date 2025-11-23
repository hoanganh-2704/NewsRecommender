"""Text encoding module for news articles"""
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from typing import Optional, Tuple


class BERTTextEncoder(nn.Module):
    """BERT-based text encoder"""
    
    def __init__(self, model_name: str = "bert-base-uncased", output_dim: int = 768):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.output_dim = output_dim
        
        # Projection layer if needed
        if output_dim != 768:
            self.projection = nn.Linear(768, output_dim)
        else:
            self.projection = nn.Identity()
    
    def encode(self, texts: list, max_length: int = 100) -> torch.Tensor:
        """
        Encode list of texts
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            
        Returns:
            Tensor of shape (batch_size, output_dim)
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Move to device
        device = next(self.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # Get BERT embeddings
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.bert(**encoded)
            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        # Project if needed
        embeddings = self.projection(embeddings)
        
        return embeddings
    
    def forward(self, title: list, abstract: list, 
                max_title_len: int = 30, max_abstract_len: int = 100) -> torch.Tensor:
        """
        Encode title and abstract, then combine
        
        Args:
            title: List of title strings
            abstract: List of abstract strings
            max_title_len: Max length for title
            max_abstract_len: Max length for abstract
            
        Returns:
            Combined text embedding (batch_size, output_dim)
        """
        title_emb = self.encode(title, max_length=max_title_len)
        abstract_emb = self.encode(abstract, max_length=max_abstract_len)
        
        # Combine: simple concatenation or weighted sum
        # Option 1: Concatenate and project
        combined = torch.cat([title_emb, abstract_emb], dim=1)
        if combined.size(1) != self.output_dim:
            combined = self.projection(combined)
        
        # Option 2: Weighted sum (uncomment to use)
        # combined = 0.6 * title_emb + 0.4 * abstract_emb
        
        return combined


class GloVeTextEncoder(nn.Module):
    """GloVe-based text encoder (simplified version)"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 300, 
                 pretrained_embeddings: Optional[torch.Tensor] = None):
        super().__init__()
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.embedding_dim = embedding_dim
        self.lstm = nn.LSTM(embedding_dim, embedding_dim // 2, 
                           num_layers=1, batch_first=True, bidirectional=True)
        self.projection = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, title_ids: torch.Tensor, abstract_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode title and abstract
        
        Args:
            title_ids: (batch_size, title_len)
            abstract_ids: (batch_size, abstract_len)
            
        Returns:
            Combined embedding (batch_size, embedding_dim)
        """
        # Embed
        title_emb = self.embedding(title_ids)  # (batch, title_len, emb_dim)
        abstract_emb = self.embedding(abstract_ids)  # (batch, abstract_len, emb_dim)
        
        # LSTM encoding
        title_out, _ = self.lstm(title_emb)
        abstract_out, _ = self.lstm(abstract_emb)
        
        # Pooling: use last hidden state
        title_pooled = title_out[:, -1, :]  # (batch, emb_dim)
        abstract_pooled = abstract_out[:, -1, :]  # (batch, emb_dim)
        
        # Combine
        combined = torch.cat([title_pooled, abstract_pooled], dim=1)
        combined = self.projection(combined)
        
        return combined


def create_text_encoder(config) -> nn.Module:
    """Factory function to create text encoder based on config"""
    if config.text_encoder_type == "bert":
        return BERTTextEncoder(
            model_name=config.bert_model_name,
            output_dim=config.news_encoder_dim
        )
    elif config.text_encoder_type == "glove":
        # Note: In practice, you'd load GloVe embeddings here
        return GloVeTextEncoder(
            vocab_size=50000,  # Should be set based on actual vocabulary
            embedding_dim=config.glove_dim
        )
    else:
        raise ValueError(f"Unknown text encoder type: {config.text_encoder_type}")

