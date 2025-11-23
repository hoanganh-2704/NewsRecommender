"""Configuration file for News Recommendation System"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for model architecture"""
    # Text encoding
    text_encoder_type: str = "bert"  # "bert" or "glove"
    bert_model_name: str = "bert-base-uncased"
    glove_dim: int = 300
    max_title_len: int = 30
    max_abstract_len: int = 100
    
    # Entity processing
    use_entities: bool = True
    entity_embedding_dim: int = 100
    
    # Category embedding
    category_embedding_dim: int = 50
    subcategory_embedding_dim: int = 50
    
    # News encoder
    news_encoder_dim: int = 768  # Should match text encoder output
    news_encoder_num_heads: int = 8
    news_encoder_num_layers: int = 2
    news_encoder_dropout: float = 0.2
    
    # User encoder
    max_history_len: int = 50
    num_interests: int = 5  # K in multi-interest modeling
    user_encoder_dim: int = 768  # Should match news_encoder_dim
    user_encoder_dropout: float = 0.2
    
    # Prediction
    final_dim: int = 768


@dataclass
class TrainingConfig:
    """Configuration for training"""
    batch_size: int = 64
    num_epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    dropout: float = 0.2
    
    # Negative sampling
    num_negatives: int = 4
    use_hard_negatives: bool = True
    
    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_steps: int = 1000
    
    # Regularization
    early_stopping_patience: int = 5
    gradient_clip: float = 1.0
    
    # Logging
    log_interval: int = 100
    eval_interval: int = 1  # Evaluate every N epochs
    save_interval: int = 1  # Save checkpoint every N epochs


@dataclass
class DataConfig:
    """Configuration for data processing"""
    data_dir: str = "data"
    train_file: str = "train/behaviors.tsv"
    val_file: str = "val/behaviors.tsv"
    test_file: str = "test/behaviors.tsv"
    news_file: str = "news.tsv"
    
    # Processing
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True
    
    # Cache
    cache_dir: str = "cache"
    use_cache: bool = True


@dataclass
class Config:
    """Main configuration class"""
    model: ModelConfig = None
    training: TrainingConfig = None
    data: DataConfig = None
    
    # Paths
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Device
    device: str = "cuda"  # "cuda" or "cpu"
    seed: int = 42
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.data is None:
            self.data = DataConfig()
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.data.cache_dir, exist_ok=True)

