"""MIND Dataset Loader"""
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
import random


class MINDDataset(Dataset):
    """Dataset for MIND news recommendation"""
    
    def __init__(self, behaviors_file: str, news_file: str, 
                 max_history_len: int = 50, num_negatives: int = 4,
                 mode: str = "train"):
        """
        Args:
            behaviors_file: Path to behaviors.tsv
            news_file: Path to news.tsv
            max_history_len: Maximum history length
            num_negatives: Number of negative samples per positive
            mode: "train", "val", or "test"
        """
        self.max_history_len = max_history_len
        self.num_negatives = num_negatives
        self.mode = mode
        
        # Load news data
        print(f"Loading news data from {news_file}...")
        self.news_df = pd.read_csv(
            news_file,
            sep='\t',
            names=['news_id', 'category', 'subcategory', 'title', 'abstract', 
                   'url', 'title_entities', 'abstract_entities']
        )
        self.news_dict = {row['news_id']: row for _, row in self.news_df.iterrows()}
        
        # Load behaviors
        print(f"Loading behaviors from {behaviors_file}...")
        self.behaviors = []
        with open(behaviors_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                
                user_id = parts[0]
                history = parts[1].split() if parts[1] else []
                impressions = parts[2].split()
                
                # Parse impressions: news_id-label pairs
                clicked_news = []
                non_clicked_news = []
                for imp in impressions:
                    if '-' in imp:
                        news_id, label = imp.rsplit('-', 1)
                        if label == '1':
                            clicked_news.append(news_id)
                        else:
                            non_clicked_news.append(news_id)
                
                self.behaviors.append({
                    'user_id': user_id,
                    'history': history,
                    'clicked': clicked_news,
                    'non_clicked': non_clicked_news
                })
        
        print(f"Loaded {len(self.behaviors)} user behaviors")
        print(f"Total news articles: {len(self.news_dict)}")
    
    def __len__(self):
        return len(self.behaviors)
    
    def __getitem__(self, idx):
        behavior = self.behaviors[idx]
        
        # Get history
        history = behavior['history'][-self.max_history_len:]
        history_news = [self.news_dict.get(nid, None) for nid in history]
        history_news = [n for n in history_news if n is not None]
        
        # Get clicked news (positive samples)
        clicked = behavior['clicked']
        if not clicked:
            # If no clicked news, skip (or use a dummy)
            clicked = [list(self.news_dict.keys())[0]]
        
        # Sample negatives
        all_news_ids = list(self.news_dict.keys())
        negatives = random.sample(
            [nid for nid in all_news_ids if nid not in clicked and nid not in history],
            min(self.num_negatives, len(all_news_ids) - len(clicked))
        )
        
        # Create samples: for each clicked news, create (positive, negatives) pairs
        samples = []
        for pos_news_id in clicked[:1]:  # Limit to 1 positive per user for simplicity
            # Positive sample
            pos_news = self.news_dict.get(pos_news_id)
            if pos_news is None:
                continue
            
            # Negative samples
            neg_samples = []
            for neg_news_id in negatives:
                neg_news = self.news_dict.get(neg_news_id)
                if neg_news:
                    neg_samples.append(neg_news)
            
            samples.append({
                'positive': pos_news,
                'negatives': neg_samples,
                'history': history_news
            })
        
        if not samples:
            # Return dummy sample if no valid samples
            dummy_news = list(self.news_dict.values())[0]
            return {
                'positive': dummy_news,
                'negatives': [dummy_news] * self.num_negatives,
                'history': []
            }
        
        return samples[0]  # Return first sample
    
    def get_news_by_id(self, news_id: str):
        """Get news article by ID"""
        return self.news_dict.get(news_id)


def collate_fn(batch):
    """Custom collate function for MIND dataset"""
    positives = [item['positive'] for item in batch]
    negatives_list = [item['negatives'] for item in batch]
    histories = [item['history'] for item in batch]
    
    # Flatten: create (positive + negatives) samples
    samples = []
    labels = []
    
    for i, (pos, negs) in enumerate(zip(positives, negatives_list)):
        # Positive
        samples.append({
            'news': pos,
            'history': histories[i]
        })
        labels.append(1)
        
        # Negatives
        for neg in negs:
            samples.append({
                'news': neg,
                'history': histories[i]
            })
            labels.append(0)
    
    return {
        'samples': samples,
        'labels': torch.tensor(labels, dtype=torch.float32)
    }

