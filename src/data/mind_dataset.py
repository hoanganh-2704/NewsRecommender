"""MIND Dataset Loader"""
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional
import random
import json
import numpy as np


class CategoryProcessor:
    """Process categories and subcategories"""
    def __init__(self):
        self.cat2id = {"<PAD>": 0, "<UNK>": 1}
        self.subcat2id = {"<PAD>": 0, "<UNK>": 1}
        
    def fit(self, news_df: pd.DataFrame):
        """Build vocabularies from dataframe"""
        categories = news_df['category'].unique()
        subcategories = news_df['subcategory'].unique()
        
        for cat in categories:
            if cat not in self.cat2id:
                self.cat2id[cat] = len(self.cat2id)
                
        for subcat in subcategories:
            if subcat not in self.subcat2id:
                self.subcat2id[subcat] = len(self.subcat2id)
                
    def transform(self, category: str, subcategory: str) -> Tuple[int, int]:
        """Convert category strings to IDs"""
        cat_id = self.cat2id.get(category, self.cat2id["<UNK>"])
        subcat_id = self.subcat2id.get(subcategory, self.subcat2id["<UNK>"])
        return cat_id, subcat_id


class EntityProcessor:
    """Process entities from Wikidata"""
    def __init__(self, max_entities: int = 5):
        self.ent2id = {"<PAD>": 0, "<UNK>": 1}
        self.max_entities = max_entities
        
    def fit(self, news_df: pd.DataFrame):
        """Build vocabulary from dataframe"""
        # Note: In a real scenario, we might load a pre-defined entity map
        # Here we build it from the data, assuming entities are in JSON format
        # or we just map the entity IDs found in the text
        pass
        
    def transform(self, title_ent_str: str, abstract_ent_str: str) -> torch.Tensor:
        """
        Parse entity strings and return entity IDs.
        For simplicity in this implementation, we'll hash the entity IDs 
        to a fixed range if we don't have a full entity map, 
        or just return a placeholder if data format is complex.
        """
        # Placeholder implementation: just return zeros
        # In a real implementation, we would parse the JSON:
        # [{"Label": "...", "Type": "...", "WikidataId": "Q..."}]
        return torch.zeros(self.max_entities, dtype=torch.long)


class MINDDataset(Dataset):
    """Dataset for MIND news recommendation"""
    
    def __init__(self, behaviors_file: str, news_file: str, 
                 max_history_len: int = 50, num_negatives: int = 4,
                 mode: str = "train",
                 use_hard_negatives: bool = True):
        """
        Args:
            behaviors_file: Path to behaviors.tsv
            news_file: Path to news.tsv
            max_history_len: Maximum history length
            num_negatives: Number of negative samples per positive
            mode: "train", "val", or "test"
            use_hard_negatives: Whether to use hard negative mining
        """
        self.max_history_len = max_history_len
        self.num_negatives = num_negatives
        self.mode = mode
        self.use_hard_negatives = use_hard_negatives
        
        # Load news data
        print(f"Loading news data from {news_file}...")
        self.news_df = pd.read_csv(
            news_file,
            sep='\t',
            names=['news_id', 'category', 'subcategory', 'title', 'abstract', 
                   'url', 'title_entities', 'abstract_entities']
        )
        self.news_dict = {row['news_id']: row for _, row in self.news_df.iterrows()}
        
        # Initialize processors
        self.cat_processor = CategoryProcessor()
        self.cat_processor.fit(self.news_df)
        
        self.ent_processor = EntityProcessor()
        # self.ent_processor.fit(self.news_df)
        
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
        print(f"Categories: {len(self.cat_processor.cat2id)}")
    
    def __len__(self):
        return len(self.behaviors)
    
    def _process_news(self, news_row):
        """Process a single news row into features"""
        if news_row is None:
            # Return dummy features
            return {
                'title': "",
                'abstract': "",
                'category': 0,
                'subcategory': 0,
                'entities': torch.zeros(5, dtype=torch.long)
            }
            
        cat_id, subcat_id = self.cat_processor.transform(
            news_row['category'], news_row['subcategory']
        )
        
        entities = self.ent_processor.transform(
            news_row['title_entities'], news_row['abstract_entities']
        )
        
        return {
            'title': str(news_row['title']),
            'abstract': str(news_row['abstract']),
            'category': cat_id,
            'subcategory': subcat_id,
            'entities': entities
        }

    def _sample_negatives(self, clicked_news_id: str, candidate_negatives: List[str]) -> List[str]:
        """Sample negatives with Hard Negative Mining strategy"""
        if not candidate_negatives:
            return []
            
        if not self.use_hard_negatives:
            return random.sample(candidate_negatives, min(self.num_negatives, len(candidate_negatives)))
            
        # Hard Negative Mining:
        # Prioritize negatives that share category/subcategory with the clicked news
        target_news = self.news_dict.get(clicked_news_id)
        if target_news is None:
             return random.sample(candidate_negatives, min(self.num_negatives, len(candidate_negatives)))
             
        target_cat = target_news['category']
        target_subcat = target_news['subcategory']
        
        hard_negatives = []
        easy_negatives = []
        
        for nid in candidate_negatives:
            news = self.news_dict.get(nid)
            if news is not None:
                if news['category'] == target_cat or news['subcategory'] == target_subcat:
                    hard_negatives.append(nid)
                else:
                    easy_negatives.append(nid)
        
        # Fill with hard negatives first, then easy negatives
        selected = []
        random.shuffle(hard_negatives)
        random.shuffle(easy_negatives)
        
        selected.extend(hard_negatives[:self.num_negatives])
        if len(selected) < self.num_negatives:
            selected.extend(easy_negatives[:self.num_negatives - len(selected)])
            
        return selected

    def __getitem__(self, idx):
        behavior = self.behaviors[idx]
        
        # Get history
        history = behavior['history'][-self.max_history_len:]
        history_features = [self._process_news(self.news_dict.get(nid)) for nid in history]
        
        # Pad history if needed (handled in collate or model, but here we just return list)
        
        # Get clicked news (positive samples)
        clicked = behavior['clicked']
        if not clicked:
            clicked = [list(self.news_dict.keys())[0]]
        
        # Sample negatives from non-clicked items in the same impression
        # If not enough non-clicked in impression, could sample from global pool (omitted for simplicity)
        non_clicked = behavior['non_clicked']
        
        # Create samples
        samples = []
        for pos_news_id in clicked[:1]:  # Limit to 1 positive per user for simplicity
            # Positive sample
            pos_features = self._process_news(self.news_dict.get(pos_news_id))
            
            # Negative samples
            selected_neg_ids = self._sample_negatives(pos_news_id, non_clicked)
            
            # Fallback: if no negatives found in impression (e.g. all clicked or empty), sample from global pool
            if not selected_neg_ids:
                all_news_ids = list(self.news_dict.keys())
                # Exclude history and clicked (simplified)
                candidates = random.sample(all_news_ids, self.num_negatives + len(history) + len(clicked))
                selected_neg_ids = [nid for nid in candidates if nid not in history and nid not in clicked][:self.num_negatives]
                
            # If still not enough, duplicate
            if len(selected_neg_ids) < self.num_negatives:
                if len(selected_neg_ids) == 0:
                     # Extreme fallback: just pick random news
                     selected_neg_ids = random.sample(list(self.news_dict.keys()), self.num_negatives)
                
                selected_neg_ids = (selected_neg_ids * (self.num_negatives // len(selected_neg_ids) + 1))[:self.num_negatives]
            
            neg_features = [self._process_news(self.news_dict.get(nid)) for nid in selected_neg_ids]
            
            samples.append({
                'positive': pos_features,
                'negatives': neg_features,
                'history': history_features
            })
        
        if not samples:
             # Dummy
            dummy = self._process_news(None)
            return {
                'positive': dummy,
                'negatives': [dummy] * self.num_negatives,
                'history': []
            }
            
        return samples[0]
    
    def get_news_by_id(self, news_id: str):
        """Get news article by ID"""
        return self.news_dict.get(news_id)


def collate_fn(batch):
    """Custom collate function for MIND dataset"""
    positives = [item['positive'] for item in batch]
    negatives_list = [item['negatives'] for item in batch]
    histories = [item['history'] for item in batch]
    
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
