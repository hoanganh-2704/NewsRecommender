import sys
import os
sys.path.append(os.path.join(os.getcwd(), "src"))

from src.data.mind_dataset import MINDDataset
import torch
import time

def verify_real():
    print("Initializing MINDDataset with real data...")
    start_time = time.time()
    
    # Path to real data
    data_dir = "data/MINDsmall_train"
    behaviors_file = os.path.join(data_dir, "behaviors.tsv")
    news_file = os.path.join(data_dir, "news.tsv")
    
    if not os.path.exists(behaviors_file) or not os.path.exists(news_file):
        print(f"Error: Data files not found in {data_dir}")
        return

    dataset = MINDDataset(
        behaviors_file=behaviors_file,
        news_file=news_file,
        max_history_len=50,
        num_negatives=4,
        use_hard_negatives=True
    )
    
    print(f"Dataset loaded in {time.time() - start_time:.2f} seconds")
    print(f"Total behaviors: {len(dataset)}")
    print(f"Total news: {len(dataset.news_dict)}")
    print(f"Total categories: {len(dataset.cat_processor.cat2id)}")
    print(f"Total subcategories: {len(dataset.cat_processor.subcat2id)}")
    
    # Check a sample
    print("\nChecking a random sample...")
    sample = dataset[0]
    
    pos = sample['positive']
    print(f"Positive News Title: {pos['title'][:50]}...")
    print(f"Category ID: {pos['category']}")
    print(f"Subcategory ID: {pos['subcategory']}")
    
    negs = sample['negatives']
    print(f"Number of negatives: {len(negs)}")
    
    # Check for hard negatives (same category)
    pos_cat = pos['category']
    hard_negs_count = sum(1 for n in negs if n['category'] == pos_cat)
    print(f"Negatives with same category as positive: {hard_negs_count}/{len(negs)}")
    
    print("\nVerification on real dataset completed successfully.")

if __name__ == "__main__":
    verify_real()
