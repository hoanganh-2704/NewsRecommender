import sys
import os
sys.path.append(os.path.join(os.getcwd(), "src"))

import torch
from src.data.text_encoder import BERTTextEncoder

class MockConfig:
    def __init__(self):
        self.bert_model_name = "bert-base-uncased"
        self.news_encoder_dim = 256

def verify_text_encoder():
    print("Initializing BERTTextEncoder...")
    # Use a small output dim to test projection
    encoder = BERTTextEncoder(model_name="bert-base-uncased", output_dim=256)
    
    print("Running forward pass...")
    # Dummy data
    titles = ["This is a title"] * 2
    abstracts = ["This is an abstract"] * 2
    
    try:
        output = encoder(titles, abstracts)
        print(f"Output shape: {output.shape}")
        
        expected_shape = (2, 256)
        if output.shape == expected_shape:
            print("SUCCESS: Output shape matches expected shape.")
        else:
            print(f"FAILURE: Expected {expected_shape}, got {output.shape}")
            
    except Exception as e:
        print(f"FAILURE: Exception during forward pass: {e}")

if __name__ == "__main__":
    verify_text_encoder()
