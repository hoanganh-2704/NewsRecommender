"""Training script for News Recommendation Model"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import json
from pathlib import Path

from src.models.hybrid_model import HybridNewsRecommendationModel
from src.data.mind_dataset import MINDDataset, collate_fn
from src.utils.config import Config, ModelConfig, TrainingConfig, DataConfig
from src.utils.metrics import evaluate_ranking


def train_epoch(model, dataloader, criterion, optimizer, device, config):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        samples = batch['samples']
        labels = batch['labels'].to(device)
        
        # Prepare inputs
        candidate_titles = [s['news']['title'] for s in samples]
        candidate_abstracts = [s['news']['abstract'] for s in samples]
        histories = [s['history'] for s in samples]
        
        # Extract history titles and abstracts
        history_titles = []
        history_abstracts = []
        for hist in histories:
            hist_titles = [h['title'] if isinstance(h, dict) else '' for h in hist[:config.model.max_history_len]]
            hist_abstracts = [h['abstract'] if isinstance(h, dict) else '' for h in hist[:config.model.max_history_len]]
            # Pad
            while len(hist_titles) < config.model.max_history_len:
                hist_titles.append('')
                hist_abstracts.append('')
            history_titles.append(hist_titles)
            history_abstracts.append(hist_abstracts)
        
        # Forward pass
        optimizer.zero_grad()
        
        # Note: This is a simplified version. In practice, you'd need to handle
        # batching more carefully for the model's forward method
        try:
            # For now, process in smaller batches
            batch_size = len(samples)
            predictions = []
            
            for i in range(batch_size):
                # Single sample prediction
                pred = model(
                    candidate_title=[candidate_titles[i]],
                    candidate_abstract=[candidate_abstracts[i]],
                    history_titles=[history_titles[i]],
                    history_abstracts=[history_abstracts[i]]
                )
                predictions.append(pred.squeeze())
            
            predictions = torch.stack(predictions)
            
            # Calculate loss
            loss = criterion(predictions, labels)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
            
        except Exception as e:
            print(f"Error in batch: {e}")
            continue
    
    return total_loss / max(num_batches, 1)


def validate(model, dataloader, device, config):
    """Validate model"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            samples = batch['samples']
            labels = batch['labels'].cpu().numpy()
            
            # Similar processing as training
            candidate_titles = [s['news']['title'] for s in samples]
            candidate_abstracts = [s['news']['abstract'] for s in samples]
            histories = [s['history'] for s in samples]
            
            history_titles = []
            history_abstracts = []
            for hist in histories:
                hist_titles = [h['title'] if isinstance(h, dict) else '' for h in hist[:config.model.max_history_len]]
                hist_abstracts = [h['abstract'] if isinstance(h, dict) else '' for h in hist[:config.model.max_history_len]]
                while len(hist_titles) < config.model.max_history_len:
                    hist_titles.append('')
                    hist_abstracts.append('')
                history_titles.append(hist_titles)
                history_abstracts.append(hist_abstracts)
            
            try:
                predictions = []
                for i in range(len(samples)):
                    pred = model(
                        candidate_title=[candidate_titles[i]],
                        candidate_abstract=[candidate_abstracts[i]],
                        history_titles=[history_titles[i]],
                        history_abstracts=[history_abstracts[i]]
                    )
                    predictions.append(pred.squeeze().cpu().item())
                
                all_predictions.extend(predictions)
                all_labels.extend(labels.tolist())
            except Exception as e:
                print(f"Error in validation batch: {e}")
                continue
    
    # Calculate metrics (simplified - would need proper ranking evaluation)
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(all_labels, all_predictions)
    except:
        auc = 0.0
    
    return auc, all_predictions, all_labels


def main():
    parser = argparse.ArgumentParser(description="Train News Recommendation Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to MIND data directory")
    parser.add_argument("--config", type=str, help="Path to config file (JSON)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()
    
    # Load config
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = Config(**config_dict)
    else:
        # Default config
        config = Config()
        config.data.data_dir = args.data_dir
        config.output_dir = args.output_dir
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Load datasets
    train_dataset = MINDDataset(
        behaviors_file=os.path.join(args.data_dir, "train", "behaviors.tsv"),
        news_file=os.path.join(args.data_dir, "news.tsv"),
        max_history_len=config.model.max_history_len,
        num_negatives=config.training.num_negatives,
        mode="train"
    )
    
    val_dataset = MINDDataset(
        behaviors_file=os.path.join(args.data_dir, "val", "behaviors.tsv"),
        news_file=os.path.join(args.data_dir, "news.tsv"),
        max_history_len=config.model.max_history_len,
        num_negatives=config.training.num_negatives,
        mode="val"
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=config.data.pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=config.data.pin_memory
    )
    
    # Create model
    model = HybridNewsRecommendationModel(config.model).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # Scheduler
    if config.training.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.num_epochs
        )
    else:
        scheduler = None
    
    # Training loop
    best_auc = 0.0
    patience_counter = 0
    
    for epoch in range(1, config.training.num_epochs + 1):
        print(f"\nEpoch {epoch}/{config.training.num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, config)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        if epoch % config.training.eval_interval == 0:
            val_auc, _, _ = validate(model, val_loader, device, config)
            print(f"Validation AUC: {val_auc:.4f}")
            
            # Save best model
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                checkpoint_path = os.path.join(config.checkpoint_dir, "best_model.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'auc': val_auc,
                    'config': config
                }, checkpoint_path)
                print(f"Saved best model (AUC: {best_auc:.4f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= config.training.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Scheduler step
        if scheduler:
            scheduler.step()
    
    print(f"\nTraining completed. Best AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()

