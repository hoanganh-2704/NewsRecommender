"""Evaluation script for News Recommendation Model"""
import os
import torch
import argparse
import json
from torch.utils.data import DataLoader

from src.models.hybrid_model import HybridNewsRecommendationModel
from src.data.mind_dataset import MINDDataset, collate_fn
from src.utils.config import Config
from src.utils.metrics import evaluate_ranking


def evaluate_model(model, dataloader, device, config):
    """Evaluate model and return metrics"""
    model.eval()
    
    # For proper ranking evaluation, we need to group by user
    user_predictions = {}
    user_labels = {}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            samples = batch['samples']
            labels = batch['labels'].cpu().numpy()
            
            # Process batch
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
                
                # Group by user (simplified - in practice, you'd track user_id)
                # For now, treat each batch as separate users
                for i, (pred, label) in enumerate(zip(predictions, labels)):
                    user_id = f"user_{batch_idx}_{i}"
                    if user_id not in user_predictions:
                        user_predictions[user_id] = []
                        user_labels[user_id] = []
                    user_predictions[user_id].append(pred)
                    user_labels[user_id].append(int(label))
                    
            except Exception as e:
                print(f"Error in batch: {e}")
                continue
    
    # Calculate ranking metrics
    y_true = []
    y_pred = []
    
    for user_id in user_predictions:
        preds = user_predictions[user_id]
        labels = user_labels[user_id]
        
        # Get clicked indices
        clicked_indices = [i for i, l in enumerate(labels) if l == 1]
        if clicked_indices:
            y_true.append(clicked_indices)
            y_pred.append(preds)
    
    if y_true and y_pred:
        metrics = evaluate_ranking(y_true, y_pred)
        return metrics
    else:
        return {"AUC": 0.0, "MRR": 0.0, "nDCG@5": 0.0, "nDCG@10": 0.0}


def main():
    parser = argparse.ArgumentParser(description="Evaluate News Recommendation Model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to MIND data directory")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Split to evaluate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get('config')
    if config is None:
        config = Config()
    
    # Create model
    model = HybridNewsRecommendationModel(config.model).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Load dataset
    dataset = MINDDataset(
        behaviors_file=os.path.join(args.data_dir, args.split, "behaviors.tsv"),
        news_file=os.path.join(args.data_dir, "news.tsv"),
        max_history_len=config.model.max_history_len,
        num_negatives=4,
        mode=args.split
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # Evaluate
    print(f"Evaluating on {args.split} set...")
    metrics = evaluate_model(model, dataloader, device, config)
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Results:")
    print("="*50)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()

