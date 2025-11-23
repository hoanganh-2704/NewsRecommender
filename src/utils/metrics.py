"""Evaluation metrics for news recommendation"""
import numpy as np
from typing import List, Tuple


def calculate_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Area Under ROC Curve (AUC)
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted probabilities
        
    Returns:
        AUC score
    """
    from sklearn.metrics import roc_auc_score
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        return 0.0


def calculate_mrr(y_true: List[List[int]], y_pred: List[List[float]], k: int = 10) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR)
    
    Args:
        y_true: List of lists, each containing ground truth indices (clicked news)
        y_pred: List of lists, each containing predicted scores for all candidates
        k: Top-K to consider
        
    Returns:
        MRR score
    """
    reciprocal_ranks = []
    
    for true_indices, pred_scores in zip(y_true, y_pred):
        # Get top-k predicted indices
        top_k_indices = np.argsort(pred_scores)[::-1][:k]
        
        # Find rank of first clicked item
        rank = None
        for i, idx in enumerate(top_k_indices, start=1):
            if idx in true_indices:
                rank = i
                break
        
        if rank is not None:
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
    
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


def calculate_ndcg(y_true: List[List[int]], y_pred: List[List[float]], k: int = 10) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (nDCG@K)
    
    Args:
        y_true: List of lists, each containing ground truth indices (clicked news)
        y_pred: List of lists, each containing predicted scores for all candidates
        k: Top-K to consider
        
    Returns:
        nDCG@K score
    """
    ndcg_scores = []
    
    for true_indices, pred_scores in zip(y_true, y_pred):
        # Get top-k predicted indices
        top_k_indices = np.argsort(pred_scores)[::-1][:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, idx in enumerate(top_k_indices, start=1):
            if idx in true_indices:
                dcg += 1.0 / np.log2(i + 1)
        
        # Calculate IDCG (ideal DCG)
        num_relevant = min(len(true_indices), k)
        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, num_relevant + 1))
        
        # Calculate nDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)
    
    return np.mean(ndcg_scores) if ndcg_scores else 0.0


def evaluate_ranking(y_true: List[List[int]], y_pred: List[List[float]]) -> dict:
    """
    Calculate all ranking metrics
    
    Args:
        y_true: List of lists, each containing ground truth indices
        y_pred: List of lists, each containing predicted scores
        
    Returns:
        Dictionary with AUC, MRR, nDCG@5, nDCG@10
    """
    # Flatten for AUC calculation
    y_true_flat = []
    y_pred_flat = []
    for true_indices, pred_scores in zip(y_true, y_pred):
        for i, score in enumerate(pred_scores):
            y_true_flat.append(1 if i in true_indices else 0)
            y_pred_flat.append(score)
    
    auc = calculate_auc(np.array(y_true_flat), np.array(y_pred_flat))
    mrr = calculate_mrr(y_true, y_pred, k=10)
    ndcg5 = calculate_ndcg(y_true, y_pred, k=5)
    ndcg10 = calculate_ndcg(y_true, y_pred, k=10)
    
    return {
        "AUC": auc,
        "MRR": mrr,
        "nDCG@5": ndcg5,
        "nDCG@10": ndcg10
    }

