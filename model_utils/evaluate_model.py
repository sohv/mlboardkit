#!/usr/bin/env python3
"""
evaluate_model.py

Standardized evaluation metrics for ML models (accuracy, F1, BLEU, etc.).
Simple and focused evaluation utility.
"""

import argparse
import json
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from collections import Counter

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, confusion_matrix, roc_auc_score
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def accuracy(predictions: List, targets: List) -> float:
    """Calculate accuracy"""
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have same length")
    
    correct = sum(1 for p, t in zip(predictions, targets) if p == t)
    return correct / len(predictions)


def precision_recall_f1(predictions: List, targets: List, average: str = 'weighted') -> Dict[str, float]:
    """Calculate precision, recall, and F1 score"""
    if HAS_SKLEARN:
        precision = precision_score(targets, predictions, average=average, zero_division=0)
        recall = recall_score(targets, predictions, average=average, zero_division=0)
        f1 = f1_score(targets, predictions, average=average, zero_division=0)
    else:
        # Simple binary case implementation
        if average == 'binary' or len(set(targets)) == 2:
            tp = sum(1 for p, t in zip(predictions, targets) if p == 1 and t == 1)
            fp = sum(1 for p, t in zip(predictions, targets) if p == 1 and t == 0)
            fn = sum(1 for p, t in zip(predictions, targets) if p == 0 and t == 1)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        else:
            # Fallback to accuracy for multi-class
            acc = accuracy(predictions, targets)
            precision = recall = f1 = acc
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def bleu_score(predictions: List[str], references: List[str], max_n: int = 4) -> float:
    """Calculate BLEU score for text generation"""
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have same length")
    
    def get_ngrams(tokens: List[str], n: int) -> Counter:
        return Counter([tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)])
    
    def modified_precision(pred_tokens: List[str], ref_tokens: List[str], n: int) -> float:
        pred_ngrams = get_ngrams(pred_tokens, n)
        ref_ngrams = get_ngrams(ref_tokens, n)
        
        if not pred_ngrams:
            return 0.0
        
        overlap = 0
        for ngram, count in pred_ngrams.items():
            overlap += min(count, ref_ngrams.get(ngram, 0))
        
        return overlap / sum(pred_ngrams.values())
    
    # Calculate for all predictions
    precisions = [[] for _ in range(max_n)]
    total_pred_len = 0
    total_ref_len = 0
    
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        
        total_pred_len += len(pred_tokens)
        total_ref_len += len(ref_tokens)
        
        for n in range(1, max_n + 1):
            prec = modified_precision(pred_tokens, ref_tokens, n)
            precisions[n-1].append(prec)
    
    # Average precisions
    avg_precisions = []
    for prec_list in precisions:
        if prec_list:
            avg_precisions.append(sum(prec_list) / len(prec_list))
        else:
            avg_precisions.append(0.0)
    
    # Brevity penalty
    bp = 1.0
    if total_pred_len < total_ref_len:
        bp = math.exp(1 - total_ref_len / total_pred_len) if total_pred_len > 0 else 0
    
    # BLEU score
    if all(p > 0 for p in avg_precisions):
        log_precisions = [math.log(p) for p in avg_precisions]
        bleu = math.exp(sum(log_precisions) / len(log_precisions))
        return bp * bleu
    else:
        return 0.0


def rouge_l(predictions: List[str], references: List[str]) -> float:
    """Calculate ROUGE-L score"""
    
    def lcs_length(x: List[str], y: List[str]) -> int:
        """Longest Common Subsequence length"""
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    total_score = 0
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        
        lcs_len = lcs_length(pred_tokens, ref_tokens)
        
        if len(pred_tokens) == 0 and len(ref_tokens) == 0:
            score = 1.0
        elif len(pred_tokens) == 0 or len(ref_tokens) == 0:
            score = 0.0
        else:
            precision = lcs_len / len(pred_tokens)
            recall = lcs_len / len(ref_tokens)
            score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        total_score += score
    
    return total_score / len(predictions)


def perplexity(log_probabilities: List[float]) -> float:
    """Calculate perplexity from log probabilities"""
    if not log_probabilities:
        return float('inf')
    
    avg_log_prob = sum(log_probabilities) / len(log_probabilities)
    return math.exp(-avg_log_prob)


def load_predictions_and_targets(pred_file: str, target_file: str = None,
                                pred_key: str = 'prediction', target_key: str = 'target'):
    """Load predictions and targets from files"""
    
    def load_data(file_path: str):
        path = Path(file_path)
        if path.suffix.lower() == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif path.suffix.lower() == '.jsonl':
            data = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
            return data
        else:
            with open(path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
    
    pred_data = load_data(pred_file)
    
    if target_file:
        target_data = load_data(target_file)
    else:
        target_data = pred_data  # Assume same file has both
    
    # Extract predictions and targets
    predictions = []
    targets = []
    
    if isinstance(pred_data[0], dict):
        for item in pred_data:
            predictions.append(item.get(pred_key, item))
    else:
        predictions = pred_data
    
    if isinstance(target_data[0], dict):
        for item in target_data:
            targets.append(item.get(target_key, item))
    else:
        targets = target_data
    
    return predictions, targets


def evaluate_classification(predictions: List, targets: List) -> Dict[str, Any]:
    """Evaluate classification metrics"""
    results = {
        'accuracy': accuracy(predictions, targets),
        **precision_recall_f1(predictions, targets)
    }
    
    if HAS_SKLEARN:
        # Add detailed classification report
        try:
            report = classification_report(targets, predictions, output_dict=True)
            results['classification_report'] = report
            
            # Add confusion matrix
            cm = confusion_matrix(targets, predictions)
            results['confusion_matrix'] = cm.tolist()
            
            # Add AUC if binary classification
            unique_labels = list(set(targets))
            if len(unique_labels) == 2:
                try:
                    auc = roc_auc_score(targets, predictions)
                    results['auc'] = auc
                except:
                    pass  # Skip if can't calculate AUC
                    
        except Exception as e:
            print(f"Warning: Could not compute detailed metrics: {e}")
    
    return results


def evaluate_generation(predictions: List[str], targets: List[str]) -> Dict[str, Any]:
    """Evaluate text generation metrics"""
    results = {
        'bleu': bleu_score(predictions, targets),
        'rouge_l': rouge_l(predictions, targets)
    }
    
    # Add length statistics
    pred_lengths = [len(pred.split()) for pred in predictions]
    target_lengths = [len(target.split()) for target in targets]
    
    results['avg_pred_length'] = sum(pred_lengths) / len(pred_lengths)
    results['avg_target_length'] = sum(target_lengths) / len(target_lengths)
    results['length_ratio'] = results['avg_pred_length'] / results['avg_target_length']
    
    return results


def evaluate_regression(predictions: List[float], targets: List[float]) -> Dict[str, Any]:
    """Evaluate regression metrics"""
    if HAS_NUMPY:
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        
        # R-squared
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
    else:
        # Manual calculation
        n = len(predictions)
        mse = sum((p - t) ** 2 for p, t in zip(predictions, targets)) / n
        rmse = math.sqrt(mse)
        mae = sum(abs(p - t) for p, t in zip(predictions, targets)) / n
        
        mean_target = sum(targets) / len(targets)
        ss_res = sum((t - p) ** 2 for p, t in zip(predictions, targets))
        ss_tot = sum((t - mean_target) ** 2 for t in targets)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2)
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate ML model predictions")
    parser.add_argument('predictions', help='Predictions file')
    parser.add_argument('--targets', help='Targets file (if separate from predictions)')
    parser.add_argument('--task', choices=['classification', 'generation', 'regression'],
                       default='classification', help='Task type')
    parser.add_argument('--pred-key', default='prediction', 
                       help='Key for predictions in JSON')
    parser.add_argument('--target-key', default='target',
                       help='Key for targets in JSON')
    parser.add_argument('--output', help='Output file for results')
    parser.add_argument('--format', choices=['json', 'text'], default='text',
                       help='Output format')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading predictions from {args.predictions}...")
    predictions, targets = load_predictions_and_targets(
        args.predictions, args.targets, args.pred_key, args.target_key
    )
    
    print(f"Loaded {len(predictions)} predictions and {len(targets)} targets")
    
    # Evaluate based on task type
    if args.task == 'classification':
        # Convert to appropriate types for classification
        try:
            predictions = [int(float(p)) for p in predictions]
            targets = [int(float(t)) for t in targets]
        except:
            pass  # Keep as strings if conversion fails
        
        results = evaluate_classification(predictions, targets)
    
    elif args.task == 'generation':
        predictions = [str(p) for p in predictions]
        targets = [str(t) for t in targets]
        results = evaluate_generation(predictions, targets)
    
    elif args.task == 'regression':
        predictions = [float(p) for p in predictions]
        targets = [float(t) for t in targets]
        results = evaluate_regression(predictions, targets)
    
    # Output results
    if args.format == 'json':
        output_text = json.dumps(results, indent=2, default=str)
    else:
        output_lines = [f"Evaluation Results ({args.task}):", "=" * 40]
        
        for key, value in results.items():
            if isinstance(value, dict):
                output_lines.append(f"\n{key}:")
                for sub_key, sub_value in value.items():
                    output_lines.append(f"  {sub_key}: {sub_value}")
            elif isinstance(value, list):
                output_lines.append(f"{key}: {value}")
            else:
                output_lines.append(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
        
        output_text = "\n".join(output_lines)
    
    # Save or print results
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_text)
        print(f"Results saved to {args.output}")
    else:
        print(output_text)


if __name__ == "__main__":
    main()