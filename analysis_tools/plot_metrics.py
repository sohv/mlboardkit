#!/usr/bin/env python3
"""
plot_metrics.py

Plot losses, confusion matrices, and embedding projections for ML models.
Simple and focused visualization utility.
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from sklearn.metrics import confusion_matrix
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def plot_training_curves(data: Dict[str, List[float]], output_file: str = None, title: str = "Training Curves"):
    """Plot training and validation curves"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss curves
    if 'train_loss' in data:
        axes[0].plot(data['train_loss'], label='Train Loss', color='blue')
    if 'val_loss' in data:
        axes[0].plot(data['val_loss'], label='Validation Loss', color='red')
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy/metric curves
    metric_keys = [k for k in data.keys() if 'acc' in k.lower() or 'f1' in k.lower()]
    
    for key in metric_keys:
        axes[1].plot(data[key], label=key.replace('_', ' ').title())
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Performance Metrics')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {output_file}")
    else:
        plt.show()


def plot_confusion_matrix(y_true: List, y_pred: List, labels: List[str] = None, 
                         output_file: str = None, title: str = "Confusion Matrix"):
    """Plot confusion matrix"""
    if not HAS_SKLEARN:
        print("Warning: scikit-learn required for confusion matrix")
        return
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    
    if labels:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {output_file}")
    else:
        plt.show()


def plot_embeddings_2d(embeddings: np.ndarray, labels: List = None, method: str = 'tsne',
                       output_file: str = None, title: str = "Embedding Visualization"):
    """Plot 2D embeddings using t-SNE or PCA"""
    if not HAS_SKLEARN:
        print("Warning: scikit-learn required for embedding visualization")
        return
    
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    else:
        raise ValueError("Method must be 'tsne' or 'pca'")
    
    embeddings_2d = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    
    if labels:
        unique_labels = list(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       c=[colors[i]], label=label, alpha=0.7)
        
        plt.legend()
    else:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
    
    plt.title(f"{title} ({method.upper()})")
    plt.xlabel(f"{method.upper()} Component 1")
    plt.ylabel(f"{method.upper()} Component 2")
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Embedding plot saved to {output_file}")
    else:
        plt.show()


def plot_distribution(data: List[float], bins: int = 50, output_file: str = None, 
                     title: str = "Data Distribution"):
    """Plot data distribution histogram"""
    plt.figure(figsize=(10, 6))
    
    plt.hist(data, bins=bins, alpha=0.7, edgecolor='black')
    
    # Add statistics
    mean_val = np.mean(data)
    std_val = np.std(data)
    
    plt.axvline(mean_val, color='red', linestyle='--', 
               label=f'Mean: {mean_val:.3f}')
    plt.axvline(mean_val + std_val, color='orange', linestyle='--', 
               label=f'+1 STD: {mean_val + std_val:.3f}')
    plt.axvline(mean_val - std_val, color='orange', linestyle='--', 
               label=f'-1 STD: {mean_val - std_val:.3f}')
    
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Distribution plot saved to {output_file}")
    else:
        plt.show()


def plot_metric_comparison(data: Dict[str, List[float]], output_file: str = None,
                          title: str = "Metric Comparison"):
    """Plot comparison of multiple metrics"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_positions = np.arange(len(data))
    bar_width = 0.8
    
    for i, (metric_name, values) in enumerate(data.items()):
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        ax.bar(i, mean_val, bar_width, yerr=std_val, 
               capsize=5, label=metric_name, alpha=0.8)
        
        # Add value labels on bars
        ax.text(i, mean_val + std_val + 0.01, f'{mean_val:.3f}', 
               ha='center', va='bottom')
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(data.keys(), rotation=45)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Metric comparison saved to {output_file}")
    else:
        plt.show()


def load_data(file_path: str) -> Dict[str, Any]:
    """Load data from various file formats"""
    path = Path(file_path)
    
    if path.suffix.lower() == '.json':
        with open(path, 'r') as f:
            return json.load(f)
    
    elif path.suffix.lower() == '.csv' and HAS_PANDAS:
        df = pd.read_csv(path)
        return df.to_dict('list')
    
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def main():
    parser = argparse.ArgumentParser(description="Plot ML metrics and visualizations")
    parser.add_argument('data_file', help='Data file (JSON or CSV)')
    parser.add_argument('--plot-type', choices=['training', 'confusion', 'embeddings', 'distribution', 'comparison'],
                       required=True, help='Type of plot to generate')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--title', help='Plot title')
    
    # Training curves options
    parser.add_argument('--loss-keys', nargs='+', default=['train_loss', 'val_loss'],
                       help='Keys for loss values in training curves')
    parser.add_argument('--metric-keys', nargs='+', 
                       help='Keys for metric values in training curves')
    
    # Confusion matrix options
    parser.add_argument('--true-key', default='y_true', help='Key for true labels')
    parser.add_argument('--pred-key', default='y_pred', help='Key for predictions')
    parser.add_argument('--label-names', nargs='+', help='Label names for confusion matrix')
    
    # Embeddings options
    parser.add_argument('--embedding-key', default='embeddings', help='Key for embeddings')
    parser.add_argument('--label-key', default='labels', help='Key for labels')
    parser.add_argument('--reduction-method', choices=['tsne', 'pca'], default='tsne',
                       help='Dimensionality reduction method')
    
    # Distribution options
    parser.add_argument('--value-key', default='values', help='Key for distribution values')
    parser.add_argument('--bins', type=int, default=50, help='Number of histogram bins')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data_file}...")
    data = load_data(args.data_file)
    
    # Set default title
    title = args.title or f"{args.plot_type.title()} Plot"
    
    # Generate plot based on type
    if args.plot_type == 'training':
        # Extract training data
        plot_data = {}
        for key in args.loss_keys:
            if key in data:
                plot_data[key] = data[key]
        
        if args.metric_keys:
            for key in args.metric_keys:
                if key in data:
                    plot_data[key] = data[key]
        
        plot_training_curves(plot_data, args.output, title)
    
    elif args.plot_type == 'confusion':
        y_true = data[args.true_key]
        y_pred = data[args.pred_key]
        plot_confusion_matrix(y_true, y_pred, args.label_names, args.output, title)
    
    elif args.plot_type == 'embeddings':
        embeddings = np.array(data[args.embedding_key])
        labels = data.get(args.label_key, None)
        plot_embeddings_2d(embeddings, labels, args.reduction_method, args.output, title)
    
    elif args.plot_type == 'distribution':
        values = data[args.value_key]
        plot_distribution(values, args.bins, args.output, title)
    
    elif args.plot_type == 'comparison':
        # Assume data contains multiple metrics to compare
        metric_data = {k: v for k, v in data.items() if isinstance(v, list)}
        plot_metric_comparison(metric_data, args.output, title)


if __name__ == "__main__":
    main()