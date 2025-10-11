#!/usr/bin/env python3
"""
viz_utils.py

Visualization helpers for ML workflows: confusion matrix, ROC, PR, feature importance.
"""

from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_confusion_matrix(cm: List[List[int]], labels: Optional[List[str]] = None,
                          normalize: bool = False, cmap: str = 'Blues',
                          outfile: Optional[str] = None):
    cm = np.array(cm)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    if labels is None:
        labels = [str(i) for i in range(cm.shape[0])]

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           ylabel='True label', xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    if outfile:
        plt.savefig(outfile, bbox_inches='tight')
        print(f"Saved confusion matrix to {outfile}")
    else:
        plt.show()
    plt.close(fig)


def plot_roc(fpr: List[float], tpr: List[float], auc: Optional[float] = None, outfile: Optional[str] = None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})' if auc else 'ROC curve')
    ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate', title='Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    if outfile:
        fig.savefig(outfile, bbox_inches='tight')
        print(f"Saved ROC plot to {outfile}")
    else:
        plt.show()
    plt.close(fig)


def plot_pr(precisions: List[float], recalls: List[float], average_precision: Optional[float] = None, outfile: Optional[str] = None):
    fig, ax = plt.subplots()
    ax.plot(recalls, precisions, color='b', lw=2, label=f'AP = {average_precision:.2f}' if average_precision else 'PR curve')
    ax.set(xlabel='Recall', ylabel='Precision', title='Precision-Recall curve')
    ax.legend(loc="lower left")
    if outfile:
        fig.savefig(outfile, bbox_inches='tight')
        print(f"Saved PR plot to {outfile}")
    else:
        plt.show()
    plt.close(fig)


def plot_feature_importances(importances: List[float], feature_names: List[str], top_k: int = 20, outfile: Optional[str] = None):
    importances = np.array(importances)
    idx = np.argsort(importances)[::-1][:top_k]
    names = [feature_names[i] for i in idx]
    vals = importances[idx]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.3)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, vals[::-1], align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names[::-1])
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importances')
    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, bbox_inches='tight')
        print(f"Saved feature importances to {outfile}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    print('viz_utils loaded - plotting helpers available')
