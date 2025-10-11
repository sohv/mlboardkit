#!/usr/bin/env python3
"""
metrics_utils.py

Small utilities for common ML metrics used in experiments.
Provides classification and regression metrics, CLI, and a simple JSON report writer.
"""

from typing import List, Dict, Any
import argparse
import json
import math


def accuracy(y_true: List[Any], y_pred: List[Any]) -> float:
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / len(y_true) if y_true else 0.0


def precision_recall_f1(y_true: List[int], y_pred: List[int], pos_label=1):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == pos_label and p == pos_label)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != pos_label and p == pos_label)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == pos_label and p != pos_label)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / n


def root_mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true: List[float], y_pred: List[float]) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / n


def regression_report(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': root_mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred)
    }


def classification_report(y_true: List[int], y_pred: List[int], pos_label=1) -> Dict[str, float]:
    acc = accuracy(y_true, y_pred)
    precision, recall, f1 = precision_recall_f1(y_true, y_pred, pos_label)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}


def main():
    parser = argparse.ArgumentParser(description='Metrics utilities')
    sub = parser.add_subparsers(dest='cmd')

    cls = sub.add_parser('classification', help='Compute classification metrics')
    cls.add_argument('--y_true', required=True, help='JSON list of true labels')
    cls.add_argument('--y_pred', required=True, help='JSON list of predicted labels')
    cls.add_argument('--pos', type=int, default=1, help='Positive label')
    cls.add_argument('--output', help='Output JSON file')

    reg = sub.add_parser('regression', help='Compute regression metrics')
    reg.add_argument('--y_true', required=True, help='JSON list of true values')
    reg.add_argument('--y_pred', required=True, help='JSON list of predicted values')
    reg.add_argument('--output', help='Output JSON file')

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        return

    if args.cmd == 'classification':
        y_true = json.loads(args.y_true)
        y_pred = json.loads(args.y_pred)
        report = classification_report(y_true, y_pred, args.pos)
    else:
        y_true = json.loads(args.y_true)
        y_pred = json.loads(args.y_pred)
        report = regression_report(y_true, y_pred)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Saved report to {args.output}")
    else:
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
