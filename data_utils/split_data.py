#!/usr/bin/env python3
"""
split_data.py

Split datasets into train/validation/test sets with various strategies.
Simple and focused data splitting utility.
"""

import argparse
import json
import random
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def random_split(data: List[Any], train_ratio: float = 0.8, val_ratio: float = 0.1, 
                seed: int = 42) -> Tuple[List[Any], List[Any], List[Any]]:
    """Random split of data"""
    random.seed(seed)
    if HAS_NUMPY:
        np.random.seed(seed)
    
    data_copy = data.copy()
    random.shuffle(data_copy)
    
    n = len(data_copy)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    train_data = data_copy[:train_size]
    val_data = data_copy[train_size:train_size + val_size]
    test_data = data_copy[train_size + val_size:]
    
    return train_data, val_data, test_data


def stratified_split(data: List[Dict[str, Any]], label_key: str = 'label',
                    train_ratio: float = 0.8, val_ratio: float = 0.1,
                    seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Stratified split maintaining label distribution"""
    random.seed(seed)
    
    # Group data by labels
    label_groups: Dict[str, List[Dict]] = {}
    for item in data:
        label = str(item.get(label_key, 'unknown'))
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(item)
    
    train_data, val_data, test_data = [], [], []
    
    # Split each label group
    for label, items in label_groups.items():
        random.shuffle(items)
        
        n = len(items)
        train_size = max(1, int(n * train_ratio))
        val_size = max(0, int(n * val_ratio))
        
        train_data.extend(items[:train_size])
        val_data.extend(items[train_size:train_size + val_size])
        test_data.extend(items[train_size + val_size:])
    
    # Shuffle final splits
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    return train_data, val_data, test_data


def temporal_split(data: List[Dict[str, Any]], time_key: str = 'timestamp',
                  train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split data based on temporal order"""
    # Sort by timestamp
    sorted_data = sorted(data, key=lambda x: x.get(time_key, 0))
    
    n = len(sorted_data)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    train_data = sorted_data[:train_size]
    val_data = sorted_data[train_size:train_size + val_size]
    test_data = sorted_data[train_size + val_size:]
    
    return train_data, val_data, test_data


def group_split(data: List[Dict[str, Any]], group_key: str = 'group',
               train_ratio: float = 0.8, val_ratio: float = 0.1,
               seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split data ensuring groups don't span across splits"""
    random.seed(seed)
    
    # Group data by group key
    groups: Dict[str, List[Dict]] = {}
    for item in data:
        group = str(item.get(group_key, 'default'))
        if group not in groups:
            groups[group] = []
        groups[group].append(item)
    
    # Shuffle group names
    group_names = list(groups.keys())
    random.shuffle(group_names)
    
    # Assign groups to splits
    total_items = len(data)
    train_target = int(total_items * train_ratio)
    val_target = int(total_items * val_ratio)
    
    train_data, val_data, test_data = [], [], []
    train_count, val_count = 0, 0
    
    for group_name in group_names:
        group_items = groups[group_name]
        
        if train_count < train_target:
            train_data.extend(group_items)
            train_count += len(group_items)
        elif val_count < val_target:
            val_data.extend(group_items)
            val_count += len(group_items)
        else:
            test_data.extend(group_items)
    
    return train_data, val_data, test_data


def load_data(file_path: str) -> List[Any]:
    """Load data from various file formats"""
    path = Path(file_path)
    
    if path.suffix.lower() == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data if isinstance(data, list) else [data]
    
    elif path.suffix.lower() == '.jsonl':
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except json.JSONDecodeError:
                    continue
        return data
    
    else:  # Text file - each line is an item
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]


def save_split(data: List[Any], file_path: str, format: str = 'json'):
    """Save data split to file"""
    path = Path(file_path)
    
    if format == 'json':
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    elif format == 'jsonl':
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
    
    else:  # txt format
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(str(item) + '\n')


def print_split_stats(train_data: List[Any], val_data: List[Any], test_data: List[Any],
                     label_key: Optional[str] = None):
    """Print statistics about the split"""
    total = len(train_data) + len(val_data) + len(test_data)
    
    print(f"\nSplit Statistics:")
    print(f"  Total items: {total}")
    print(f"  Train: {len(train_data)} ({len(train_data)/total*100:.1f}%)")
    print(f"  Validation: {len(val_data)} ({len(val_data)/total*100:.1f}%)")
    print(f"  Test: {len(test_data)} ({len(test_data)/total*100:.1f}%)")
    
    # Label distribution if applicable
    if label_key and all(isinstance(item, dict) for item in train_data):
        print(f"\nLabel Distribution:")
        
        for split_name, split_data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
            if not split_data:
                continue
                
            label_counts = {}
            for item in split_data:
                label = str(item.get(label_key, 'unknown'))
                label_counts[label] = label_counts.get(label, 0) + 1
            
            print(f"  {split_name}:")
            for label, count in sorted(label_counts.items()):
                print(f"    {label}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Split datasets into train/validation/test sets")
    parser.add_argument('input_file', help='Input file path')
    parser.add_argument('--output-dir', default='.', help='Output directory')
    parser.add_argument('--prefix', default='data', help='Output file prefix')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='Validation set ratio')
    parser.add_argument('--method', choices=['random', 'stratified', 'temporal', 'group'],
                       default='random', help='Split method')
    parser.add_argument('--label-key', help='Key for labels (stratified split)')
    parser.add_argument('--time-key', help='Key for timestamps (temporal split)')
    parser.add_argument('--group-key', help='Key for groups (group split)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--format', choices=['json', 'jsonl', 'txt'], default='json',
                       help='Output format')
    parser.add_argument('--stats', action='store_true', help='Show detailed statistics')
    
    args = parser.parse_args()
    
    # Validate ratios
    if args.train_ratio + args.val_ratio >= 1.0:
        print("Error: train_ratio + val_ratio must be less than 1.0")
        return
    
    # Load data
    print(f"Loading data from {args.input_file}...")
    data = load_data(args.input_file)
    print(f"Loaded {len(data)} items")
    
    # Split data
    print(f"Splitting data using {args.method} method...")
    
    if args.method == 'random':
        train_data, val_data, test_data = random_split(
            data, args.train_ratio, args.val_ratio, args.seed
        )
    
    elif args.method == 'stratified':
        if not args.label_key:
            print("Error: --label-key required for stratified split")
            return
        train_data, val_data, test_data = stratified_split(
            data, args.label_key, args.train_ratio, args.val_ratio, args.seed
        )
    
    elif args.method == 'temporal':
        if not args.time_key:
            print("Error: --time-key required for temporal split")
            return
        train_data, val_data, test_data = temporal_split(
            data, args.time_key, args.train_ratio, args.val_ratio
        )
    
    elif args.method == 'group':
        if not args.group_key:
            print("Error: --group-key required for group split")
            return
        train_data, val_data, test_data = group_split(
            data, args.group_key, args.train_ratio, args.val_ratio, args.seed
        )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save splits
    format_ext = 'json' if args.format == 'json' else args.format
    
    train_file = output_dir / f"{args.prefix}_train.{format_ext}"
    val_file = output_dir / f"{args.prefix}_val.{format_ext}"
    test_file = output_dir / f"{args.prefix}_test.{format_ext}"
    
    save_split(train_data, train_file, args.format)
    save_split(val_data, val_file, args.format)
    save_split(test_data, test_file, args.format)
    
    print(f"\nFiles saved:")
    print(f"  Train: {train_file}")
    print(f"  Validation: {val_file}")
    print(f"  Test: {test_file}")
    
    # Show statistics
    if args.stats or True:  # Always show basic stats
        print_split_stats(train_data, val_data, test_data, args.label_key)


if __name__ == "__main__":
    main()