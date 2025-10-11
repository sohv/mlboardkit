#!/usr/bin/env python3
"""
data_label_inspector.py

Inspect label distributions and sample examples per class.
"""

import argparse
import csv
import json
from collections import defaultdict, Counter
from typing import Dict, List


def infer_label_column(header: List[str]) -> int:
    # heuristic: look for 'label' column
    for i, h in enumerate(header):
        if 'label' in h.lower():
            return i
    return 0


def inspect_csv(path: str, label_col: int = None, samples_per_class: int = 3):
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        if label_col is None:
            label_col = infer_label_column(header)
        samples = defaultdict(list)
        counts = Counter()
        for row in reader:
            label = row[label_col] if label_col < len(row) else 'UNKNOWN'
            counts[label] += 1
            if len(samples[label]) < samples_per_class:
                samples[label].append(row)
    return {'counts': dict(counts), 'samples': {k: v for k, v in samples.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--label-col', type=int, help='0-based label column index')
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--output', help='Save report JSON')
    args = parser.parse_args()

    report = inspect_csv(args.input, args.label_col, args.samples)
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Saved report to {args.output}")
    else:
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
