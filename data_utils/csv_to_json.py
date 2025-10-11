#!/usr/bin/env python3
"""
csv_to_json.py

Convert CSV files to JSON format with various output options.
"""

import argparse
import csv
import json
from pathlib import Path


def csv_to_json(input_path: str, output_path: str, format_type: str = 'records'):
    """
    Convert CSV to JSON.
    format_type: 'records' (list of dicts) or 'columns' (dict of lists)
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    if format_type == 'columns':
        if data:
            columns = {}
            for key in data[0].keys():
                columns[key] = [row[key] for row in data]
            data = columns
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {input_path} to {output_path} ({len(data)} records)")


def main():
    parser = argparse.ArgumentParser(description='Convert CSV to JSON')
    parser.add_argument('input', help='Input CSV file')
    parser.add_argument('output', help='Output JSON file')
    parser.add_argument('--format', choices=['records', 'columns'], 
                       default='records', help='JSON format')
    
    args = parser.parse_args()
    csv_to_json(args.input, args.output, args.format)


if __name__ == '__main__':
    main()