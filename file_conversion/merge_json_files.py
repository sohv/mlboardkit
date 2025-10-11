#!/usr/bin/env python3
"""
merge_json_files.py

Merge multiple JSON/JSONL files into a single file.
"""

import argparse
import json
import glob
from pathlib import Path


def merge_json_files(pattern: str, output_path: str, format_type: str = 'jsonl'):
    """Merge JSON files matching pattern."""
    files = glob.glob(pattern)
    if not files:
        print(f"No files found matching pattern: {pattern}")
        return
    
    print(f"Found {len(files)} files to merge")
    
    all_data = []
    
    for file_path in sorted(files):
        print(f"Processing: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.jsonl'):
                    # JSONL format
                    for line in f:
                        line = line.strip()
                        if line:
                            all_data.append(json.loads(line))
                else:
                    # Regular JSON
                    data = json.load(f)
                    if isinstance(data, list):
                        all_data.extend(data)
                    else:
                        all_data.append(data)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Write merged output
    with open(output_path, 'w', encoding='utf-8') as f:
        if format_type == 'jsonl':
            for item in all_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        else:
            json.dump(all_data, f, indent=2, ensure_ascii=False)
    
    print(f"Merged {len(all_data)} records to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Merge JSON/JSONL files')
    parser.add_argument('pattern', help='File pattern (e.g., "data/*.json")')
    parser.add_argument('output', help='Output file')
    parser.add_argument('--format', choices=['json', 'jsonl'], 
                       default='jsonl', help='Output format')
    
    args = parser.parse_args()
    merge_json_files(args.pattern, args.output, args.format)


if __name__ == '__main__':
    main()