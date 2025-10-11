#!/usr/bin/env python3
"""
txt_to_jsonl.py

Convert plain text files to JSON Lines format with optional metadata.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime


def txt_to_jsonl(input_path: str, output_path: str, add_metadata: bool = False):
    """Convert text file to JSONL format."""
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, line in enumerate(lines):
            record = {'text': line}
            if add_metadata:
                record.update({
                    'id': i,
                    'source_file': Path(input_path).name,
                    'created_at': datetime.now().isoformat()
                })
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"Converted {len(lines)} lines to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert text to JSONL')
    parser.add_argument('input', help='Input text file')
    parser.add_argument('output', help='Output JSONL file')
    parser.add_argument('--metadata', action='store_true', help='Add metadata fields')
    
    args = parser.parse_args()
    txt_to_jsonl(args.input, args.output, args.metadata)


if __name__ == '__main__':
    main()