#!/usr/bin/env python3
"""
jsonl_splitter.py

Split large JSONL files into smaller chunks for easier processing.
"""

import argparse
import json
from pathlib import Path


def split_jsonl(input_path: str, output_dir: str, chunk_size: int = 1000):
    """Split JSONL file into smaller chunks."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    input_name = Path(input_path).stem
    chunk_count = 0
    line_count = 0
    current_chunk = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            current_chunk.append(line)
            line_count += 1
            
            if len(current_chunk) >= chunk_size:
                # Write chunk
                chunk_file = output_path / f"{input_name}_chunk_{chunk_count:04d}.jsonl"
                with open(chunk_file, 'w', encoding='utf-8') as cf:
                    for chunk_line in current_chunk:
                        cf.write(chunk_line + '\n')
                
                print(f"Wrote chunk {chunk_count}: {len(current_chunk)} lines")
                chunk_count += 1
                current_chunk = []
    
    # Write remaining lines
    if current_chunk:
        chunk_file = output_path / f"{input_name}_chunk_{chunk_count:04d}.jsonl"
        with open(chunk_file, 'w', encoding='utf-8') as cf:
            for chunk_line in current_chunk:
                cf.write(chunk_line + '\n')
        print(f"Wrote final chunk {chunk_count}: {len(current_chunk)} lines")
        chunk_count += 1
    
    print(f"Split {line_count} lines into {chunk_count} chunks")


def main():
    parser = argparse.ArgumentParser(description='Split JSONL files into chunks')
    parser.add_argument('input', help='Input JSONL file')
    parser.add_argument('output_dir', help='Output directory for chunks')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Lines per chunk')
    
    args = parser.parse_args()
    split_jsonl(args.input, args.output_dir, args.chunk_size)


if __name__ == '__main__':
    main()