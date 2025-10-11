#!/usr/bin/env python3
"""
remove_duplicates.py

Detect and remove duplicate lines or JSON entries from files.
Simple and focused deduplication utility.
"""

import argparse
import json
import hashlib
from pathlib import Path
from typing import List, Set, Any


def remove_duplicate_lines(file_path: str, output_path: str = None, case_sensitive: bool = True):
    """Remove duplicate lines from text file"""
    seen = set()
    unique_lines = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            original_line = line.rstrip('\n\r')
            
            # Normalize for comparison
            compare_line = original_line if case_sensitive else original_line.lower()
            
            if compare_line not in seen:
                seen.add(compare_line)
                unique_lines.append(original_line)
    
    # Write output
    output_file = output_path or str(Path(file_path).with_suffix('.dedup.txt'))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in unique_lines:
            f.write(line + '\n')
    
    removed_count = line_num - len(unique_lines)
    print(f"Removed {removed_count} duplicate lines")
    print(f"Output saved to: {output_file}")


def remove_duplicate_json_entries(file_path: str, output_path: str = None, key_field: str = None):
    """Remove duplicate JSON entries"""
    path = Path(file_path)
    
    if path.suffix.lower() == '.jsonl':
        # Handle JSONL format
        seen = set()
        unique_entries = []
        
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line.strip())
                    
                    # Create identifier for deduplication
                    if key_field and isinstance(entry, dict):
                        identifier = entry.get(key_field, json.dumps(entry, sort_keys=True))
                    else:
                        identifier = json.dumps(entry, sort_keys=True)
                    
                    if identifier not in seen:
                        seen.add(identifier)
                        unique_entries.append(entry)
                        
                except json.JSONDecodeError:
                    continue
        
        # Write output
        output_file = output_path or str(path.with_suffix('.dedup.jsonl'))
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in unique_entries:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
    
    else:
        # Handle regular JSON format
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            seen = set()
            unique_entries = []
            
            for entry in data:
                # Create identifier
                if key_field and isinstance(entry, dict):
                    identifier = entry.get(key_field, json.dumps(entry, sort_keys=True))
                else:
                    identifier = json.dumps(entry, sort_keys=True)
                
                if identifier not in seen:
                    seen.add(identifier)
                    unique_entries.append(entry)
            
            # Write output
            output_file = output_path or str(path.with_suffix('.dedup.json'))
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(unique_entries, f, indent=2, ensure_ascii=False)
            
            removed_count = len(data) - len(unique_entries)
        else:
            # Single object, no deduplication needed
            unique_entries = data
            removed_count = 0
            output_file = output_path or str(path.with_suffix('.dedup.json'))
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
    
    removed_count = line_num - len(unique_entries) if path.suffix.lower() == '.jsonl' else removed_count
    print(f"Removed {removed_count} duplicate entries")
    print(f"Output saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Remove duplicate lines or JSON entries")
    parser.add_argument('input_file', help='Input file path')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--format', choices=['text', 'json', 'jsonl', 'auto'], 
                       default='auto', help='File format')
    parser.add_argument('--key-field', help='JSON field to use for deduplication')
    parser.add_argument('--case-sensitive', action='store_true', default=True,
                       help='Case sensitive comparison for text files')
    parser.add_argument('--stats', action='store_true', help='Show detailed statistics')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    
    if not input_path.exists():
        print(f"Error: Input file does not exist: {args.input_file}")
        return
    
    # Determine format
    if args.format == 'auto':
        if input_path.suffix.lower() == '.json':
            format_type = 'json'
        elif input_path.suffix.lower() == '.jsonl':
            format_type = 'jsonl'
        else:
            format_type = 'text'
    else:
        format_type = args.format
    
    print(f"Processing {args.input_file} as {format_type} format...")
    
    # Remove duplicates based on format
    if format_type == 'text':
        remove_duplicate_lines(args.input_file, args.output, args.case_sensitive)
    else:
        remove_duplicate_json_entries(args.input_file, args.output, args.key_field)


if __name__ == "__main__":
    main()