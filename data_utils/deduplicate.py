#!/usr/bin/env python3
"""
deduplicate.py

Remove duplicate entries from text datasets with various similarity metrics.
Simple and focused deduplication utility.
"""

import argparse
import json
import hashlib
from pathlib import Path
from typing import List, Set, Dict, Any


def exact_dedup(texts: List[str]) -> List[str]:
    """Remove exact duplicates while preserving order"""
    seen = set()
    result = []
    for text in texts:
        if text not in seen:
            seen.add(text)
            result.append(text)
    return result


def hash_dedup(texts: List[str]) -> List[str]:
    """Remove duplicates using hash comparison"""
    seen_hashes = set()
    result = []
    for text in texts:
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        if text_hash not in seen_hashes:
            seen_hashes.add(text_hash)
            result.append(text)
    return result


def normalize_and_dedup(texts: List[str]) -> List[str]:
    """Remove duplicates after basic normalization"""
    seen = set()
    result = []
    for text in texts:
        # Basic normalization
        normalized = text.lower().strip()
        normalized = ' '.join(normalized.split())  # Normalize whitespace
        
        if normalized not in seen and normalized:
            seen.add(normalized)
            result.append(text)  # Keep original text
    return result


def length_based_dedup(texts: List[str], max_length_diff: int = 10) -> List[str]:
    """Remove texts with very similar lengths (potential duplicates)"""
    length_groups: Dict[int, List[str]] = {}
    
    # Group by length
    for text in texts:
        length = len(text)
        if length not in length_groups:
            length_groups[length] = []
        length_groups[length].append(text)
    
    result = []
    for length, group_texts in length_groups.items():
        if len(group_texts) == 1:
            result.extend(group_texts)
        else:
            # Keep only first occurrence for same length
            result.append(group_texts[0])
    
    return result


def jaccard_similarity(text1: str, text2: str) -> float:
    """Calculate Jaccard similarity between two texts"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)


def jaccard_dedup(texts: List[str], threshold: float = 0.8) -> List[str]:
    """Remove texts with high Jaccard similarity"""
    result = []
    
    for i, text in enumerate(texts):
        is_duplicate = False
        for j in range(i):
            if jaccard_similarity(text, texts[j]) > threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            result.append(text)
    
    return result


def load_texts(file_path: str) -> List[str]:
    """Load texts from various file formats"""
    path = Path(file_path)
    
    if path.suffix.lower() == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return [str(item) for item in data]
            else:
                return [str(data)]
    
    elif path.suffix.lower() == '.jsonl':
        texts = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if isinstance(data, dict):
                        # Extract text field or convert to string
                        text = data.get('text', str(data))
                    else:
                        text = str(data)
                    texts.append(text)
                except json.JSONDecodeError:
                    continue
        return texts
    
    else:  # Treat as text file
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]


def save_texts(texts: List[str], file_path: str, format: str = 'txt'):
    """Save texts to file in specified format"""
    path = Path(file_path)
    
    if format == 'json':
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(texts, f, indent=2, ensure_ascii=False)
    
    elif format == 'jsonl':
        with open(path, 'w', encoding='utf-8') as f:
            for text in texts:
                json.dump(text, f, ensure_ascii=False)
                f.write('\n')
    
    else:  # txt format
        with open(path, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')


def main():
    parser = argparse.ArgumentParser(description="Remove duplicate entries from text datasets")
    parser.add_argument('input_file', help='Input file path')
    parser.add_argument('--output', help='Output file path (default: input_file.dedup.ext)')
    parser.add_argument('--method', choices=['exact', 'hash', 'normalize', 'length', 'jaccard'], 
                       default='exact', help='Deduplication method')
    parser.add_argument('--threshold', type=float, default=0.8,
                       help='Similarity threshold for jaccard method')
    parser.add_argument('--format', choices=['txt', 'json', 'jsonl'], 
                       help='Output format (default: same as input)')
    parser.add_argument('--stats', action='store_true',
                       help='Show deduplication statistics')
    
    args = parser.parse_args()
    
    # Load texts
    print(f"Loading texts from {args.input_file}...")
    texts = load_texts(args.input_file)
    original_count = len(texts)
    print(f"Loaded {original_count} texts")
    
    # Apply deduplication
    print(f"Applying {args.method} deduplication...")
    
    if args.method == 'exact':
        deduplicated = exact_dedup(texts)
    elif args.method == 'hash':
        deduplicated = hash_dedup(texts)
    elif args.method == 'normalize':
        deduplicated = normalize_and_dedup(texts)
    elif args.method == 'length':
        deduplicated = length_based_dedup(texts)
    elif args.method == 'jaccard':
        deduplicated = jaccard_dedup(texts, args.threshold)
    
    final_count = len(deduplicated)
    removed_count = original_count - final_count
    
    # Determine output format
    if args.format:
        output_format = args.format
    else:
        input_path = Path(args.input_file)
        if input_path.suffix.lower() in ['.json', '.jsonl']:
            output_format = input_path.suffix.lower().lstrip('.')
        else:
            output_format = 'txt'
    
    # Determine output file
    if args.output:
        output_file = args.output
    else:
        input_path = Path(args.input_file)
        output_file = str(input_path.with_stem(f"{input_path.stem}.dedup"))
    
    # Save results
    save_texts(deduplicated, output_file, output_format)
    
    # Show statistics
    if args.stats or True:  # Always show basic stats
        print(f"\nDeduplication Results:")
        print(f"  Original texts: {original_count}")
        print(f"  After deduplication: {final_count}")
        print(f"  Removed: {removed_count} ({removed_count/original_count*100:.1f}%)")
        print(f"  Saved to: {output_file}")


if __name__ == "__main__":
    main()