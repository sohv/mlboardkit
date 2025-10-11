#!/usr/bin/env python3
"""
toxicity_filter.py

Flag and filter toxic outputs from AI models.
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

# Toxicity keywords and patterns
TOXICITY_PATTERNS = {
    'hate_speech': [
        r'\b(hate|hatred)\b.*\b(people|group|race|religion)\b',
        r'\b(disgusting|inferior|subhuman)\b',
        r'\b(terrorists?|extremists?)\b.*\b(all|every)\b'
    ],
    'violence': [
        r'\b(kill|murder|assassinate|eliminate)\b',
        r'\b(weapon|gun|knife|bomb)\b.*\b(use|make|create)\b',
        r'\b(attack|assault|harm|hurt)\b.*\b(people|person)\b'
    ],
    'harassment': [
        r'\b(ugly|stupid|worthless|pathetic)\b.*\b(you|person)\b',
        r'\b(shut up|go away|leave)\b',
        r'\b(bully|harass|intimidate)\b'
    ],
    'discrimination': [
        r'\b(inferior|superior)\b.*\b(race|gender|religion)\b',
        r'\b(stereotype|prejudice|bias)\b',
        r'\b(should not|cannot)\b.*\b(because|race|gender)\b'
    ],
    'self_harm': [
        r'\b(suicide|self-harm|cut yourself)\b',
        r'\b(end it all|not worth living)\b',
        r'\b(hurt yourself|harm yourself)\b'
    ]
}

PROFANITY_WORDS = [
    # Add common profanity words (kept minimal for professional use)
    'damn', 'hell', 'crap', 'stupid', 'idiot', 'moron'
]

def calculate_toxicity_score(text: str) -> Dict[str, Any]:
    """Calculate toxicity score for text"""
    text_lower = text.lower()
    
    # Count pattern matches
    pattern_matches = {}
    total_matches = 0
    
    for category, patterns in TOXICITY_PATTERNS.items():
        matches = 0
        for pattern in patterns:
            matches += len(re.findall(pattern, text_lower))
        pattern_matches[category] = matches
        total_matches += matches
    
    # Count profanity
    profanity_count = sum(1 for word in PROFANITY_WORDS if word in text_lower)
    
    # Calculate overall toxicity score
    text_length = len(text.split())
    if text_length == 0:
        toxicity_score = 0.0
    else:
        # Normalize by text length
        toxicity_score = min(1.0, (total_matches + profanity_count) / max(text_length, 10))
    
    # Determine toxicity level
    if toxicity_score >= 0.7:
        toxicity_level = 'high'
    elif toxicity_score >= 0.3:
        toxicity_level = 'medium'
    elif toxicity_score > 0:
        toxicity_level = 'low'
    else:
        toxicity_level = 'none'
    
    return {
        'toxicity_score': toxicity_score,
        'toxicity_level': toxicity_level,
        'pattern_matches': pattern_matches,
        'profanity_count': profanity_count,
        'total_matches': total_matches,
        'flagged': toxicity_score > 0.3
    }

def analyze_sentiment(text: str) -> Dict[str, Any]:
    """Simple sentiment analysis"""
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'helpful', 'kind']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate', 'angry']
    
    text_lower = text.lower()
    words = text_lower.split()
    
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    
    if positive_count + negative_count == 0:
        sentiment_score = 0.0
        sentiment = 'neutral'
    else:
        sentiment_score = (positive_count - negative_count) / (positive_count + negative_count)
        if sentiment_score > 0.3:
            sentiment = 'positive'
        elif sentiment_score < -0.3:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
    
    return {
        'sentiment': sentiment,
        'sentiment_score': sentiment_score,
        'positive_words': positive_count,
        'negative_words': negative_count
    }

def filter_toxic_content(data: List[Dict[str, Any]], 
                        text_key: str = 'text',
                        threshold: float = 0.3) -> Dict[str, Any]:
    """Filter toxic content from dataset"""
    
    filtered_data = []
    toxic_data = []
    stats = {
        'total': len(data),
        'filtered': 0,
        'toxic': 0,
        'by_category': {category: 0 for category in TOXICITY_PATTERNS.keys()}
    }
    
    for item in data:
        text = item.get(text_key, str(item))
        
        # Analyze toxicity
        toxicity_analysis = calculate_toxicity_score(text)
        sentiment_analysis = analyze_sentiment(text)
        
        # Add analysis to item
        analyzed_item = {
            **item,
            **toxicity_analysis,
            **sentiment_analysis
        }
        
        if toxicity_analysis['toxicity_score'] <= threshold:
            filtered_data.append(analyzed_item)
            stats['filtered'] += 1
        else:
            toxic_data.append(analyzed_item)
            stats['toxic'] += 1
            
            # Count by category
            for category, count in toxicity_analysis['pattern_matches'].items():
                if count > 0:
                    stats['by_category'][category] += 1
    
    return {
        'filtered_data': filtered_data,
        'toxic_data': toxic_data,
        'stats': stats
    }

def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load data from file"""
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
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        return data
    
    else:
        with open(path, 'r', encoding='utf-8') as f:
            return [{'text': line.strip()} for line in f if line.strip()]

def main():
    parser = argparse.ArgumentParser(description="Flag and filter toxic outputs")
    parser.add_argument('input_file', help='Input file with text data')
    parser.add_argument('--text-key', default='text', help='Key for text content')
    parser.add_argument('--threshold', type=float, default=0.3, help='Toxicity threshold')
    parser.add_argument('--output-clean', help='Output file for clean data')
    parser.add_argument('--output-toxic', help='Output file for toxic data')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze, do not filter')
    parser.add_argument('--stats', action='store_true', help='Show detailed statistics')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input_file}...")
    data = load_data(args.input_file)
    print(f"Loaded {len(data)} items")
    
    # Filter toxic content
    print(f"Analyzing toxicity (threshold: {args.threshold})...")
    results = filter_toxic_content(data, args.text_key, args.threshold)
    
    # Show statistics
    stats = results['stats']
    print(f"\nToxicity Analysis Results:")
    print(f"  Total items: {stats['total']}")
    print(f"  Clean items: {stats['filtered']} ({stats['filtered']/stats['total']*100:.1f}%)")
    print(f"  Toxic items: {stats['toxic']} ({stats['toxic']/stats['total']*100:.1f}%)")
    
    if args.stats and stats['toxic'] > 0:
        print(f"\nToxic Content by Category:")
        for category, count in stats['by_category'].items():
            if count > 0:
                print(f"  {category}: {count}")
    
    # Save results
    if not args.analyze_only:
        if args.output_clean:
            with open(args.output_clean, 'w', encoding='utf-8') as f:
                json.dump(results['filtered_data'], f, indent=2, ensure_ascii=False)
            print(f"Clean data saved to {args.output_clean}")
        
        if args.output_toxic:
            with open(args.output_toxic, 'w', encoding='utf-8') as f:
                json.dump(results['toxic_data'], f, indent=2, ensure_ascii=False)
            print(f"Toxic data saved to {args.output_toxic}")
        
        if not args.output_clean and not args.output_toxic:
            # Save analysis results
            output_file = Path(args.input_file).with_suffix('.analyzed.json')
            all_data = results['filtered_data'] + results['toxic_data']
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_data, f, indent=2, ensure_ascii=False)
            print(f"Analysis results saved to {output_file}")

if __name__ == "__main__":
    main()