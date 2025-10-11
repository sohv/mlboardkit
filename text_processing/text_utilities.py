#!/usr/bin/env python3
"""
Text Processing and NLP Utilities

Comprehensive text processing toolkit for ML/AI workflows including text cleaning,
preprocessing, analysis, and natural language processing operations.

Usage:
    python3 text_utilities.py clean --input text.txt --output clean_text.txt
    python3 text_utilities.py analyze --input documents/ --output analysis.json
    python3 text_utilities.py tokenize --input text.txt --model bert-base-uncased
    python3 text_utilities.py extract --input documents/ --type entities --output entities.json
    python3 text_utilities.py translate --input text.txt --target es --output translated.txt
"""

import argparse
import json
import re
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from datetime import datetime
import logging
from collections import Counter, defaultdict
import unicodedata
import string
import csv
from dataclasses import dataclass
import tempfile


@dataclass
class TextStats:
    """Text statistics container."""
    char_count: int = 0
    word_count: int = 0
    sentence_count: int = 0
    paragraph_count: int = 0
    avg_words_per_sentence: float = 0.0
    avg_chars_per_word: float = 0.0
    unique_words: int = 0
    reading_level: str = ""
    most_common_words: List[Tuple[str, int]] = None
    language: str = ""


class TextCleaner:
    """Text cleaning and preprocessing utilities."""
    
    def __init__(self):
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )
        
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        
        self.phone_pattern = re.compile(
            r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'
        )
        
        self.html_pattern = re.compile(r'<[^<]+?>')
        
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
    
    def clean_text(self, text: str, options: Dict[str, bool] = None) -> str:
        """Clean text based on specified options."""
        if options is None:
            options = {
                'remove_html': True,
                'remove_urls': True,
                'remove_emails': True,
                'remove_phones': True,
                'remove_emojis': True,
                'remove_mentions': True,
                'remove_hashtags': True,
                'normalize_whitespace': True,
                'normalize_unicode': True,
                'remove_punctuation': False,
                'lowercase': False,
                'remove_numbers': False,
                'remove_extra_spaces': True
            }
        
        cleaned = text
        
        # Remove HTML tags
        if options.get('remove_html'):
            cleaned = self.html_pattern.sub('', cleaned)
        
        # Remove URLs
        if options.get('remove_urls'):
            cleaned = self.url_pattern.sub('', cleaned)
        
        # Remove emails
        if options.get('remove_emails'):
            cleaned = self.email_pattern.sub('', cleaned)
        
        # Remove phone numbers
        if options.get('remove_phones'):
            cleaned = self.phone_pattern.sub('', cleaned)
        
        # Remove emojis
        if options.get('remove_emojis'):
            cleaned = self.emoji_pattern.sub('', cleaned)
        
        # Remove mentions and hashtags
        if options.get('remove_mentions'):
            cleaned = self.mention_pattern.sub('', cleaned)
        
        if options.get('remove_hashtags'):
            cleaned = self.hashtag_pattern.sub('', cleaned)
        
        # Normalize unicode
        if options.get('normalize_unicode'):
            cleaned = unicodedata.normalize('NFKD', cleaned)
        
        # Remove punctuation
        if options.get('remove_punctuation'):
            cleaned = cleaned.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        if options.get('remove_numbers'):
            cleaned = re.sub(r'\d+', '', cleaned)
        
        # Lowercase
        if options.get('lowercase'):
            cleaned = cleaned.lower()
        
        # Normalize whitespace
        if options.get('normalize_whitespace'):
            cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove extra spaces
        if options.get('remove_extra_spaces'):
            cleaned = cleaned.strip()
        
        return cleaned
    
    def extract_patterns(self, text: str) -> Dict[str, List[str]]:
        """Extract common patterns from text."""
        return {
            'urls': self.url_pattern.findall(text),
            'emails': self.email_pattern.findall(text),
            'phones': self.phone_pattern.findall(text),
            'mentions': self.mention_pattern.findall(text),
            'hashtags': self.hashtag_pattern.findall(text)
        }


class TextAnalyzer:
    """Text analysis and statistics utilities."""
    
    def __init__(self):
        # Common stop words
        self.stop_words = {
            'english': {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                       'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                       'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                       'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
                       'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
                       'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'}
        }
    
    def analyze_text(self, text: str, language: str = 'english') -> TextStats:
        """Comprehensive text analysis."""
        stats = TextStats()
        
        # Basic counts
        stats.char_count = len(text)
        words = self._tokenize_words(text)
        stats.word_count = len(words)
        sentences = self._tokenize_sentences(text)
        stats.sentence_count = len(sentences)
        paragraphs = text.split('\n\n')
        stats.paragraph_count = len([p for p in paragraphs if p.strip()])
        
        # Calculate averages
        if stats.sentence_count > 0:
            stats.avg_words_per_sentence = stats.word_count / stats.sentence_count
        
        if stats.word_count > 0:
            stats.avg_chars_per_word = stats.char_count / stats.word_count
        
        # Unique words (case-insensitive)
        unique_words = set(word.lower() for word in words if word.isalpha())
        stats.unique_words = len(unique_words)
        
        # Most common words (excluding stop words)
        stop_words = self.stop_words.get(language, set())
        filtered_words = [word.lower() for word in words 
                         if word.isalpha() and word.lower() not in stop_words]
        word_counts = Counter(filtered_words)
        stats.most_common_words = word_counts.most_common(10)
        
        # Reading level (simplified)
        stats.reading_level = self._calculate_reading_level(stats)
        
        # Try to detect language
        stats.language = self._detect_language(text)
        
        return stats
    
    def _tokenize_words(self, text: str) -> List[str]:
        """Simple word tokenization."""
        return re.findall(r'\b\w+\b', text)
    
    def _tokenize_sentences(self, text: str) -> List[str]:
        """Simple sentence tokenization."""
        # Split on sentence endings
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_reading_level(self, stats: TextStats) -> str:
        """Calculate approximate reading level using Flesch formula."""
        if stats.sentence_count == 0 or stats.word_count == 0:
            return "Unknown"
        
        # Simplified Flesch Reading Ease
        avg_sentence_length = stats.avg_words_per_sentence
        avg_syllables = stats.avg_chars_per_word * 0.5  # Rough approximation
        
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
        
        if score >= 90:
            return "Very Easy"
        elif score >= 80:
            return "Easy"
        elif score >= 70:
            return "Fairly Easy"
        elif score >= 60:
            return "Standard"
        elif score >= 50:
            return "Fairly Difficult"
        elif score >= 30:
            return "Difficult"
        else:
            return "Very Difficult"
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection based on character patterns."""
        # Very basic detection - in practice, use langdetect library
        sample = text[:1000].lower()
        
        # English indicators
        english_words = ['the', 'and', 'to', 'of', 'a', 'in', 'for', 'is', 'on', 'that']
        english_score = sum(1 for word in english_words if word in sample)
        
        # Spanish indicators
        spanish_words = ['el', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no']
        spanish_score = sum(1 for word in spanish_words if word in sample)
        
        # French indicators
        french_words = ['le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir']
        french_score = sum(1 for word in french_words if word in sample)
        
        scores = {'english': english_score, 'spanish': spanish_score, 'french': french_score}
        detected = max(scores, key=scores.get)
        
        return detected if scores[detected] > 2 else 'unknown'
    
    def extract_keywords(self, text: str, num_keywords: int = 10) -> List[Tuple[str, float]]:
        """Extract keywords using TF-IDF like scoring."""
        words = self._tokenize_words(text.lower())
        
        # Filter words
        filtered_words = [word for word in words 
                         if len(word) > 2 and word.isalpha() 
                         and word not in self.stop_words.get('english', set())]
        
        if not filtered_words:
            return []
        
        # Calculate frequencies
        word_freq = Counter(filtered_words)
        total_words = len(filtered_words)
        
        # Simple TF scoring (normalized frequency)
        tf_scores = {word: freq / total_words for word, freq in word_freq.items()}
        
        # Sort by score
        keywords = sorted(tf_scores.items(), key=lambda x: x[1], reverse=True)
        
        return keywords[:num_keywords]
    
    def find_similar_sentences(self, text: str, threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """Find similar sentences in text."""
        sentences = self._tokenize_sentences(text)
        
        if len(sentences) < 2:
            return []
        
        similar_pairs = []
        
        for i, sent1 in enumerate(sentences):
            for j, sent2 in enumerate(sentences[i+1:], i+1):
                similarity = self._calculate_sentence_similarity(sent1, sent2)
                if similarity >= threshold:
                    similar_pairs.append((sent1, sent2, similarity))
        
        return sorted(similar_pairs, key=lambda x: x[2], reverse=True)
    
    def _calculate_sentence_similarity(self, sent1: str, sent2: str) -> float:
        """Calculate simple word overlap similarity."""
        words1 = set(self._tokenize_words(sent1.lower()))
        words2 = set(self._tokenize_words(sent2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0


class TextProcessor:
    """Advanced text processing operations."""
    
    def __init__(self):
        self.cleaner = TextCleaner()
        self.analyzer = TextAnalyzer()
    
    def process_file(self, input_path: str, output_path: str, 
                    operations: List[str], options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process text file with specified operations."""
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        print(f"📖 Processing file: {input_path}")
        
        # Read input
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        results = {'original_stats': self.analyzer.analyze_text(text)}
        processed_text = text
        
        # Apply operations
        for operation in operations:
            if operation == 'clean':
                clean_options = options.get('clean_options', {}) if options else {}
                processed_text = self.cleaner.clean_text(processed_text, clean_options)
                results['cleaned'] = True
            
            elif operation == 'analyze':
                stats = self.analyzer.analyze_text(processed_text)
                results['analysis'] = stats.__dict__
            
            elif operation == 'extract_keywords':
                num_keywords = options.get('num_keywords', 10) if options else 10
                keywords = self.analyzer.extract_keywords(processed_text, num_keywords)
                results['keywords'] = keywords
            
            elif operation == 'extract_patterns':
                patterns = self.cleaner.extract_patterns(processed_text)
                results['patterns'] = patterns
            
            elif operation == 'find_duplicates':
                threshold = options.get('similarity_threshold', 0.7) if options else 0.7
                duplicates = self.analyzer.find_similar_sentences(processed_text, threshold)
                results['similar_sentences'] = duplicates
            
            elif operation == 'split_sentences':
                sentences = self.analyzer._tokenize_sentences(processed_text)
                results['sentences'] = sentences
            
            elif operation == 'split_paragraphs':
                paragraphs = [p.strip() for p in processed_text.split('\n\n') if p.strip()]
                results['paragraphs'] = paragraphs
        
        # Save output
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix.lower() == '.json':
            # Save as JSON with results
            output_data = {
                'processed_text': processed_text,
                'results': results,
                'processing_info': {
                    'operations': operations,
                    'options': options,
                    'processed_at': datetime.now().isoformat()
                }
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
        else:
            # Save processed text
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(processed_text)
        
        print(f"✅ Processed text saved to: {output_path}")
        return results
    
    def process_directory(self, input_dir: str, output_dir: str,
                         operations: List[str], file_pattern: str = "*.txt",
                         options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process all files in directory."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Processing directory: {input_dir}")
        
        files = list(input_dir.glob(file_pattern))
        if not files:
            print(f"⚠️  No files found matching pattern: {file_pattern}")
            return {}
        
        results = {}
        
        for file_path in files:
            print(f"📄 Processing: {file_path.name}")
            
            try:
                output_file = output_dir / f"{file_path.stem}_processed{file_path.suffix}"
                file_results = self.process_file(str(file_path), str(output_file), 
                                               operations, options)
                results[str(file_path)] = file_results
                
            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")
                results[str(file_path)] = {'error': str(e)}
        
        # Save summary
        summary_path = output_dir / 'processing_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"📊 Summary saved to: {summary_path}")
        return results
    
    def create_text_corpus(self, input_dir: str, output_path: str,
                          file_pattern: str = "*.txt", 
                          clean_options: Dict[str, bool] = None) -> Dict[str, Any]:
        """Create cleaned text corpus from directory of files."""
        input_dir = Path(input_dir)
        output_path = Path(output_path)
        
        print(f"📚 Creating text corpus from: {input_dir}")
        
        files = list(input_dir.glob(file_pattern))
        if not files:
            raise ValueError(f"No files found matching pattern: {file_pattern}")
        
        corpus_data = []
        stats = {
            'total_files': len(files),
            'total_chars': 0,
            'total_words': 0,
            'total_sentences': 0,
            'files_processed': []
        }
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Clean text if options provided
                if clean_options:
                    text = self.cleaner.clean_text(text, clean_options)
                
                # Analyze
                file_stats = self.analyzer.analyze_text(text)
                
                # Add to corpus
                corpus_data.append({
                    'file': str(file_path.relative_to(input_dir)),
                    'text': text,
                    'stats': file_stats.__dict__
                })
                
                # Update totals
                stats['total_chars'] += file_stats.char_count
                stats['total_words'] += file_stats.word_count
                stats['total_sentences'] += file_stats.sentence_count
                stats['files_processed'].append(str(file_path.name))
                
                print(f"✅ Processed: {file_path.name} ({file_stats.word_count} words)")
                
            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")
        
        # Save corpus
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix.lower() == '.json':
            corpus = {
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'source_directory': str(input_dir),
                    'file_pattern': file_pattern,
                    'clean_options': clean_options,
                    'statistics': stats
                },
                'documents': corpus_data
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(corpus, f, indent=2, ensure_ascii=False, default=str)
        
        else:
            # Save as plain text
            with open(output_path, 'w', encoding='utf-8') as f:
                for doc in corpus_data:
                    f.write(f"=== {doc['file']} ===\n")
                    f.write(doc['text'])
                    f.write("\n\n")
        
        print(f"📚 Corpus created: {output_path}")
        print(f"📊 Total: {stats['total_files']} files, {stats['total_words']} words")
        
        return stats


class NLPUtilities:
    """Natural Language Processing utilities (basic implementations)."""
    
    def __init__(self):
        self.analyzer = TextAnalyzer()
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Simple entity extraction (basic patterns)."""
        entities = {
            'emails': [],
            'urls': [],
            'phones': [],
            'dates': [],
            'numbers': [],
            'capitalized_words': []
        }
        
        cleaner = TextCleaner()
        
        # Extract known patterns
        entities['emails'] = cleaner.email_pattern.findall(text)
        entities['urls'] = cleaner.url_pattern.findall(text)
        entities['phones'] = cleaner.phone_pattern.findall(text)
        
        # Extract dates (simple patterns)
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b'
        ]
        
        for pattern in date_patterns:
            entities['dates'].extend(re.findall(pattern, text, re.IGNORECASE))
        
        # Extract numbers
        entities['numbers'] = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        
        # Extract capitalized words (potential names/places)
        words = re.findall(r'\b[A-Z][a-z]+\b', text)
        # Filter out common words
        common_words = {'The', 'This', 'That', 'And', 'But', 'Or', 'If', 'When', 'Where'}
        entities['capitalized_words'] = [w for w in words if w not in common_words]
        
        return entities
    
    def sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Basic sentiment analysis using word lists."""
        # Simple positive/negative word lists
        positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love',
            'like', 'enjoy', 'happy', 'pleased', 'satisfied', 'perfect', 'best',
            'awesome', 'brilliant', 'outstanding', 'superb', 'marvelous'
        }
        
        negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'angry',
            'sad', 'disappointed', 'frustrated', 'worst', 'disgusting', 'annoying',
            'boring', 'stupid', 'useless', 'pathetic', 'ridiculous', 'dreadful'
        }
        
        words = self.analyzer._tokenize_words(text.lower())
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        total_words = len(words)
        
        if total_words == 0:
            return {'sentiment': 'neutral', 'confidence': 0.0, 'scores': {}}
        
        positive_score = positive_count / total_words
        negative_score = negative_count / total_words
        
        if positive_score > negative_score:
            sentiment = 'positive'
            confidence = positive_score / (positive_score + negative_score) if (positive_score + negative_score) > 0 else 0.5
        elif negative_score > positive_score:
            sentiment = 'negative'
            confidence = negative_score / (positive_score + negative_score) if (positive_score + negative_score) > 0 else 0.5
        else:
            sentiment = 'neutral'
            confidence = 0.5
        
        return {
            'sentiment': sentiment,
            'confidence': min(confidence, 1.0),
            'scores': {
                'positive': positive_score,
                'negative': negative_score,
                'neutral': 1.0 - positive_score - negative_score
            },
            'word_counts': {
                'positive_words': positive_count,
                'negative_words': negative_count,
                'total_words': total_words
            }
        }
    
    def text_summarization(self, text: str, num_sentences: int = 3) -> str:
        """Simple extractive text summarization."""
        sentences = self.analyzer._tokenize_sentences(text)
        
        if len(sentences) <= num_sentences:
            return text
        
        # Score sentences based on word frequency
        words = self.analyzer._tokenize_words(text.lower())
        word_freq = Counter(words)
        
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            sentence_words = self.analyzer._tokenize_words(sentence.lower())
            score = sum(word_freq.get(word, 0) for word in sentence_words)
            sentence_scores[i] = score / len(sentence_words) if sentence_words else 0
        
        # Select top sentences
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
        top_sentences.sort(key=lambda x: x[0])  # Sort by original order
        
        summary_sentences = [sentences[i] for i, _ in top_sentences]
        return '. '.join(summary_sentences) + '.'


def main():
    parser = argparse.ArgumentParser(description="Text processing and NLP utilities")
    parser.add_argument('--config', help='Configuration file (YAML/JSON)')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Clean and preprocess text')
    clean_parser.add_argument('--input', required=True, help='Input file or directory')
    clean_parser.add_argument('--output', required=True, help='Output file or directory')
    clean_parser.add_argument('--remove-html', action='store_true', help='Remove HTML tags')
    clean_parser.add_argument('--remove-urls', action='store_true', help='Remove URLs')
    clean_parser.add_argument('--remove-emails', action='store_true', help='Remove email addresses')
    clean_parser.add_argument('--remove-emojis', action='store_true', help='Remove emojis')
    clean_parser.add_argument('--lowercase', action='store_true', help='Convert to lowercase')
    clean_parser.add_argument('--remove-punctuation', action='store_true', help='Remove punctuation')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze text statistics')
    analyze_parser.add_argument('--input', required=True, help='Input file or directory')
    analyze_parser.add_argument('--output', help='Output JSON file for results')
    analyze_parser.add_argument('--language', default='english', help='Text language')

    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract information from text')
    extract_parser.add_argument('--input', required=True, help='Input file or directory')
    extract_parser.add_argument('--type', choices=['entities', 'keywords', 'patterns'], 
                               required=True, help='Type of extraction')
    extract_parser.add_argument('--output', required=True, help='Output JSON file')
    extract_parser.add_argument('--num-keywords', type=int, default=10, help='Number of keywords to extract')

    # Corpus command
    corpus_parser = subparsers.add_parser('corpus', help='Create text corpus')
    corpus_parser.add_argument('--input', required=True, help='Input directory')
    corpus_parser.add_argument('--output', required=True, help='Output corpus file')
    corpus_parser.add_argument('--pattern', default='*.txt', help='File pattern to match')
    corpus_parser.add_argument('--clean', action='store_true', help='Clean text during corpus creation')

    # Sentiment command
    sentiment_parser = subparsers.add_parser('sentiment', help='Analyze sentiment')
    sentiment_parser.add_argument('--input', required=True, help='Input file')
    sentiment_parser.add_argument('--output', help='Output JSON file')

    # Summarize command
    summarize_parser = subparsers.add_parser('summarize', help='Summarize text')
    summarize_parser.add_argument('--input', required=True, help='Input file')
    summarize_parser.add_argument('--output', required=True, help='Output file')
    summarize_parser.add_argument('--sentences', type=int, default=3, help='Number of sentences in summary')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        processor = TextProcessor()
        nlp = NLPUtilities()

        if args.command == 'clean':
            clean_options = {
                'remove_html': args.remove_html,
                'remove_urls': args.remove_urls,
                'remove_emails': args.remove_emails,
                'remove_emojis': args.remove_emojis,
                'lowercase': args.lowercase,
                'remove_punctuation': args.remove_punctuation,
                'normalize_whitespace': True,
                'remove_extra_spaces': True
            }
            
            input_path = Path(args.input)
            if input_path.is_file():
                processor.process_file(args.input, args.output, ['clean'], 
                                     {'clean_options': clean_options})
            else:
                processor.process_directory(args.input, args.output, ['clean'],
                                          options={'clean_options': clean_options})

        elif args.command == 'analyze':
            input_path = Path(args.input)
            
            if input_path.is_file():
                with open(input_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                stats = processor.analyzer.analyze_text(text, args.language)
                result = {
                    'file': str(input_path),
                    'analysis': stats.__dict__
                }
                
                if args.output:
                    with open(args.output, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
                    print(f"📊 Analysis saved to: {args.output}")
                else:
                    print(json.dumps(result, indent=2, default=str))
            
            else:
                results = processor.process_directory(args.input, args.output or tempfile.mkdtemp(),
                                                    ['analyze'])
                if args.output:
                    with open(args.output, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        elif args.command == 'extract':
            input_path = Path(args.input)
            
            if input_path.is_file():
                with open(input_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                if args.type == 'entities':
                    result = nlp.extract_entities(text)
                elif args.type == 'keywords':
                    result = processor.analyzer.extract_keywords(text, args.num_keywords)
                elif args.type == 'patterns':
                    result = processor.cleaner.extract_patterns(text)
                
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump({'file': str(input_path), 'extracted': result}, 
                             f, indent=2, ensure_ascii=False, default=str)
                
                print(f"📤 Extraction results saved to: {args.output}")

        elif args.command == 'corpus':
            clean_options = None
            if args.clean:
                clean_options = {
                    'remove_html': True,
                    'remove_urls': True,
                    'normalize_whitespace': True,
                    'remove_extra_spaces': True
                }
            
            processor.create_text_corpus(args.input, args.output, args.pattern, clean_options)

        elif args.command == 'sentiment':
            with open(args.input, 'r', encoding='utf-8') as f:
                text = f.read()
            
            sentiment_result = nlp.sentiment_analysis(text)
            result = {
                'file': args.input,
                'sentiment_analysis': sentiment_result
            }
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"😊 Sentiment analysis saved to: {args.output}")
            else:
                print(json.dumps(result, indent=2))

        elif args.command == 'summarize':
            with open(args.input, 'r', encoding='utf-8') as f:
                text = f.read()
            
            summary = nlp.text_summarization(text, args.sentences)
            
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(summary)
            
            print(f"📝 Summary saved to: {args.output}")
            print(f"Original: {len(text)} chars, Summary: {len(summary)} chars")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()