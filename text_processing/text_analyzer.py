#!/usr/bin/env python3
"""
text_analyzer.py

Comprehensive text analysis including sentiment, readability, and linguistic features.
"""

import argparse
import json
import re
import string
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextAnalyzer:
    def __init__(self):
        self.stop_words = self.get_english_stop_words()
        self.sentiment_lexicon = self.create_sentiment_lexicon()
    
    def get_english_stop_words(self) -> set:
        """Get English stop words"""
        return {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'but', 'or', 'not', 'this', 'they',
            'have', 'had', 'what', 'said', 'each', 'which', 'do', 'how',
            'their', 'if', 'up', 'out', 'many', 'then', 'them', 'these',
            'so', 'some', 'her', 'would', 'make', 'like', 'into', 'him',
            'time', 'two', 'more', 'go', 'no', 'way', 'could', 'my', 'than',
            'first', 'been', 'call', 'who', 'oil', 'sit', 'now', 'find',
            'down', 'day', 'did', 'get', 'come', 'made', 'may', 'part'
        }
    
    def create_sentiment_lexicon(self) -> Dict[str, float]:
        """Create a basic sentiment lexicon"""
        positive_words = {
            'good': 1, 'great': 2, 'excellent': 3, 'amazing': 3, 'wonderful': 2,
            'fantastic': 3, 'awesome': 2, 'perfect': 3, 'love': 2, 'best': 2,
            'beautiful': 2, 'brilliant': 2, 'outstanding': 3, 'superb': 2,
            'marvelous': 2, 'terrific': 2, 'fabulous': 2, 'incredible': 2,
            'magnificent': 2, 'spectacular': 2, 'pleased': 1, 'happy': 2,
            'delighted': 2, 'satisfied': 1, 'impressed': 1, 'positive': 1
        }
        
        negative_words = {
            'bad': -1, 'terrible': -3, 'awful': -3, 'horrible': -3, 'disgusting': -3,
            'hate': -2, 'worst': -2, 'ugly': -2, 'stupid': -2, 'ridiculous': -2,
            'pathetic': -2, 'useless': -2, 'worthless': -3, 'disappointing': -2,
            'frustrating': -2, 'annoying': -1, 'boring': -1, 'sad': -1,
            'angry': -2, 'furious': -3, 'outraged': -3, 'disgusted': -2,
            'disappointed': -2, 'upset': -1, 'unhappy': -1, 'negative': -1
        }
        
        return {**positive_words, **negative_words}
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Split into words
        words = text.split()
        
        # Remove empty strings
        words = [word for word in words if word.strip()]
        
        return words
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def basic_stats(self, text: str) -> Dict[str, Any]:
        """Calculate basic text statistics"""
        
        # Character-level stats
        char_count = len(text)
        char_count_no_spaces = len(text.replace(' ', ''))
        
        # Word-level stats
        words = self.tokenize(text)
        word_count = len(words)
        unique_words = len(set(words))
        
        # Sentence-level stats
        sentences = self.extract_sentences(text)
        sentence_count = len(sentences)
        
        # Paragraph-level stats
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        paragraph_count = len(paragraphs)
        
        # Average calculations
        avg_words_per_sentence = word_count / max(sentence_count, 1)
        avg_chars_per_word = char_count_no_spaces / max(word_count, 1)
        avg_sentences_per_paragraph = sentence_count / max(paragraph_count, 1)
        
        return {
            'character_count': char_count,
            'character_count_no_spaces': char_count_no_spaces,
            'word_count': word_count,
            'unique_word_count': unique_words,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count,
            'lexical_diversity': unique_words / max(word_count, 1),
            'avg_words_per_sentence': avg_words_per_sentence,
            'avg_chars_per_word': avg_chars_per_word,
            'avg_sentences_per_paragraph': avg_sentences_per_paragraph
        }
    
    def sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text"""
        
        words = self.tokenize(text)
        
        sentiment_scores = []
        word_sentiments = {}
        
        for word in words:
            if word in self.sentiment_lexicon:
                score = self.sentiment_lexicon[word]
                sentiment_scores.append(score)
                word_sentiments[word] = score
        
        if sentiment_scores:
            total_score = sum(sentiment_scores)
            avg_score = total_score / len(sentiment_scores)
            
            # Normalize to -1 to 1 scale
            normalized_score = max(-1, min(1, avg_score / 3))
            
            # Classify sentiment
            if normalized_score > 0.1:
                sentiment_label = 'positive'
            elif normalized_score < -0.1:
                sentiment_label = 'negative'
            else:
                sentiment_label = 'neutral'
        else:
            total_score = 0
            avg_score = 0
            normalized_score = 0
            sentiment_label = 'neutral'
        
        return {
            'sentiment_label': sentiment_label,
            'sentiment_score': normalized_score,
            'raw_score': total_score,
            'avg_score': avg_score,
            'sentiment_words_count': len(sentiment_scores),
            'positive_words': [w for w, s in word_sentiments.items() if s > 0],
            'negative_words': [w for w, s in word_sentiments.items() if s < 0]
        }
    
    def readability_analysis(self, text: str) -> Dict[str, Any]:
        """Analyze text readability"""
        
        stats = self.basic_stats(text)
        
        # Flesch Reading Ease Score
        # Formula: 206.835 - (1.015 × ASL) - (84.6 × ASW)
        # ASL = Average Sentence Length (words per sentence)
        # ASW = Average Syllables per Word
        
        asl = stats['avg_words_per_sentence']
        
        # Estimate syllables (simple approximation)
        words = self.tokenize(text)
        total_syllables = sum(self.count_syllables(word) for word in words)
        asw = total_syllables / max(len(words), 1)
        
        flesch_score = 206.835 - (1.015 * asl) - (84.6 * asw)
        flesch_score = max(0, min(100, flesch_score))  # Clamp to 0-100
        
        # Flesch-Kincaid Grade Level
        # Formula: (0.39 × ASL) + (11.8 × ASW) - 15.59
        fk_grade = (0.39 * asl) + (11.8 * asw) - 15.59
        fk_grade = max(0, fk_grade)
        
        # Classify readability
        if flesch_score >= 90:
            readability_level = 'very_easy'
        elif flesch_score >= 80:
            readability_level = 'easy'
        elif flesch_score >= 70:
            readability_level = 'fairly_easy'
        elif flesch_score >= 60:
            readability_level = 'standard'
        elif flesch_score >= 50:
            readability_level = 'fairly_difficult'
        elif flesch_score >= 30:
            readability_level = 'difficult'
        else:
            readability_level = 'very_difficult'
        
        return {
            'flesch_reading_ease': flesch_score,
            'flesch_kincaid_grade': fk_grade,
            'readability_level': readability_level,
            'avg_sentence_length': asl,
            'avg_syllables_per_word': asw,
            'estimated_reading_time_minutes': stats['word_count'] / 200  # 200 WPM average
        }
    
    def count_syllables(self, word: str) -> int:
        """Estimate syllable count for a word"""
        word = word.lower()
        
        # Remove non-alphabetic characters
        word = re.sub(r'[^a-z]', '', word)
        
        if not word:
            return 0
        
        # Count vowel groups
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        # Every word has at least 1 syllable
        return max(1, syllable_count)
    
    def keyword_analysis(self, text: str, top_k: int = 10) -> Dict[str, Any]:
        """Extract and analyze keywords"""
        
        words = self.tokenize(text)
        
        # Filter out stop words
        content_words = [word for word in words if word not in self.stop_words]
        
        # Count word frequencies
        word_freq = Counter(content_words)
        
        # Get top keywords
        top_keywords = word_freq.most_common(top_k)
        
        # Calculate keyword density
        total_words = len(words)
        keyword_densities = {
            word: (count / total_words) * 100
            for word, count in top_keywords
        }
        
        return {
            'top_keywords': top_keywords,
            'keyword_densities': keyword_densities,
            'total_unique_words': len(word_freq),
            'stop_words_filtered': len(words) - len(content_words)
        }
    
    def linguistic_features(self, text: str) -> Dict[str, Any]:
        """Extract linguistic features"""
        
        # Count different types of characters and patterns
        uppercase_count = sum(1 for c in text if c.isupper())
        lowercase_count = sum(1 for c in text if c.islower())
        digit_count = sum(1 for c in text if c.isdigit())
        punctuation_count = sum(1 for c in text if c in string.punctuation)
        
        # Count specific punctuation
        exclamation_count = text.count('!')
        question_count = text.count('?')
        comma_count = text.count(',')
        period_count = text.count('.')
        
        # Pattern matching
        url_count = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
        email_count = len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
        hashtag_count = len(re.findall(r'#\w+', text))
        mention_count = len(re.findall(r'@\w+', text))
        
        # Calculate ratios
        total_chars = len(text)
        
        return {
            'uppercase_ratio': uppercase_count / max(total_chars, 1),
            'lowercase_ratio': lowercase_count / max(total_chars, 1),
            'digit_ratio': digit_count / max(total_chars, 1),
            'punctuation_ratio': punctuation_count / max(total_chars, 1),
            'exclamation_count': exclamation_count,
            'question_count': question_count,
            'comma_count': comma_count,
            'period_count': period_count,
            'url_count': url_count,
            'email_count': email_count,
            'hashtag_count': hashtag_count,
            'mention_count': mention_count,
            'caps_lock_ratio': uppercase_count / max(uppercase_count + lowercase_count, 1)
        }
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Perform comprehensive text analysis"""
        
        analysis = {
            'basic_stats': self.basic_stats(text),
            'sentiment': self.sentiment_analysis(text),
            'readability': self.readability_analysis(text),
            'keywords': self.keyword_analysis(text),
            'linguistic_features': self.linguistic_features(text)
        }
        
        return analysis
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze text file"""
        
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        analysis = self.analyze_text(text)
        analysis['file_info'] = {
            'file_path': str(path),
            'file_size': path.stat().st_size,
            'file_encoding': 'utf-8'
        }
        
        return analysis
    
    def batch_analyze(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple texts"""
        
        results = []
        
        for i, text in enumerate(texts):
            logger.info(f"Analyzing text {i+1}/{len(texts)}")
            analysis = self.analyze_text(text)
            analysis['text_id'] = i
            results.append(analysis)
        
        return results
    
    def compare_texts(self, text1: str, text2: str) -> Dict[str, Any]:
        """Compare two texts"""
        
        analysis1 = self.analyze_text(text1)
        analysis2 = self.analyze_text(text2)
        
        # Calculate differences
        comparison = {
            'text1_analysis': analysis1,
            'text2_analysis': analysis2,
            'differences': {}
        }
        
        # Compare basic stats
        stats1 = analysis1['basic_stats']
        stats2 = analysis2['basic_stats']
        
        comparison['differences']['word_count_diff'] = stats2['word_count'] - stats1['word_count']
        comparison['differences']['sentence_count_diff'] = stats2['sentence_count'] - stats1['sentence_count']
        comparison['differences']['lexical_diversity_diff'] = stats2['lexical_diversity'] - stats1['lexical_diversity']
        
        # Compare sentiment
        sent1 = analysis1['sentiment']
        sent2 = analysis2['sentiment']
        
        comparison['differences']['sentiment_score_diff'] = sent2['sentiment_score'] - sent1['sentiment_score']
        comparison['differences']['sentiment_changed'] = sent1['sentiment_label'] != sent2['sentiment_label']
        
        # Compare readability
        read1 = analysis1['readability']
        read2 = analysis2['readability']
        
        comparison['differences']['flesch_score_diff'] = read2['flesch_reading_ease'] - read1['flesch_reading_ease']
        comparison['differences']['grade_level_diff'] = read2['flesch_kincaid_grade'] - read1['flesch_kincaid_grade']
        
        return comparison

def main():
    parser = argparse.ArgumentParser(description="Comprehensive text analysis")
    parser.add_argument('--text', help='Text to analyze (as string)')
    parser.add_argument('--file', help='Text file to analyze')
    parser.add_argument('--batch', help='JSON file with list of texts to analyze')
    parser.add_argument('--output', help='Output file for analysis results')
    parser.add_argument('--compare', nargs=2, help='Compare two text files')
    parser.add_argument('--keywords', type=int, default=10, help='Number of top keywords to extract')
    parser.add_argument('--format', choices=['json', 'text'], default='json', help='Output format')
    
    args = parser.parse_args()
    
    analyzer = TextAnalyzer()
    
    if args.compare:
        # Compare two files
        print(f"Comparing {args.compare[0]} and {args.compare[1]}...")
        
        with open(args.compare[0], 'r', encoding='utf-8') as f:
            text1 = f.read()
        
        with open(args.compare[1], 'r', encoding='utf-8') as f:
            text2 = f.read()
        
        comparison = analyzer.compare_texts(text1, text2)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, indent=2, ensure_ascii=False)
            print(f"Comparison saved to {args.output}")
        else:
            print(json.dumps(comparison, indent=2, ensure_ascii=False))
    
    elif args.batch:
        # Batch analysis
        print(f"Loading texts from {args.batch}...")
        with open(args.batch, 'r', encoding='utf-8') as f:
            texts = json.load(f)
        
        if isinstance(texts, dict):
            texts = list(texts.values())
        
        results = analyzer.batch_analyze(texts)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Batch analysis saved to {args.output}")
        else:
            print(f"Analyzed {len(results)} texts")
            for result in results:
                print(f"Text {result['text_id']}: {result['basic_stats']['word_count']} words, "
                      f"sentiment: {result['sentiment']['sentiment_label']}")
    
    elif args.file:
        # Analyze file
        print(f"Analyzing file: {args.file}")
        analysis = analyzer.analyze_file(args.file)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            print(f"Analysis saved to {args.output}")
        else:
            _display_analysis(analysis, args.format)
    
    elif args.text:
        # Analyze text string
        print("Analyzing provided text...")
        analysis = analyzer.analyze_text(args.text)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            print(f"Analysis saved to {args.output}")
        else:
            _display_analysis(analysis, args.format)
    
    else:
        # Demo with sample text
        sample_text = """
        This is a sample text for analysis. It contains multiple sentences with different sentiments.
        Some parts are positive and exciting! Other parts might be more neutral or even negative.
        The text analyzes various linguistic features including readability, sentiment, and keyword density.
        
        This tool can help you understand the characteristics of your text content.
        It's particularly useful for content creators, researchers, and data scientists.
        """
        
        print("Demo: Analyzing sample text...")
        analysis = analyzer.analyze_text(sample_text.strip())
        _display_analysis(analysis, args.format)

def _display_analysis(analysis: Dict[str, Any], format_type: str = 'json'):
    """Display analysis results"""
    
    if format_type == 'json':
        print(json.dumps(analysis, indent=2, ensure_ascii=False))
    
    else:  # text format
        print("TEXT ANALYSIS RESULTS")
        print("=" * 50)
        
        # Basic stats
        stats = analysis['basic_stats']
        print(f"\nBASIC STATISTICS:")
        print(f"  Words: {stats['word_count']:,}")
        print(f"  Characters: {stats['character_count']:,}")
        print(f"  Sentences: {stats['sentence_count']}")
        print(f"  Paragraphs: {stats['paragraph_count']}")
        print(f"  Lexical Diversity: {stats['lexical_diversity']:.3f}")
        print(f"  Avg Words/Sentence: {stats['avg_words_per_sentence']:.1f}")
        
        # Sentiment
        sentiment = analysis['sentiment']
        print(f"\nSENTIMENT ANALYSIS:")
        print(f"  Sentiment: {sentiment['sentiment_label'].title()}")
        print(f"  Score: {sentiment['sentiment_score']:.3f}")
        print(f"  Positive words: {', '.join(sentiment['positive_words'][:5])}")
        print(f"  Negative words: {', '.join(sentiment['negative_words'][:5])}")
        
        # Readability
        readability = analysis['readability']
        print(f"\nREADABILITY:")
        print(f"  Level: {readability['readability_level'].replace('_', ' ').title()}")
        print(f"  Flesch Score: {readability['flesch_reading_ease']:.1f}")
        print(f"  Grade Level: {readability['flesch_kincaid_grade']:.1f}")
        print(f"  Reading Time: {readability['estimated_reading_time_minutes']:.1f} minutes")
        
        # Keywords
        keywords = analysis['keywords']
        print(f"\nTOP KEYWORDS:")
        for word, count in keywords['top_keywords'][:5]:
            density = keywords['keyword_densities'][word]
            print(f"  {word}: {count} times ({density:.1f}%)")

if __name__ == "__main__":
    main()