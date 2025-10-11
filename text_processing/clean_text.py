#!/usr/bin/env python3
"""
clean_text.py

Comprehensive text cleaning and preprocessing utilities for NLP and ML.
Includes advanced cleaning, normalization, and preprocessing pipelines.
"""

import argparse
import re
import json
import html
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass

try:
    import ftfy
    HAS_FTFY = True
except ImportError:
    HAS_FTFY = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

try:
    import langdetect
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False


@dataclass
class CleaningConfig:
    """Configuration for text cleaning operations"""
    # Basic cleaning
    remove_html: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    remove_phone_numbers: bool = True
    remove_social_handles: bool = True
    
    # Unicode and encoding
    fix_encoding: bool = True
    normalize_unicode: bool = True
    remove_non_printable: bool = True
    
    # Whitespace and formatting
    normalize_whitespace: bool = True
    remove_extra_newlines: bool = True
    strip_whitespace: bool = True
    
    # Case and formatting
    lowercase: bool = False
    remove_accents: bool = False
    normalize_quotes: bool = True
    normalize_dashes: bool = True
    
    # Content filtering
    min_length: int = 0
    max_length: int = 0  # 0 means no limit
    remove_duplicates: bool = False
    language_filter: Optional[str] = None
    
    # Custom patterns
    custom_patterns: List[str] = None
    replacement_dict: Dict[str, str] = None
    
    def __post_init__(self):
        if self.custom_patterns is None:
            self.custom_patterns = []
        if self.replacement_dict is None:
            self.replacement_dict = {}


class TextCleaner:
    """Comprehensive text cleaning utility"""
    
    def __init__(self, config: CleaningConfig = None):
        self.config = config or CleaningConfig()
        
        # Compile common regex patterns
        self.patterns = {
            'url': re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'),
            'social_handle': re.compile(r'@\w+|#\w+'),
            'extra_whitespace': re.compile(r'\s+'),
            'extra_newlines': re.compile(r'\n{3,}'),
            'non_printable': re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]'),
            'quotes': re.compile(r'[""''`´]'),
            'dashes': re.compile(r'[—–−]'),
        }
        
        # Compile custom patterns
        self.custom_compiled_patterns = []
        for pattern in self.config.custom_patterns:
            try:
                self.custom_compiled_patterns.append(re.compile(pattern))
            except re.error as e:
                print(f"Warning: Invalid regex pattern '{pattern}': {e}")
    
    def clean_html(self, text: str) -> str:
        """Remove HTML tags and decode HTML entities"""
        if not self.config.remove_html:
            return text
        
        # Use BeautifulSoup if available for better HTML cleaning
        if HAS_BS4:
            try:
                soup = BeautifulSoup(text, 'html.parser')
                text = soup.get_text(separator=' ')
            except Exception:
                # Fallback to simple regex
                text = re.sub(r'<[^>]+>', ' ', text)
        else:
            # Simple HTML tag removal
            text = re.sub(r'<[^>]+>', ' ', text)
        
        # Decode HTML entities
        text = html.unescape(text)
        return text
    
    def fix_encoding_issues(self, text: str) -> str:
        """Fix common encoding issues"""
        if not self.config.fix_encoding:
            return text
        
        if HAS_FTFY:
            try:
                text = ftfy.fix_text(text)
            except Exception:
                pass
        
        # Manual fixes for common issues
        replacements = {
            'â€™': "'",
            'â€œ': '"',
            'â€': '"',
            'â€¦': '...',
            'â€"': '—',
            'â€"': '–',
            'Ã¡': 'á',
            'Ã©': 'é',
            'Ã­': 'í',
            'Ã³': 'ó',
            'Ãº': 'ú',
            'Ã±': 'ñ',
        }
        
        for wrong, correct in replacements.items():
            text = text.replace(wrong, correct)
        
        return text
    
    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters"""
        if not self.config.normalize_unicode:
            return text
        
        # NFKD normalization: canonical decomposition, then canonical combining
        text = unicodedata.normalize('NFKD', text)
        
        if self.config.remove_accents:
            # Remove combining characters (accents)
            text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        
        return text
    
    def remove_patterns(self, text: str) -> str:
        """Remove various patterns based on configuration"""
        if self.config.remove_urls:
            text = self.patterns['url'].sub(' ', text)
        
        if self.config.remove_emails:
            text = self.patterns['email'].sub(' ', text)
        
        if self.config.remove_phone_numbers:
            text = self.patterns['phone'].sub(' ', text)
        
        if self.config.remove_social_handles:
            text = self.patterns['social_handle'].sub(' ', text)
        
        if self.config.remove_non_printable:
            text = self.patterns['non_printable'].sub(' ', text)
        
        # Custom patterns
        for pattern in self.custom_compiled_patterns:
            text = pattern.sub(' ', text)
        
        return text
    
    def normalize_formatting(self, text: str) -> str:
        """Normalize formatting and punctuation"""
        if self.config.normalize_quotes:
            text = self.patterns['quotes'].sub('"', text)
        
        if self.config.normalize_dashes:
            text = self.patterns['dashes'].sub('-', text)
        
        return text
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace characters"""
        if self.config.remove_extra_newlines:
            text = self.patterns['extra_newlines'].sub('\n\n', text)
        
        if self.config.normalize_whitespace:
            text = self.patterns['extra_whitespace'].sub(' ', text)
        
        if self.config.strip_whitespace:
            text = text.strip()
        
        return text
    
    def apply_replacements(self, text: str) -> str:
        """Apply custom replacement dictionary"""
        for old, new in self.config.replacement_dict.items():
            text = text.replace(old, new)
        return text
    
    def filter_by_language(self, text: str) -> Optional[str]:
        """Filter text by language if configured"""
        if not self.config.language_filter or not HAS_LANGDETECT:
            return text
        
        try:
            detected_lang = langdetect.detect(text)
            if detected_lang != self.config.language_filter:
                return None
        except:
            # If language detection fails, keep the text
            pass
        
        return text
    
    def filter_by_length(self, text: str) -> Optional[str]:
        """Filter text by length constraints"""
        text_length = len(text)
        
        if self.config.min_length > 0 and text_length < self.config.min_length:
            return None
        
        if self.config.max_length > 0 and text_length > self.config.max_length:
            return None
        
        return text
    
    def clean_single(self, text: str) -> Optional[str]:
        """Clean a single text string"""
        if not isinstance(text, str):
            return None
        
        # Apply cleaning steps in order
        text = self.clean_html(text)
        text = self.fix_encoding_issues(text)
        text = self.normalize_unicode(text)
        text = self.remove_patterns(text)
        text = self.normalize_formatting(text)
        text = self.apply_replacements(text)
        text = self.normalize_whitespace(text)
        
        # Apply case transformation
        if self.config.lowercase:
            text = text.lower()
        
        # Apply filters
        text = self.filter_by_language(text)
        if text is None:
            return None
        
        text = self.filter_by_length(text)
        if text is None:
            return None
        
        return text
    
    def clean_batch(self, texts: List[str]) -> List[str]:
        """Clean a batch of texts"""
        cleaned_texts = []
        
        for text in texts:
            cleaned = self.clean_single(text)
            if cleaned is not None:
                cleaned_texts.append(cleaned)
        
        # Remove duplicates if configured
        if self.config.remove_duplicates:
            seen = set()
            unique_texts = []
            for text in cleaned_texts:
                if text not in seen:
                    seen.add(text)
                    unique_texts.append(text)
            cleaned_texts = unique_texts
        
        return cleaned_texts
    
    def clean_file(self, input_file: str, output_file: str = None, 
                   input_format: str = 'txt', output_format: str = None):
        """Clean text from file"""
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        output_format = output_format or input_format
        output_file = output_file or str(input_path.with_suffix(f'.cleaned.{input_path.suffix.lstrip(".")}'))
        
        if input_format == 'txt':
            with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
                if output_format == 'txt':
                    # Process line by line for large files
                    with open(output_file, 'w', encoding='utf-8') as out_f:
                        for line in f:
                            cleaned = self.clean_single(line.strip())
                            if cleaned:
                                out_f.write(cleaned + '\n')
                else:
                    # Load all lines and save as JSON
                    lines = [line.strip() for line in f]
                    cleaned_lines = self.clean_batch(lines)
                    with open(output_file, 'w', encoding='utf-8') as out_f:
                        json.dump(cleaned_lines, out_f, indent=2, ensure_ascii=False)
        
        elif input_format == 'json':
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                cleaned_data = self.clean_batch(data)
            elif isinstance(data, dict):
                # Clean all string values in dict
                cleaned_data = {}
                for key, value in data.items():
                    if isinstance(value, str):
                        cleaned = self.clean_single(value)
                        if cleaned:
                            cleaned_data[key] = cleaned
                    else:
                        cleaned_data[key] = value
            else:
                cleaned_data = self.clean_single(str(data))
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
        
        print(f"Cleaned text saved to: {output_file}")
    
    def get_cleaning_stats(self, original_texts: List[str], 
                          cleaned_texts: List[str]) -> Dict[str, Any]:
        """Get statistics about the cleaning process"""
        original_count = len(original_texts)
        cleaned_count = len(cleaned_texts)
        
        original_lengths = [len(text) for text in original_texts]
        cleaned_lengths = [len(text) for text in cleaned_texts]
        
        stats = {
            'original_count': original_count,
            'cleaned_count': cleaned_count,
            'removed_count': original_count - cleaned_count,
            'removal_rate': (original_count - cleaned_count) / original_count if original_count > 0 else 0,
            'average_original_length': sum(original_lengths) / len(original_lengths) if original_lengths else 0,
            'average_cleaned_length': sum(cleaned_lengths) / len(cleaned_lengths) if cleaned_lengths else 0,
            'total_characters_removed': sum(original_lengths) - sum(cleaned_lengths)
        }
        
        return stats


def create_cleaning_config_from_args(args) -> CleaningConfig:
    """Create cleaning configuration from command line arguments"""
    return CleaningConfig(
        remove_html=args.remove_html,
        remove_urls=args.remove_urls,
        remove_emails=args.remove_emails,
        remove_phone_numbers=args.remove_phone,
        remove_social_handles=args.remove_social,
        fix_encoding=args.fix_encoding,
        normalize_unicode=args.normalize_unicode,
        remove_non_printable=args.remove_non_printable,
        normalize_whitespace=args.normalize_whitespace,
        remove_extra_newlines=args.remove_extra_newlines,
        strip_whitespace=args.strip_whitespace,
        lowercase=args.lowercase,
        remove_accents=args.remove_accents,
        normalize_quotes=args.normalize_quotes,
        normalize_dashes=args.normalize_dashes,
        min_length=args.min_length,
        max_length=args.max_length,
        remove_duplicates=args.remove_duplicates,
        language_filter=args.language_filter,
        custom_patterns=args.custom_patterns or [],
        replacement_dict=json.loads(args.replacements) if args.replacements else {}
    )


def main():
    parser = argparse.ArgumentParser(description="Comprehensive text cleaning utility")
    
    # Input/Output
    parser.add_argument('input_file', help='Input file path')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--input-format', choices=['txt', 'json'], default='txt',
                       help='Input file format')
    parser.add_argument('--output-format', choices=['txt', 'json'],
                       help='Output file format (default: same as input)')
    
    # Basic cleaning options
    parser.add_argument('--remove-html', action='store_true', default=True,
                       help='Remove HTML tags and entities')
    parser.add_argument('--remove-urls', action='store_true', default=True,
                       help='Remove URLs')
    parser.add_argument('--remove-emails', action='store_true', default=True,
                       help='Remove email addresses')
    parser.add_argument('--remove-phone', action='store_true', default=True,
                       help='Remove phone numbers')
    parser.add_argument('--remove-social', action='store_true', default=True,
                       help='Remove social media handles (@user, #hashtag)')
    
    # Unicode and encoding
    parser.add_argument('--fix-encoding', action='store_true', default=True,
                       help='Fix encoding issues')
    parser.add_argument('--normalize-unicode', action='store_true', default=True,
                       help='Normalize Unicode characters')
    parser.add_argument('--remove-non-printable', action='store_true', default=True,
                       help='Remove non-printable characters')
    
    # Whitespace and formatting
    parser.add_argument('--normalize-whitespace', action='store_true', default=True,
                       help='Normalize whitespace')
    parser.add_argument('--remove-extra-newlines', action='store_true', default=True,
                       help='Remove extra newlines')
    parser.add_argument('--strip-whitespace', action='store_true', default=True,
                       help='Strip leading/trailing whitespace')
    
    # Case and formatting
    parser.add_argument('--lowercase', action='store_true',
                       help='Convert text to lowercase')
    parser.add_argument('--remove-accents', action='store_true',
                       help='Remove accents from characters')
    parser.add_argument('--normalize-quotes', action='store_true', default=True,
                       help='Normalize quote characters')
    parser.add_argument('--normalize-dashes', action='store_true', default=True,
                       help='Normalize dash characters')
    
    # Content filtering
    parser.add_argument('--min-length', type=int, default=0,
                       help='Minimum text length')
    parser.add_argument('--max-length', type=int, default=0,
                       help='Maximum text length (0 = no limit)')
    parser.add_argument('--remove-duplicates', action='store_true',
                       help='Remove duplicate texts')
    parser.add_argument('--language-filter',
                       help='Keep only texts in specified language (ISO code)')
    
    # Custom options
    parser.add_argument('--custom-patterns', nargs='+',
                       help='Custom regex patterns to remove')
    parser.add_argument('--replacements',
                       help='JSON string of replacements {"old": "new"}')
    
    # Configuration
    parser.add_argument('--config', help='Load configuration from JSON file')
    parser.add_argument('--save-config', help='Save current configuration to file')
    parser.add_argument('--stats', action='store_true',
                       help='Show cleaning statistics')
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = CleaningConfig(**config_dict)
    else:
        config = create_cleaning_config_from_args(args)
    
    # Save configuration if requested
    if args.save_config:
        with open(args.save_config, 'w') as f:
            json.dump(config.__dict__, f, indent=2)
        print(f"Configuration saved to {args.save_config}")
        return
    
    # Create cleaner and process file
    cleaner = TextCleaner(config)
    
    if args.stats:
        # Load original texts for statistics
        if args.input_format == 'txt':
            with open(args.input_file, 'r', encoding='utf-8', errors='ignore') as f:
                original_texts = [line.strip() for line in f]
        else:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    original_texts = data
                else:
                    original_texts = [str(data)]
        
        cleaned_texts = cleaner.clean_batch(original_texts)
        stats = cleaner.get_cleaning_stats(original_texts, cleaned_texts)
        
        print("Cleaning Statistics:")
        print(f"  Original texts: {stats['original_count']}")
        print(f"  Cleaned texts: {stats['cleaned_count']}")
        print(f"  Removed texts: {stats['removed_count']} ({stats['removal_rate']:.1%})")
        print(f"  Avg length before: {stats['average_original_length']:.1f}")
        print(f"  Avg length after: {stats['average_cleaned_length']:.1f}")
        print(f"  Characters removed: {stats['total_characters_removed']}")
    
    # Clean the file
    cleaner.clean_file(
        args.input_file, 
        args.output, 
        args.input_format, 
        args.output_format
    )


if __name__ == "__main__":
    main()