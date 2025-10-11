#!/usr/bin/env python3
"""
augment_text.py

Text data augmentation using synonym replacement and simple transformations.
Supports back-translation if translation libraries are available.
"""

import argparse
import json
import random
import re
from typing import List, Dict, Any
from pathlib import Path


class TextAugmenter:
    def __init__(self):
        self.synonyms = {
            'good': ['excellent', 'great', 'wonderful', 'fantastic'],
            'bad': ['terrible', 'awful', 'horrible', 'poor'],
            'big': ['large', 'huge', 'enormous', 'massive'],
            'small': ['tiny', 'little', 'mini', 'compact'],
            'fast': ['quick', 'rapid', 'swift', 'speedy'],
            'slow': ['sluggish', 'gradual', 'leisurely', 'delayed']
        }
    
    def synonym_replacement(self, text: str, n_replacements: int = 1) -> str:
        words = text.split()
        for _ in range(n_replacements):
            if not words:
                break
            word_idx = random.randint(0, len(words) - 1)
            word = words[word_idx].lower().strip('.,!?')
            if word in self.synonyms:
                synonym = random.choice(self.synonyms[word])
                words[word_idx] = words[word_idx].replace(word, synonym)
        return ' '.join(words)
    
    def random_insertion(self, text: str, n_insertions: int = 1) -> str:
        words = text.split()
        for _ in range(n_insertions):
            if not words:
                break
            # Insert random synonym of existing word
            word = random.choice(words).lower().strip('.,!?')
            if word in self.synonyms:
                synonym = random.choice(self.synonyms[word])
                insert_pos = random.randint(0, len(words))
                words.insert(insert_pos, synonym)
        return ' '.join(words)
    
    def random_swap(self, text: str, n_swaps: int = 1) -> str:
        words = text.split()
        for _ in range(n_swaps):
            if len(words) < 2:
                break
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        return ' '.join(words)
    
    def random_deletion(self, text: str, p_delete: float = 0.1) -> str:
        words = text.split()
        if len(words) == 1:
            return text
        new_words = [word for word in words if random.random() > p_delete]
        return ' '.join(new_words) if new_words else text
    
    def back_translate(self, text: str, intermediate_lang: str = 'es') -> str:
        """Attempt back-translation if googletrans is available."""
        try:
            from googletrans import Translator
            translator = Translator()
            # Translate to intermediate language then back to English
            intermediate = translator.translate(text, dest=intermediate_lang).text
            back_translated = translator.translate(intermediate, dest='en').text
            return back_translated
        except ImportError:
            print("googletrans not available, skipping back-translation")
            return text
        except Exception as e:
            print(f"Back-translation failed: {e}")
            return text
    
    def augment_text(self, text: str, techniques: List[str], n_aug: int = 1) -> List[str]:
        augmented = []
        for _ in range(n_aug):
            aug_text = text
            for technique in techniques:
                if technique == 'synonym':
                    aug_text = self.synonym_replacement(aug_text)
                elif technique == 'insertion':
                    aug_text = self.random_insertion(aug_text)
                elif technique == 'swap':
                    aug_text = self.random_swap(aug_text)
                elif technique == 'deletion':
                    aug_text = self.random_deletion(aug_text)
                elif technique == 'backtrans':
                    aug_text = self.back_translate(aug_text)
            augmented.append(aug_text)
        return augmented


def main():
    parser = argparse.ArgumentParser(description='Text data augmentation')
    parser.add_argument('--input', required=True, help='Input file (JSON list or text file)')
    parser.add_argument('--output', required=True, help='Output file')
    parser.add_argument('--techniques', nargs='+', 
                       choices=['synonym', 'insertion', 'swap', 'deletion', 'backtrans'],
                       default=['synonym'], help='Augmentation techniques')
    parser.add_argument('--multiplier', type=int, default=2, help='Augmentations per text')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    # Load input
    input_path = Path(args.input)
    if input_path.suffix.lower() == '.json':
        with open(input_path, 'r') as f:
            texts = json.load(f)
    else:
        with open(input_path, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]
    
    augmenter = TextAugmenter()
    
    # Augment texts
    all_texts = []
    for text in texts:
        all_texts.append(text)  # Original
        augmented = augmenter.augment_text(text, args.techniques, args.multiplier)
        all_texts.extend(augmented)
    
    # Save output
    output_path = Path(args.output)
    if output_path.suffix.lower() == '.json':
        with open(output_path, 'w') as f:
            json.dump(all_texts, f, indent=2)
    else:
        with open(output_path, 'w') as f:
            for text in all_texts:
                f.write(f"{text}\n")
    
    print(f"Augmented {len(texts)} texts to {len(all_texts)} (multiplier: {args.multiplier})")
    print(f"Saved to {output_path}")


if __name__ == '__main__':
    main()