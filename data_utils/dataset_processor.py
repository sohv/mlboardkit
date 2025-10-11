#!/usr/bin/env python3
"""
Dataset Processing Utilities for ML/AI Workflows

Advanced dataset manipulation including sampling, augmentation, quality checks,
and format conversion specifically designed for ML/AI training pipelines.

Usage:
    python3 dataset_processor.py sample dataset.csv --size 1000 --method random
    python3 dataset_processor.py augment images/ --techniques flip,rotate,noise
    python3 dataset_processor.py quality-check dataset.jsonl --report quality_report.json
    python3 dataset_processor.py balance dataset.csv label_column --method oversample
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import random
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from collections import Counter
import hashlib
import re


class DatasetSampler:
    """Advanced dataset sampling techniques."""
    
    @staticmethod
    def random_sample(data: pd.DataFrame, 
                     size: Union[int, float], 
                     random_state: int = 42) -> pd.DataFrame:
        """Random sampling with optional stratification."""
        if isinstance(size, float):
            if not 0 < size <= 1:
                raise ValueError("Float size must be between 0 and 1")
            size = int(len(data) * size)
        
        if size > len(data):
            raise ValueError(f"Sample size {size} larger than dataset {len(data)}")
        
        return data.sample(n=size, random_state=random_state)
    
    @staticmethod
    def stratified_sample(data: pd.DataFrame, 
                         size: Union[int, float],
                         stratify_column: str,
                         random_state: int = 42) -> pd.DataFrame:
        """Stratified sampling maintaining class distribution."""
        from sklearn.model_selection import train_test_split
        
        if stratify_column not in data.columns:
            raise ValueError(f"Stratify column '{stratify_column}' not found")
        
        if isinstance(size, float):
            test_size = 1 - size
        else:
            test_size = 1 - (size / len(data))
        
        sample, _ = train_test_split(
            data, 
            test_size=test_size,
            stratify=data[stratify_column],
            random_state=random_state
        )
        
        return sample
    
    @staticmethod
    def reservoir_sample(data_iterator, 
                        size: int, 
                        random_state: int = 42) -> List[Any]:
        """Reservoir sampling for large datasets that don't fit in memory."""
        random.seed(random_state)
        reservoir = []
        
        for i, item in enumerate(data_iterator):
            if i < size:
                reservoir.append(item)
            else:
                # Randomly replace elements in the reservoir
                j = random.randint(0, i)
                if j < size:
                    reservoir[j] = item
        
        return reservoir
    
    @staticmethod
    def time_based_sample(data: pd.DataFrame,
                         date_column: str,
                         start_date: str = None,
                         end_date: str = None,
                         sample_rate: str = 'D') -> pd.DataFrame:
        """Time-based sampling for temporal data."""
        data[date_column] = pd.to_datetime(data[date_column])
        
        # Filter by date range if specified
        if start_date:
            data = data[data[date_column] >= start_date]
        if end_date:
            data = data[data[date_column] <= end_date]
        
        # Sample at specified rate (D=daily, W=weekly, M=monthly, etc.)
        data = data.set_index(date_column)
        sampled = data.groupby(pd.Grouper(freq=sample_rate)).apply(
            lambda x: x.sample(min(len(x), 1)) if len(x) > 0 else x
        ).reset_index(drop=True)
        
        return sampled.reset_index()


class DatasetBalancer:
    """Balance datasets for classification tasks."""
    
    @staticmethod
    def analyze_imbalance(data: pd.DataFrame, label_column: str) -> Dict[str, Any]:
        """Analyze class imbalance in the dataset."""
        if label_column not in data.columns:
            raise ValueError(f"Label column '{label_column}' not found")
        
        class_counts = data[label_column].value_counts()
        total_samples = len(data)
        
        analysis = {
            'total_samples': total_samples,
            'num_classes': len(class_counts),
            'class_distribution': class_counts.to_dict(),
            'class_percentages': (class_counts / total_samples * 100).to_dict(),
            'imbalance_ratio': class_counts.max() / class_counts.min(),
            'is_balanced': class_counts.max() / class_counts.min() <= 2.0
        }
        
        return analysis
    
    @staticmethod
    def oversample(data: pd.DataFrame, 
                  label_column: str, 
                  random_state: int = 42) -> pd.DataFrame:
        """Oversample minority classes to balance the dataset."""
        from sklearn.utils import resample
        
        # Get class counts
        class_counts = data[label_column].value_counts()
        max_count = class_counts.max()
        
        # Oversample each class to match the majority class
        balanced_data = []
        
        for class_label in class_counts.index:
            class_data = data[data[label_column] == class_label]
            
            if len(class_data) < max_count:
                # Oversample this class
                oversampled = resample(
                    class_data,
                    replace=True,
                    n_samples=max_count,
                    random_state=random_state
                )
                balanced_data.append(oversampled)
            else:
                balanced_data.append(class_data)
        
        return pd.concat(balanced_data, ignore_index=True).sample(frac=1, random_state=random_state)
    
    @staticmethod
    def undersample(data: pd.DataFrame, 
                   label_column: str, 
                   random_state: int = 42) -> pd.DataFrame:
        """Undersample majority classes to balance the dataset."""
        from sklearn.utils import resample
        
        # Get class counts
        class_counts = data[label_column].value_counts()
        min_count = class_counts.min()
        
        # Undersample each class to match the minority class
        balanced_data = []
        
        for class_label in class_counts.index:
            class_data = data[data[label_column] == class_label]
            
            if len(class_data) > min_count:
                # Undersample this class
                undersampled = resample(
                    class_data,
                    replace=False,
                    n_samples=min_count,
                    random_state=random_state
                )
                balanced_data.append(undersampled)
            else:
                balanced_data.append(class_data)
        
        return pd.concat(balanced_data, ignore_index=True).sample(frac=1, random_state=random_state)
    
    @staticmethod
    def smote_oversample(data: pd.DataFrame, 
                        label_column: str, 
                        random_state: int = 42) -> pd.DataFrame:
        """Use SMOTE for synthetic oversampling."""
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            raise ImportError("Please install imbalanced-learn: pip install imbalanced-learn")
        
        # Separate features and labels
        X = data.drop(columns=[label_column])
        y = data[label_column]
        
        # Apply SMOTE
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Combine back into DataFrame
        result = pd.DataFrame(X_resampled, columns=X.columns)
        result[label_column] = y_resampled
        
        return result


class DatasetQualityChecker:
    """Comprehensive dataset quality assessment."""
    
    @staticmethod
    def check_duplicates(data: pd.DataFrame, 
                        columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Check for duplicate rows."""
        total_rows = len(data)
        
        if columns:
            duplicate_mask = data.duplicated(subset=columns)
        else:
            duplicate_mask = data.duplicated()
        
        num_duplicates = duplicate_mask.sum()
        duplicate_percentage = (num_duplicates / total_rows) * 100 if total_rows > 0 else 0
        
        return {
            'total_rows': total_rows,
            'duplicate_rows': int(num_duplicates),
            'duplicate_percentage': duplicate_percentage,
            'unique_rows': total_rows - num_duplicates,
            'duplicate_indices': duplicate_mask[duplicate_mask].index.tolist()
        }
    
    @staticmethod
    def check_missing_values(data: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive missing value analysis."""
        missing_info = {}
        
        for column in data.columns:
            null_count = data[column].isnull().sum()
            null_percentage = (null_count / len(data)) * 100
            
            missing_info[column] = {
                'missing_count': int(null_count),
                'missing_percentage': null_percentage,
                'data_type': str(data[column].dtype),
                'non_missing_count': len(data) - null_count
            }
        
        # Overall statistics
        total_missing = sum(info['missing_count'] for info in missing_info.values())
        total_cells = len(data) * len(data.columns)
        
        return {
            'column_details': missing_info,
            'total_missing_values': total_missing,
            'overall_missing_percentage': (total_missing / total_cells) * 100 if total_cells > 0 else 0,
            'columns_with_missing': [col for col, info in missing_info.items() if info['missing_count'] > 0]
        }
    
    @staticmethod
    def check_data_consistency(data: pd.DataFrame) -> Dict[str, Any]:
        """Check data consistency and format issues."""
        issues = []
        
        for column in data.columns:
            col_data = data[column].dropna()
            
            if col_data.dtype == 'object':
                # Check for mixed case issues
                if col_data.astype(str).str.lower().nunique() < col_data.nunique():
                    issues.append({
                        'column': column,
                        'issue': 'mixed_case',
                        'description': 'Column contains mixed case values'
                    })
                
                # Check for leading/trailing whitespace
                if col_data.astype(str).str.strip().nunique() < col_data.nunique():
                    issues.append({
                        'column': column,
                        'issue': 'whitespace',
                        'description': 'Column contains leading/trailing whitespace'
                    })
                
                # Check for date-like strings
                date_pattern = r'\d{4}-\d{2}-\d{2}'
                if col_data.astype(str).str.match(date_pattern).any():
                    try:
                        pd.to_datetime(col_data.iloc[0])
                        issues.append({
                            'column': column,
                            'issue': 'date_as_string',
                            'description': 'Column contains date strings that could be datetime'
                        })
                    except:
                        pass
        
        return {
            'issues': issues,
            'num_issues': len(issues)
        }
    
    @staticmethod
    def check_outliers(data: pd.DataFrame, 
                      numeric_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Detect outliers in numeric columns."""
        if numeric_columns is None:
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_info = {}
        
        for column in numeric_columns:
            if column in data.columns:
                col_data = data[column].dropna()
                
                if len(col_data) > 0:
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                    outlier_percentage = (len(outliers) / len(col_data)) * 100
                    
                    outlier_info[column] = {
                        'outlier_count': len(outliers),
                        'outlier_percentage': outlier_percentage,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound,
                        'outlier_indices': outliers.index.tolist()
                    }
        
        return outlier_info
    
    @staticmethod
    def generate_quality_report(data: pd.DataFrame, 
                               output_file: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        report = {
            'dataset_info': {
                'shape': data.shape,
                'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
                'dtypes': data.dtypes.value_counts().to_dict()
            },
            'duplicates': DatasetQualityChecker.check_duplicates(data),
            'missing_values': DatasetQualityChecker.check_missing_values(data),
            'consistency': DatasetQualityChecker.check_data_consistency(data),
            'outliers': DatasetQualityChecker.check_outliers(data)
        }
        
        # Calculate overall quality score
        duplicate_score = max(0, 100 - report['duplicates']['duplicate_percentage'])
        missing_score = max(0, 100 - report['missing_values']['overall_missing_percentage'])
        consistency_score = max(0, 100 - (report['consistency']['num_issues'] * 10))
        
        report['quality_score'] = {
            'overall': (duplicate_score + missing_score + consistency_score) / 3,
            'duplicate_score': duplicate_score,
            'missing_score': missing_score,
            'consistency_score': consistency_score
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        return report


class TextDataAugmenter:
    """Text data augmentation techniques."""
    
    @staticmethod
    def synonym_replacement(text: str, num_replacements: int = 1) -> str:
        """Replace words with synonyms (requires nltk)."""
        try:
            import nltk
            from nltk.corpus import wordnet
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except ImportError:
            return text
        
        words = text.split()
        new_words = words.copy()
        
        for _ in range(num_replacements):
            random_word_idx = random.randint(0, len(words) - 1)
            random_word = words[random_word_idx]
            
            synonyms = []
            for syn in wordnet.synsets(random_word):
                for lemma in syn.lemmas():
                    if lemma.name() != random_word:
                        synonyms.append(lemma.name().replace('_', ' '))
            
            if synonyms:
                new_words[random_word_idx] = random.choice(synonyms)
        
        return ' '.join(new_words)
    
    @staticmethod
    def random_insertion(text: str, num_insertions: int = 1) -> str:
        """Randomly insert words."""
        words = text.split()
        
        for _ in range(num_insertions):
            # Insert a random word from the text
            if words:
                random_word = random.choice(words)
                random_idx = random.randint(0, len(words))
                words.insert(random_idx, random_word)
        
        return ' '.join(words)
    
    @staticmethod
    def random_deletion(text: str, deletion_prob: float = 0.1) -> str:
        """Randomly delete words."""
        words = text.split()
        
        if len(words) == 1:
            return text
        
        new_words = []
        for word in words:
            if random.random() > deletion_prob:
                new_words.append(word)
        
        if not new_words:
            return random.choice(words)
        
        return ' '.join(new_words)
    
    @staticmethod
    def random_swap(text: str, num_swaps: int = 1) -> str:
        """Randomly swap words."""
        words = text.split()
        
        for _ in range(num_swaps):
            if len(words) >= 2:
                idx1, idx2 = random.sample(range(len(words)), 2)
                words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)


def main():
    parser = argparse.ArgumentParser(description="Dataset processing utilities")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Sample command
    sample_parser = subparsers.add_parser('sample', help='Sample dataset')
    sample_parser.add_argument('input', help='Input dataset file')
    sample_parser.add_argument('--output', help='Output file (default: sampled_<input>)')
    sample_parser.add_argument('--size', type=str, required=True, 
                              help='Sample size (integer or float 0-1)')
    sample_parser.add_argument('--method', choices=['random', 'stratified', 'time'], 
                              default='random', help='Sampling method')
    sample_parser.add_argument('--stratify-column', help='Column for stratified sampling')
    sample_parser.add_argument('--date-column', help='Date column for time-based sampling')
    sample_parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Balance command
    balance_parser = subparsers.add_parser('balance', help='Balance dataset classes')
    balance_parser.add_argument('input', help='Input dataset file')
    balance_parser.add_argument('label_column', help='Label column name')
    balance_parser.add_argument('--output', help='Output file (default: balanced_<input>)')
    balance_parser.add_argument('--method', choices=['oversample', 'undersample', 'smote'], 
                               default='oversample', help='Balancing method')
    balance_parser.add_argument('--analyze-only', action='store_true', 
                               help='Only analyze imbalance, don\'t balance')

    # Quality check command
    quality_parser = subparsers.add_parser('quality-check', help='Check dataset quality')
    quality_parser.add_argument('input', help='Input dataset file')
    quality_parser.add_argument('--report', help='Save detailed report to file')
    quality_parser.add_argument('--fix-issues', action='store_true', 
                               help='Attempt to fix common issues')

    # Augment command
    augment_parser = subparsers.add_parser('augment', help='Augment text dataset')
    augment_parser.add_argument('input', help='Input dataset file')
    augment_parser.add_argument('text_column', help='Text column to augment')
    augment_parser.add_argument('--output', help='Output file (default: augmented_<input>)')
    augment_parser.add_argument('--techniques', 
                               choices=['synonym', 'insertion', 'deletion', 'swap'],
                               nargs='+', default=['synonym'], help='Augmentation techniques')
    augment_parser.add_argument('--multiplier', type=int, default=2, 
                               help='How many augmented samples per original')

    # Deduplicate command
    dedup_parser = subparsers.add_parser('deduplicate', help='Remove duplicates')
    dedup_parser.add_argument('input', help='Input dataset file')
    dedup_parser.add_argument('--output', help='Output file (default: dedup_<input>)')
    dedup_parser.add_argument('--columns', nargs='*', 
                             help='Specific columns to check for duplicates')
    dedup_parser.add_argument('--keep', choices=['first', 'last'], default='first',
                             help='Which duplicate to keep')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == 'sample':
            print(f"📊 Sampling dataset: {args.input}")
            
            # Load data
            data = pd.read_csv(args.input) if args.input.endswith('.csv') else pd.read_json(args.input, lines=True)
            original_size = len(data)
            
            # Parse size
            try:
                size = float(args.size)
                if size <= 1:
                    size = int(original_size * size)
                else:
                    size = int(size)
            except ValueError:
                print("Error: Size must be a number")
                return
            
            # Sample data
            if args.method == 'random':
                sampled = DatasetSampler.random_sample(data, size, args.seed)
            elif args.method == 'stratified':
                if not args.stratify_column:
                    print("Error: --stratify-column required for stratified sampling")
                    return
                sampled = DatasetSampler.stratified_sample(data, size, args.stratify_column, args.seed)
            elif args.method == 'time':
                if not args.date_column:
                    print("Error: --date-column required for time-based sampling")
                    return
                sampled = DatasetSampler.time_based_sample(data, args.date_column)
            
            # Save output
            output_file = args.output or f"sampled_{Path(args.input).name}"
            sampled.to_csv(output_file, index=False)
            
            print(f"✅ Sampling complete: {original_size} → {len(sampled)} rows")
            print(f"📁 Output saved: {output_file}")

        elif args.command == 'balance':
            print(f"⚖️  Processing dataset: {args.input}")
            
            # Load data
            data = pd.read_csv(args.input) if args.input.endswith('.csv') else pd.read_json(args.input, lines=True)
            
            # Analyze imbalance
            analysis = DatasetBalancer.analyze_imbalance(data, args.label_column)
            print(f"📊 Dataset Analysis:")
            print(f"  Total samples: {analysis['total_samples']}")
            print(f"  Number of classes: {analysis['num_classes']}")
            print(f"  Imbalance ratio: {analysis['imbalance_ratio']:.2f}")
            print(f"  Is balanced: {analysis['is_balanced']}")
            
            for class_label, count in analysis['class_distribution'].items():
                percentage = analysis['class_percentages'][class_label]
                print(f"    {class_label}: {count} ({percentage:.1f}%)")
            
            if args.analyze_only:
                return
            
            # Balance the dataset
            if args.method == 'oversample':
                balanced = DatasetBalancer.oversample(data, args.label_column)
            elif args.method == 'undersample':
                balanced = DatasetBalancer.undersample(data, args.label_column)
            elif args.method == 'smote':
                balanced = DatasetBalancer.smote_oversample(data, args.label_column)
            
            # Save output
            output_file = args.output or f"balanced_{Path(args.input).name}"
            balanced.to_csv(output_file, index=False)
            
            print(f"✅ Balancing complete: {len(data)} → {len(balanced)} rows")
            print(f"📁 Output saved: {output_file}")

        elif args.command == 'quality-check':
            print(f"🔍 Checking quality of: {args.input}")
            
            # Load data
            data = pd.read_csv(args.input) if args.input.endswith('.csv') else pd.read_json(args.input, lines=True)
            
            # Generate quality report
            report = DatasetQualityChecker.generate_quality_report(data, args.report)
            
            # Print summary
            print(f"📊 Dataset Quality Report:")
            print(f"  Overall Quality Score: {report['quality_score']['overall']:.1f}/100")
            print(f"  Shape: {report['dataset_info']['shape']}")
            print(f"  Memory Usage: {report['dataset_info']['memory_usage_mb']:.2f} MB")
            print(f"  Duplicates: {report['duplicates']['duplicate_rows']} ({report['duplicates']['duplicate_percentage']:.1f}%)")
            print(f"  Missing Values: {report['missing_values']['total_missing_values']} ({report['missing_values']['overall_missing_percentage']:.1f}%)")
            print(f"  Consistency Issues: {report['consistency']['num_issues']}")
            
            if args.report:
                print(f"📁 Detailed report saved: {args.report}")

        elif args.command == 'augment':
            print(f"🔄 Augmenting text data: {args.input}")
            
            # Load data
            data = pd.read_csv(args.input) if args.input.endswith('.csv') else pd.read_json(args.input, lines=True)
            
            if args.text_column not in data.columns:
                print(f"Error: Column '{args.text_column}' not found")
                return
            
            augmented_rows = []
            
            for _, row in data.iterrows():
                # Keep original
                augmented_rows.append(row)
                
                # Create augmented versions
                for _ in range(args.multiplier - 1):
                    new_row = row.copy()
                    text = str(row[args.text_column])
                    
                    for technique in args.techniques:
                        if technique == 'synonym':
                            text = TextDataAugmenter.synonym_replacement(text)
                        elif technique == 'insertion':
                            text = TextDataAugmenter.random_insertion(text)
                        elif technique == 'deletion':
                            text = TextDataAugmenter.random_deletion(text)
                        elif technique == 'swap':
                            text = TextDataAugmenter.random_swap(text)
                    
                    new_row[args.text_column] = text
                    augmented_rows.append(new_row)
            
            # Create augmented dataset
            augmented_data = pd.DataFrame(augmented_rows)
            
            # Save output
            output_file = args.output or f"augmented_{Path(args.input).name}"
            augmented_data.to_csv(output_file, index=False)
            
            print(f"✅ Augmentation complete: {len(data)} → {len(augmented_data)} rows")
            print(f"📁 Output saved: {output_file}")

        elif args.command == 'deduplicate':
            print(f"🧹 Removing duplicates from: {args.input}")
            
            # Load data
            data = pd.read_csv(args.input) if args.input.endswith('.csv') else pd.read_json(args.input, lines=True)
            original_size = len(data)
            
            # Remove duplicates
            if args.columns:
                deduped = data.drop_duplicates(subset=args.columns, keep=args.keep)
            else:
                deduped = data.drop_duplicates(keep=args.keep)
            
            # Save output
            output_file = args.output or f"dedup_{Path(args.input).name}"
            deduped.to_csv(output_file, index=False)
            
            removed_count = original_size - len(deduped)
            print(f"✅ Deduplication complete: {original_size} → {len(deduped)} rows")
            print(f"🗑️  Removed {removed_count} duplicates")
            print(f"📁 Output saved: {output_file}")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()