#!/usr/bin/env python3
"""
Data Conversion Utilities for ML/AI Workflows

Convert between JSON, CSV, TSV, Parquet, and JSONL formats with schema validation,
data cleaning, and preprocessing options.

Usage:
    python3 data_converter.py convert input.json output.csv --format csv
    python3 data_converter.py validate data.csv --schema schema.json
    python3 data_converter.py clean messy_data.csv clean_data.csv --remove-duplicates --fill-missing
    python3 data_converter.py split dataset.csv --train 0.8 --val 0.1 --test 0.1
"""

import argparse
import json
import csv
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Any, List, Optional, Union
import re
from datetime import datetime
import logging


class DataValidator:
    """Validates data against schema and performs quality checks."""
    
    @staticmethod
    def validate_schema(data: pd.DataFrame, schema: Dict[str, Any]) -> List[str]:
        """Validate DataFrame against JSON schema."""
        issues = []
        
        if 'required_columns' in schema:
            missing_cols = set(schema['required_columns']) - set(data.columns)
            if missing_cols:
                issues.append(f"Missing required columns: {missing_cols}")
        
        if 'column_types' in schema:
            for col, expected_type in schema['column_types'].items():
                if col in data.columns:
                    actual_type = str(data[col].dtype)
                    if not DataValidator._type_matches(actual_type, expected_type):
                        issues.append(f"Column '{col}' type mismatch: expected {expected_type}, got {actual_type}")
        
        if 'constraints' in schema:
            for col, constraints in schema['constraints'].items():
                if col in data.columns:
                    if 'min_value' in constraints:
                        if data[col].min() < constraints['min_value']:
                            issues.append(f"Column '{col}' has values below minimum: {constraints['min_value']}")
                    
                    if 'max_value' in constraints:
                        if data[col].max() > constraints['max_value']:
                            issues.append(f"Column '{col}' has values above maximum: {constraints['max_value']}")
                    
                    if 'allowed_values' in constraints:
                        invalid_values = set(data[col].unique()) - set(constraints['allowed_values'])
                        if invalid_values:
                            issues.append(f"Column '{col}' has invalid values: {invalid_values}")
        
        return issues
    
    @staticmethod
    def _type_matches(actual: str, expected: str) -> bool:
        """Check if actual data type matches expected type."""
        type_mappings = {
            'string': ['object', 'string'],
            'integer': ['int64', 'int32', 'int16', 'int8'],
            'float': ['float64', 'float32', 'float16'],
            'boolean': ['bool'],
            'datetime': ['datetime64', 'datetime']
        }
        
        return actual in type_mappings.get(expected, [expected])
    
    @staticmethod
    def quality_check(data: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive data quality checks."""
        report = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
            'columns': {}
        }
        
        for col in data.columns:
            col_stats = {
                'dtype': str(data[col].dtype),
                'non_null_count': data[col].count(),
                'null_count': data[col].isnull().sum(),
                'null_percentage': (data[col].isnull().sum() / len(data)) * 100,
                'unique_count': data[col].nunique(),
                'duplicate_count': data[col].duplicated().sum()
            }
            
            if pd.api.types.is_numeric_dtype(data[col]):
                col_stats.update({
                    'min': data[col].min(),
                    'max': data[col].max(),
                    'mean': data[col].mean(),
                    'std': data[col].std(),
                    'median': data[col].median()
                })
            elif pd.api.types.is_string_dtype(data[col]):
                col_stats.update({
                    'avg_length': data[col].str.len().mean(),
                    'max_length': data[col].str.len().max(),
                    'min_length': data[col].str.len().min()
                })
            
            report['columns'][col] = col_stats
        
        return report


class DataCleaner:
    """Cleans and preprocesses data."""
    
    @staticmethod
    def remove_duplicates(data: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """Remove duplicate rows."""
        before_count = len(data)
        data = data.drop_duplicates(subset=subset)
        after_count = len(data)
        print(f"Removed {before_count - after_count} duplicate rows")
        return data
    
    @staticmethod
    def handle_missing_values(data: pd.DataFrame, strategy: str = 'drop', fill_value: Any = None) -> pd.DataFrame:
        """Handle missing values with different strategies."""
        if strategy == 'drop':
            before_count = len(data)
            data = data.dropna()
            after_count = len(data)
            print(f"Dropped {before_count - after_count} rows with missing values")
        
        elif strategy == 'fill':
            if fill_value is not None:
                data = data.fillna(fill_value)
            else:
                # Use different strategies for different column types
                for col in data.columns:
                    if pd.api.types.is_numeric_dtype(data[col]):
                        data[col] = data[col].fillna(data[col].median())
                    else:
                        data[col] = data[col].fillna(data[col].mode().iloc[0] if not data[col].mode().empty else 'Unknown')
        
        elif strategy == 'interpolate':
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = data[numeric_cols].interpolate()
        
        return data
    
    @staticmethod
    def standardize_text(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Standardize text columns."""
        for col in columns:
            if col in data.columns:
                # Convert to string and strip whitespace
                data[col] = data[col].astype(str).str.strip()
                
                # Remove extra whitespace
                data[col] = data[col].str.replace(r'\s+', ' ', regex=True)
                
                # Standardize case (optional)
                # data[col] = data[col].str.lower()
        
        return data
    
    @staticmethod
    def remove_outliers(data: pd.DataFrame, columns: List[str], method: str = 'iqr') -> pd.DataFrame:
        """Remove outliers using IQR or Z-score method."""
        before_count = len(data)
        
        for col in columns:
            if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                if method == 'iqr':
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
                
                elif method == 'zscore':
                    z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                    data = data[z_scores < 3]
        
        after_count = len(data)
        print(f"Removed {before_count - after_count} outlier rows")
        return data


class DataConverter:
    """Converts between different data formats."""
    
    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        """Load data from various formats."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        suffix = path.suffix.lower()
        
        try:
            if suffix == '.csv':
                return pd.read_csv(file_path)
            elif suffix == '.tsv':
                return pd.read_csv(file_path, sep='\t')
            elif suffix == '.json':
                return pd.read_json(file_path)
            elif suffix == '.jsonl':
                return pd.read_json(file_path, lines=True)
            elif suffix == '.parquet':
                return pd.read_parquet(file_path)
            elif suffix in ['.xlsx', '.xls']:
                return pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {suffix}")
        
        except Exception as e:
            raise ValueError(f"Failed to load {file_path}: {e}")
    
    @staticmethod
    def save_data(data: pd.DataFrame, file_path: str, format: Optional[str] = None):
        """Save data to various formats."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format is None:
            format = path.suffix.lower()[1:]  # Remove the dot
        
        try:
            if format == 'csv':
                data.to_csv(file_path, index=False)
            elif format == 'tsv':
                data.to_csv(file_path, sep='\t', index=False)
            elif format == 'json':
                data.to_json(file_path, orient='records', indent=2)
            elif format == 'jsonl':
                data.to_json(file_path, orient='records', lines=True)
            elif format == 'parquet':
                data.to_parquet(file_path, index=False)
            elif format in ['xlsx', 'xls']:
                data.to_excel(file_path, index=False)
            else:
                raise ValueError(f"Unsupported output format: {format}")
        
        except Exception as e:
            raise ValueError(f"Failed to save to {file_path}: {e}")
    
    @staticmethod
    def split_dataset(data: pd.DataFrame, 
                     train_ratio: float = 0.8, 
                     val_ratio: float = 0.1, 
                     test_ratio: float = 0.1,
                     stratify_column: Optional[str] = None,
                     random_state: int = 42) -> Dict[str, pd.DataFrame]:
        """Split dataset into train/validation/test sets."""
        from sklearn.model_selection import train_test_split
        
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        # Prepare stratification
        stratify = None
        if stratify_column and stratify_column in data.columns:
            stratify = data[stratify_column]
        
        # First split: separate train from (val + test)
        if val_ratio + test_ratio > 0:
            train_data, temp_data = train_test_split(
                data, 
                test_size=val_ratio + test_ratio,
                stratify=stratify,
                random_state=random_state
            )
        else:
            train_data = data.copy()
            temp_data = pd.DataFrame()
        
        # Second split: separate val from test
        if val_ratio > 0 and test_ratio > 0:
            val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
            val_data, test_data = train_test_split(
                temp_data,
                test_size=1 - val_ratio_adjusted,
                stratify=temp_data[stratify_column] if stratify_column else None,
                random_state=random_state
            )
        elif val_ratio > 0:
            val_data = temp_data.copy()
            test_data = pd.DataFrame()
        elif test_ratio > 0:
            test_data = temp_data.copy()
            val_data = pd.DataFrame()
        else:
            val_data = pd.DataFrame()
            test_data = pd.DataFrame()
        
        return {
            'train': train_data,
            'validation': val_data,
            'test': test_data
        }


def main():
    parser = argparse.ArgumentParser(description="Data conversion and processing utilities")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert between data formats')
    convert_parser.add_argument('input', help='Input file path')
    convert_parser.add_argument('output', help='Output file path')
    convert_parser.add_argument('--format', choices=['csv', 'tsv', 'json', 'jsonl', 'parquet', 'xlsx'],
                               help='Output format (auto-detected from extension if not specified)')

    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate data against schema')
    validate_parser.add_argument('data_file', help='Data file to validate')
    validate_parser.add_argument('--schema', help='JSON schema file')
    validate_parser.add_argument('--quality-check', action='store_true', help='Perform quality check')

    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Clean and preprocess data')
    clean_parser.add_argument('input', help='Input file path')
    clean_parser.add_argument('output', help='Output file path')
    clean_parser.add_argument('--remove-duplicates', action='store_true', help='Remove duplicate rows')
    clean_parser.add_argument('--handle-missing', choices=['drop', 'fill', 'interpolate'], 
                             help='Strategy for handling missing values')
    clean_parser.add_argument('--fill-value', help='Value to use when filling missing data')
    clean_parser.add_argument('--standardize-text', nargs='*', help='Text columns to standardize')
    clean_parser.add_argument('--remove-outliers', nargs='*', help='Numeric columns to remove outliers from')
    clean_parser.add_argument('--outlier-method', choices=['iqr', 'zscore'], default='iqr',
                             help='Method for outlier detection')

    # Split command
    split_parser = subparsers.add_parser('split', help='Split dataset into train/val/test')
    split_parser.add_argument('input', help='Input dataset file')
    split_parser.add_argument('--train', type=float, default=0.8, help='Training set ratio')
    split_parser.add_argument('--val', type=float, default=0.1, help='Validation set ratio')
    split_parser.add_argument('--test', type=float, default=0.1, help='Test set ratio')
    split_parser.add_argument('--stratify', help='Column to use for stratified splitting')
    split_parser.add_argument('--output-dir', default='.', help='Output directory for split files')
    split_parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    # Info command
    info_parser = subparsers.add_parser('info', help='Display dataset information')
    info_parser.add_argument('input', help='Input file path')
    info_parser.add_argument('--detailed', action='store_true', help='Show detailed statistics')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == 'convert':
            print(f"Converting {args.input} to {args.output}")
            data = DataConverter.load_data(args.input)
            DataConverter.save_data(data, args.output, args.format)
            print(f"Conversion completed: {len(data)} rows, {len(data.columns)} columns")

        elif args.command == 'validate':
            print(f"Validating {args.data_file}")
            data = DataConverter.load_data(args.data_file)
            
            if args.schema:
                with open(args.schema, 'r') as f:
                    schema = json.load(f)
                issues = DataValidator.validate_schema(data, schema)
                
                if issues:
                    print("Validation issues found:")
                    for issue in issues:
                        print(f"  - {issue}")
                    sys.exit(1)
                else:
                    print("Data validation passed!")
            
            if args.quality_check:
                report = DataValidator.quality_check(data)
                print("\nData Quality Report:")
                print(f"Total rows: {report['total_rows']}")
                print(f"Total columns: {report['total_columns']}")
                print(f"Memory usage: {report['memory_usage_mb']:.2f} MB")
                
                for col, stats in report['columns'].items():
                    print(f"\nColumn '{col}':")
                    print(f"  Type: {stats['dtype']}")
                    print(f"  Non-null: {stats['non_null_count']} ({100 - stats['null_percentage']:.1f}%)")
                    print(f"  Unique values: {stats['unique_count']}")

        elif args.command == 'clean':
            print(f"Cleaning {args.input}")
            data = DataConverter.load_data(args.input)
            original_count = len(data)
            
            if args.remove_duplicates:
                data = DataCleaner.remove_duplicates(data)
            
            if args.handle_missing:
                fill_val = args.fill_value
                if fill_val and fill_val.isdigit():
                    fill_val = float(fill_val)
                data = DataCleaner.handle_missing_values(data, args.handle_missing, fill_val)
            
            if args.standardize_text:
                data = DataCleaner.standardize_text(data, args.standardize_text)
                print(f"Standardized text in columns: {args.standardize_text}")
            
            if args.remove_outliers:
                data = DataCleaner.remove_outliers(data, args.remove_outliers, args.outlier_method)
            
            DataConverter.save_data(data, args.output)
            print(f"Cleaning completed: {original_count} -> {len(data)} rows")

        elif args.command == 'split':
            print(f"Splitting {args.input}")
            data = DataConverter.load_data(args.input)
            
            splits = DataConverter.split_dataset(
                data, args.train, args.val, args.test, 
                args.stratify, args.seed
            )
            
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            input_path = Path(args.input)
            base_name = input_path.stem
            extension = input_path.suffix
            
            for split_name, split_data in splits.items():
                if len(split_data) > 0:
                    output_file = output_dir / f"{base_name}_{split_name}{extension}"
                    DataConverter.save_data(split_data, str(output_file))
                    print(f"{split_name}: {len(split_data)} rows -> {output_file}")

        elif args.command == 'info':
            print(f"Analyzing {args.input}")
            data = DataConverter.load_data(args.input)
            
            print(f"Dataset shape: {data.shape}")
            print(f"Columns: {list(data.columns)}")
            print(f"Data types:")
            for col, dtype in data.dtypes.items():
                print(f"  {col}: {dtype}")
            
            if args.detailed:
                print(f"\nDetailed statistics:")
                print(data.describe(include='all'))

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()