#!/usr/bin/env python3
"""
data_sampler.py

Smart data sampling with stratification, balancing, and statistical preservation.
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from collections import Counter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataSampler:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def random_sample(self, data: Union[pd.DataFrame, np.ndarray], 
                     sample_size: Union[int, float], 
                     replace: bool = False) -> Union[pd.DataFrame, np.ndarray]:
        """Random sampling with or without replacement"""
        
        n_samples = len(data)
        
        if isinstance(sample_size, float):
            if not 0 < sample_size <= 1:
                raise ValueError("Sample size as fraction must be between 0 and 1")
            sample_size = int(n_samples * sample_size)
        
        if sample_size > n_samples and not replace:
            logger.warning(f"Sample size ({sample_size}) larger than data size ({n_samples}). Using all data.")
            return data
        
        # Generate random indices
        if replace:
            indices = np.random.choice(n_samples, size=sample_size, replace=True)
        else:
            indices = np.random.choice(n_samples, size=sample_size, replace=False)
        
        # Return sampled data
        if isinstance(data, pd.DataFrame):
            return data.iloc[indices].reset_index(drop=True)
        else:
            return data[indices]
    
    def stratified_sample(self, data: pd.DataFrame, target_column: str, 
                         sample_size: Union[int, float],
                         min_samples_per_class: int = 1) -> pd.DataFrame:
        """Stratified sampling maintaining class proportions"""
        
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        n_samples = len(data)
        
        if isinstance(sample_size, float):
            if not 0 < sample_size <= 1:
                raise ValueError("Sample size as fraction must be between 0 and 1")
            sample_size = int(n_samples * sample_size)
        
        # Get class distribution
        class_counts = data[target_column].value_counts()
        classes = class_counts.index.tolist()
        
        # Calculate samples per class maintaining proportions
        sampled_indices = []
        
        for class_label in classes:
            class_data = data[data[target_column] == class_label]
            class_size = len(class_data)
            
            # Calculate proportional sample size for this class
            class_proportion = class_size / n_samples
            class_sample_size = int(sample_size * class_proportion)
            
            # Ensure minimum samples per class
            class_sample_size = max(class_sample_size, min_samples_per_class)
            
            # Don't sample more than available
            class_sample_size = min(class_sample_size, class_size)
            
            # Sample from this class
            if class_sample_size > 0:
                class_indices = np.random.choice(
                    class_data.index, 
                    size=class_sample_size, 
                    replace=False
                )
                sampled_indices.extend(class_indices)
        
        # Return stratified sample
        sampled_data = data.loc[sampled_indices].reset_index(drop=True)
        
        logger.info(f"Stratified sample created: {len(sampled_data)} samples")
        self._log_class_distribution(sampled_data, target_column, "After stratified sampling")
        
        return sampled_data
    
    def balanced_sample(self, data: pd.DataFrame, target_column: str,
                       strategy: str = 'undersample',
                       target_size: Optional[int] = None) -> pd.DataFrame:
        """Create balanced dataset by under/oversampling"""
        
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        class_counts = data[target_column].value_counts()
        
        if strategy == 'undersample':
            # Undersample to minority class size
            target_count = target_size or class_counts.min()
            
            sampled_data = []
            for class_label in class_counts.index:
                class_data = data[data[target_column] == class_label]
                
                if len(class_data) >= target_count:
                    # Undersample
                    sampled_class = class_data.sample(n=target_count, random_state=self.random_state)
                else:
                    # Use all available samples
                    sampled_class = class_data
                
                sampled_data.append(sampled_class)
            
            balanced_data = pd.concat(sampled_data, ignore_index=True)
        
        elif strategy == 'oversample':
            # Oversample to majority class size
            target_count = target_size or class_counts.max()
            
            sampled_data = []
            for class_label in class_counts.index:
                class_data = data[data[target_column] == class_label]
                
                if len(class_data) < target_count:
                    # Oversample with replacement
                    sampled_class = class_data.sample(
                        n=target_count, 
                        replace=True, 
                        random_state=self.random_state
                    )
                else:
                    # Use original data
                    sampled_class = class_data
                
                sampled_data.append(sampled_class)
            
            balanced_data = pd.concat(sampled_data, ignore_index=True)
        
        elif strategy == 'smote':
            # SMOTE oversampling (requires imblearn)
            try:
                from imblearn.over_sampling import SMOTE
                
                # Separate features and target
                X = data.drop(columns=[target_column])
                y = data[target_column]
                
                # Apply SMOTE
                smote = SMOTE(random_state=self.random_state)
                X_resampled, y_resampled = smote.fit_resample(X, y)
                
                # Combine back to DataFrame
                balanced_data = pd.DataFrame(X_resampled, columns=X.columns)
                balanced_data[target_column] = y_resampled
            
            except ImportError:
                logger.error("imblearn not available for SMOTE. Using oversample strategy instead.")
                return self.balanced_sample(data, target_column, 'oversample', target_size)
        
        else:
            raise ValueError(f"Unknown balancing strategy: {strategy}")
        
        # Shuffle the balanced dataset
        balanced_data = balanced_data.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        logger.info(f"Balanced sample created using {strategy}: {len(balanced_data)} samples")
        self._log_class_distribution(balanced_data, target_column, f"After {strategy}")
        
        return balanced_data
    
    def systematic_sample(self, data: Union[pd.DataFrame, np.ndarray], 
                         interval: int) -> Union[pd.DataFrame, np.ndarray]:
        """Systematic sampling - every nth element"""
        
        n_samples = len(data)
        
        if interval <= 0:
            raise ValueError("Interval must be positive")
        
        if interval >= n_samples:
            logger.warning(f"Interval ({interval}) >= data size ({n_samples}). Returning first element.")
            interval = n_samples
        
        # Generate systematic indices
        start = np.random.randint(0, interval)
        indices = np.arange(start, n_samples, interval)
        
        # Return sampled data
        if isinstance(data, pd.DataFrame):
            return data.iloc[indices].reset_index(drop=True)
        else:
            return data[indices]
    
    def cluster_sample(self, data: pd.DataFrame, cluster_column: str,
                      n_clusters: int, samples_per_cluster: Optional[int] = None) -> pd.DataFrame:
        """Cluster sampling - sample from selected clusters"""
        
        if cluster_column not in data.columns:
            raise ValueError(f"Cluster column '{cluster_column}' not found in data")
        
        # Get unique clusters
        unique_clusters = data[cluster_column].unique()
        
        if n_clusters >= len(unique_clusters):
            logger.warning(f"Requested clusters ({n_clusters}) >= available clusters ({len(unique_clusters)})")
            selected_clusters = unique_clusters
        else:
            # Randomly select clusters
            selected_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=False)
        
        # Sample from selected clusters
        cluster_samples = []
        
        for cluster in selected_clusters:
            cluster_data = data[data[cluster_column] == cluster]
            
            if samples_per_cluster is None:
                # Use all data from selected clusters
                cluster_samples.append(cluster_data)
            else:
                # Sample specific number from each cluster
                sample_size = min(samples_per_cluster, len(cluster_data))
                sampled_cluster = cluster_data.sample(n=sample_size, random_state=self.random_state)
                cluster_samples.append(sampled_cluster)
        
        sampled_data = pd.concat(cluster_samples, ignore_index=True)
        
        logger.info(f"Cluster sample created: {len(sampled_data)} samples from {len(selected_clusters)} clusters")
        
        return sampled_data
    
    def reservoir_sample(self, data_stream, sample_size: int) -> List[Any]:
        """Reservoir sampling for streaming data"""
        
        reservoir = []
        
        for i, item in enumerate(data_stream):
            if i < sample_size:
                # Fill reservoir
                reservoir.append(item)
            else:
                # Replace elements with decreasing probability
                j = np.random.randint(0, i + 1)
                if j < sample_size:
                    reservoir[j] = item
        
        return reservoir
    
    def time_based_sample(self, data: pd.DataFrame, time_column: str,
                         sample_type: str = 'recent',
                         sample_size: Union[int, float] = 0.1,
                         time_period: Optional[str] = None) -> pd.DataFrame:
        """Time-based sampling strategies"""
        
        if time_column not in data.columns:
            raise ValueError(f"Time column '{time_column}' not found in data")
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(data[time_column]):
            data[time_column] = pd.to_datetime(data[time_column])
        
        # Sort by time
        data_sorted = data.sort_values(time_column)
        
        if sample_type == 'recent':
            # Sample most recent data
            if isinstance(sample_size, float):
                n_samples = int(len(data) * sample_size)
            else:
                n_samples = sample_size
            
            sampled_data = data_sorted.tail(n_samples)
        
        elif sample_type == 'oldest':
            # Sample oldest data
            if isinstance(sample_size, float):
                n_samples = int(len(data) * sample_size)
            else:
                n_samples = sample_size
            
            sampled_data = data_sorted.head(n_samples)
        
        elif sample_type == 'period':
            # Sample from specific time period
            if time_period is None:
                raise ValueError("time_period required for period sampling")
            
            # Parse time period (e.g., "2023-01-01:2023-12-31")
            if ':' in time_period:
                start_date, end_date = time_period.split(':')
                start_date = pd.to_datetime(start_date)
                end_date = pd.to_datetime(end_date)
                
                mask = (data_sorted[time_column] >= start_date) & (data_sorted[time_column] <= end_date)
                period_data = data_sorted[mask]
                
                if isinstance(sample_size, float):
                    n_samples = int(len(period_data) * sample_size)
                else:
                    n_samples = min(sample_size, len(period_data))
                
                sampled_data = period_data.sample(n=n_samples, random_state=self.random_state)
            else:
                raise ValueError("Invalid time_period format. Use 'start_date:end_date'")
        
        elif sample_type == 'uniform':
            # Uniform sampling across time range
            min_time = data_sorted[time_column].min()
            max_time = data_sorted[time_column].max()
            
            if isinstance(sample_size, float):
                n_samples = int(len(data) * sample_size)
            else:
                n_samples = sample_size
            
            # Generate uniform time points
            time_points = pd.date_range(start=min_time, end=max_time, periods=n_samples)
            
            # Find nearest data points
            sampled_indices = []
            for time_point in time_points:
                nearest_idx = (data_sorted[time_column] - time_point).abs().idxmin()
                sampled_indices.append(nearest_idx)
            
            sampled_data = data_sorted.loc[sampled_indices].drop_duplicates()
        
        else:
            raise ValueError(f"Unknown time sampling type: {sample_type}")
        
        logger.info(f"Time-based {sample_type} sample created: {len(sampled_data)} samples")
        
        return sampled_data.reset_index(drop=True)
    
    def statistical_sample(self, data: pd.DataFrame, preserve_stats: List[str] = None,
                          sample_size: Union[int, float] = 0.1,
                          tolerance: float = 0.05) -> pd.DataFrame:
        """Sample while preserving statistical properties"""
        
        if preserve_stats is None:
            preserve_stats = ['mean', 'std', 'correlation']
        
        # Get original statistics
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        original_stats = {}
        
        if 'mean' in preserve_stats:
            original_stats['mean'] = data[numeric_columns].mean()
        
        if 'std' in preserve_stats:
            original_stats['std'] = data[numeric_columns].std()
        
        if 'correlation' in preserve_stats and len(numeric_columns) > 1:
            original_stats['correlation'] = data[numeric_columns].corr()
        
        # Determine sample size
        if isinstance(sample_size, float):
            n_samples = int(len(data) * sample_size)
        else:
            n_samples = sample_size
        
        # Try multiple sampling attempts to preserve statistics
        best_sample = None
        best_score = float('inf')
        
        for attempt in range(10):  # Try up to 10 times
            # Random sample
            sample = data.sample(n=n_samples, random_state=self.random_state + attempt)
            
            # Calculate statistics difference
            score = 0
            
            if 'mean' in preserve_stats:
                sample_mean = sample[numeric_columns].mean()
                mean_diff = np.mean(np.abs(sample_mean - original_stats['mean']))
                score += mean_diff
            
            if 'std' in preserve_stats:
                sample_std = sample[numeric_columns].std()
                std_diff = np.mean(np.abs(sample_std - original_stats['std']))
                score += std_diff
            
            if 'correlation' in preserve_stats and len(numeric_columns) > 1:
                sample_corr = sample[numeric_columns].corr()
                corr_diff = np.mean(np.abs(sample_corr.values - original_stats['correlation'].values))
                score += corr_diff
            
            # Check if this is the best sample so far
            if score < best_score:
                best_score = score
                best_sample = sample
            
            # If score is within tolerance, we're done
            if score <= tolerance:
                break
        
        logger.info(f"Statistical sample created: {len(best_sample)} samples (score: {best_score:.4f})")
        
        return best_sample.reset_index(drop=True)
    
    def _log_class_distribution(self, data: pd.DataFrame, target_column: str, prefix: str = ""):
        """Log class distribution for debugging"""
        class_counts = data[target_column].value_counts().sort_index()
        logger.info(f"{prefix} class distribution:")
        for class_label, count in class_counts.items():
            percentage = (count / len(data)) * 100
            logger.info(f"  {class_label}: {count} ({percentage:.1f}%)")
    
    def create_train_test_split(self, data: pd.DataFrame, 
                               test_size: float = 0.2,
                               stratify_column: Optional[str] = None,
                               time_column: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create train/test split with various strategies"""
        
        if time_column is not None:
            # Time-based split (no shuffling)
            data_sorted = data.sort_values(time_column)
            split_idx = int(len(data_sorted) * (1 - test_size))
            
            train_data = data_sorted.iloc[:split_idx].reset_index(drop=True)
            test_data = data_sorted.iloc[split_idx:].reset_index(drop=True)
            
            logger.info(f"Time-based split: {len(train_data)} train, {len(test_data)} test")
        
        elif stratify_column is not None:
            # Stratified split
            train_data = self.stratified_sample(data, stratify_column, 1 - test_size)
            
            # Remaining data becomes test set
            test_indices = data.index.difference(train_data.index)
            test_data = data.loc[test_indices].reset_index(drop=True)
            
            logger.info(f"Stratified split: {len(train_data)} train, {len(test_data)} test")
            self._log_class_distribution(train_data, stratify_column, "Train set")
            self._log_class_distribution(test_data, stratify_column, "Test set")
        
        else:
            # Random split
            shuffled_data = data.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
            split_idx = int(len(shuffled_data) * (1 - test_size))
            
            train_data = shuffled_data.iloc[:split_idx]
            test_data = shuffled_data.iloc[split_idx:]
            
            logger.info(f"Random split: {len(train_data)} train, {len(test_data)} test")
        
        return train_data, test_data

def main():
    parser = argparse.ArgumentParser(description="Smart data sampling with various strategies")
    parser.add_argument('data_file', help='Input data file (CSV, JSON, Excel)')
    parser.add_argument('--method', choices=[
        'random', 'stratified', 'balanced', 'systematic', 'cluster',
        'time', 'statistical', 'train_test_split'
    ], default='random', help='Sampling method')
    parser.add_argument('--sample-size', type=float, default=0.1,
                       help='Sample size (fraction 0-1 or absolute number)')
    parser.add_argument('--target-column', help='Target column for stratified/balanced sampling')
    parser.add_argument('--time-column', help='Time column for time-based sampling')
    parser.add_argument('--cluster-column', help='Cluster column for cluster sampling')
    parser.add_argument('--balance-strategy', choices=['undersample', 'oversample', 'smote'],
                       default='undersample', help='Balancing strategy')
    parser.add_argument('--time-type', choices=['recent', 'oldest', 'period', 'uniform'],
                       default='recent', help='Time sampling type')
    parser.add_argument('--time-period', help='Time period for period sampling (start:end)')
    parser.add_argument('--interval', type=int, default=10, help='Interval for systematic sampling')
    parser.add_argument('--n-clusters', type=int, default=5, help='Number of clusters to sample')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test size for train/test split')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--train-output', help='Training set output (for train_test_split)')
    parser.add_argument('--test-output', help='Test set output (for train_test_split)')
    parser.add_argument('--random-state', type=int, default=42, help='Random state for reproducibility')
    parser.add_argument('--info', action='store_true', help='Show data info before sampling')
    
    args = parser.parse_args()
    
    # Initialize sampler
    sampler = DataSampler(random_state=args.random_state)
    
    # Load data
    print(f"Loading data from {args.data_file}...")
    
    if args.data_file.endswith('.csv'):
        data = pd.read_csv(args.data_file)
    elif args.data_file.endswith('.json'):
        data = pd.read_json(args.data_file)
    elif args.data_file.endswith(('.xlsx', '.xls')):
        data = pd.read_excel(args.data_file)
    else:
        raise ValueError(f"Unsupported file format: {args.data_file}")
    
    print(f"Loaded data: {data.shape[0]} rows, {data.shape[1]} columns")
    
    # Show data info if requested
    if args.info:
        print(f"\nData Info:")
        print(f"  Shape: {data.shape}")
        print(f"  Columns: {list(data.columns)}")
        print(f"  Data types: {data.dtypes.value_counts().to_dict()}")
        
        if args.target_column and args.target_column in data.columns:
            print(f"\nTarget column distribution:")
            target_dist = data[args.target_column].value_counts()
            for value, count in target_dist.items():
                percentage = (count / len(data)) * 100
                print(f"  {value}: {count} ({percentage:.1f}%)")
    
    # Apply sampling method
    print(f"\nApplying {args.method} sampling...")
    
    if args.method == 'random':
        sampled_data = sampler.random_sample(data, args.sample_size)
    
    elif args.method == 'stratified':
        if not args.target_column:
            raise ValueError("--target-column required for stratified sampling")
        sampled_data = sampler.stratified_sample(data, args.target_column, args.sample_size)
    
    elif args.method == 'balanced':
        if not args.target_column:
            raise ValueError("--target-column required for balanced sampling")
        sampled_data = sampler.balanced_sample(data, args.target_column, args.balance_strategy)
    
    elif args.method == 'systematic':
        sampled_data = sampler.systematic_sample(data, args.interval)
    
    elif args.method == 'cluster':
        if not args.cluster_column:
            raise ValueError("--cluster-column required for cluster sampling")
        sampled_data = sampler.cluster_sample(data, args.cluster_column, args.n_clusters)
    
    elif args.method == 'time':
        if not args.time_column:
            raise ValueError("--time-column required for time-based sampling")
        sampled_data = sampler.time_based_sample(
            data, args.time_column, args.time_type, args.sample_size, args.time_period
        )
    
    elif args.method == 'statistical':
        sampled_data = sampler.statistical_sample(data, sample_size=args.sample_size)
    
    elif args.method == 'train_test_split':
        train_data, test_data = sampler.create_train_test_split(
            data, 
            test_size=args.test_size,
            stratify_column=args.target_column,
            time_column=args.time_column
        )
        
        # Save train and test sets
        if args.train_output:
            train_data.to_csv(args.train_output, index=False)
            print(f"Training set saved to {args.train_output}")
        
        if args.test_output:
            test_data.to_csv(args.test_output, index=False)
            print(f"Test set saved to {args.test_output}")
        
        if not args.train_output and not args.test_output:
            print(f"Train set: {train_data.shape}")
            print(f"Test set: {test_data.shape}")
        
        return
    
    else:
        raise ValueError(f"Unknown sampling method: {args.method}")
    
    # Display sampling results
    print(f"\nSampling Results:")
    print(f"  Original size: {len(data)}")
    print(f"  Sampled size: {len(sampled_data)}")
    print(f"  Sampling ratio: {len(sampled_data) / len(data):.3f}")
    
    # Save sampled data
    if args.output:
        if args.output.endswith('.csv'):
            sampled_data.to_csv(args.output, index=False)
        elif args.output.endswith('.json'):
            sampled_data.to_json(args.output, orient='records', indent=2)
        elif args.output.endswith(('.xlsx', '.xls')):
            sampled_data.to_excel(args.output, index=False)
        else:
            sampled_data.to_csv(args.output, index=False)
        
        print(f"Sampled data saved to {args.output}")
    else:
        print(f"\nSample preview:")
        print(sampled_data.head())

if __name__ == "__main__":
    main()