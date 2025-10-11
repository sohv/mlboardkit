#!/usr/bin/env python3
"""
feature_importance.py

Analyze and rank feature importance in datasets.
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureImportanceAnalyzer:
    def __init__(self):
        self.methods = {
            'correlation': self.correlation_importance,
            'mutual_info': self.mutual_info_importance,
            'chi2': self.chi2_importance,
            'variance': self.variance_importance,
            'tree_based': self.tree_based_importance,
            'permutation': self.permutation_importance,
            'univariate': self.univariate_importance
        }
    
    def load_data(self, data_path: str, target_column: str = None) -> tuple:
        """Load data from file"""
        path = Path(data_path)
        
        if path.suffix.lower() == '.csv':
            df = pd.read_csv(path)
        elif path.suffix.lower() == '.json':
            df = pd.read_json(path)
        elif path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        # Separate features and target
        if target_column:
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            
            X = df.drop(columns=[target_column])
            y = df[target_column]
        else:
            X = df
            y = None
        
        return X, y
    
    def preprocess_data(self, X: pd.DataFrame, y: pd.Series = None) -> tuple:
        """Preprocess data for feature importance analysis"""
        
        # Handle missing values
        X_processed = X.copy()
        
        # Fill numeric columns with median
        numeric_columns = X_processed.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            X_processed[col] = X_processed[col].fillna(X_processed[col].median())
        
        # Fill categorical columns with mode
        categorical_columns = X_processed.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            mode_value = X_processed[col].mode()
            if len(mode_value) > 0:
                X_processed[col] = X_processed[col].fillna(mode_value[0])
            else:
                X_processed[col] = X_processed[col].fillna('unknown')
        
        # Encode categorical variables
        from sklearn.preprocessing import LabelEncoder
        
        label_encoders = {}
        for col in categorical_columns:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))
            label_encoders[col] = le
        
        # Process target variable if provided
        y_processed = None
        target_encoder = None
        if y is not None:
            y_processed = y.copy()
            if y_processed.dtype == 'object':
                target_encoder = LabelEncoder()
                y_processed = target_encoder.fit_transform(y_processed.astype(str))
        
        return X_processed, y_processed, label_encoders, target_encoder
    
    def correlation_importance(self, X: pd.DataFrame, y: pd.Series = None) -> Dict[str, float]:
        """Calculate feature importance using correlation"""
        
        if y is None:
            # Use correlation with first principal component
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            pc1 = pca.fit_transform(X).flatten()
            correlations = X.corrwith(pd.Series(pc1, index=X.index))
        else:
            # Use correlation with target variable
            data = X.copy()
            data['target'] = y
            correlations = data.corr()['target'].drop('target')
        
        # Return absolute correlations
        importance_scores = correlations.abs().to_dict()
        return importance_scores
    
    def mutual_info_importance(self, X: pd.DataFrame, y: pd.Series = None) -> Dict[str, float]:
        """Calculate feature importance using mutual information"""
        
        if y is None:
            logger.warning("Mutual information requires target variable, skipping")
            return {}
        
        try:
            from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
            
            # Determine if classification or regression
            if len(np.unique(y)) <= 10:  # Assume classification if few unique values
                scores = mutual_info_classif(X, y)
            else:
                scores = mutual_info_regression(X, y)
            
            importance_scores = dict(zip(X.columns, scores))
            return importance_scores
        
        except ImportError:
            logger.warning("scikit-learn not available for mutual information")
            return {}
    
    def chi2_importance(self, X: pd.DataFrame, y: pd.Series = None) -> Dict[str, float]:
        """Calculate feature importance using Chi-squared test"""
        
        if y is None:
            logger.warning("Chi-squared test requires target variable, skipping")
            return {}
        
        try:
            from sklearn.feature_selection import chi2
            
            # Ensure all values are non-negative for chi2
            X_positive = X - X.min() + 1
            
            scores, _ = chi2(X_positive, y)
            importance_scores = dict(zip(X.columns, scores))
            return importance_scores
        
        except ImportError:
            logger.warning("scikit-learn not available for Chi-squared test")
            return {}
        except Exception as e:
            logger.warning(f"Chi-squared test failed: {e}")
            return {}
    
    def variance_importance(self, X: pd.DataFrame, y: pd.Series = None) -> Dict[str, float]:
        """Calculate feature importance using variance"""
        
        variances = X.var()
        
        # Normalize by mean to get coefficient of variation
        means = X.mean()
        cv_scores = variances / (means + 1e-8)  # Add small value to avoid division by zero
        
        importance_scores = cv_scores.to_dict()
        return importance_scores
    
    def tree_based_importance(self, X: pd.DataFrame, y: pd.Series = None) -> Dict[str, float]:
        """Calculate feature importance using tree-based methods"""
        
        if y is None:
            logger.warning("Tree-based importance requires target variable, skipping")
            return {}
        
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            
            # Determine if classification or regression
            if len(np.unique(y)) <= 10:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            model.fit(X, y)
            importance_scores = dict(zip(X.columns, model.feature_importances_))
            return importance_scores
        
        except ImportError:
            logger.warning("scikit-learn not available for tree-based importance")
            return {}
    
    def permutation_importance(self, X: pd.DataFrame, y: pd.Series = None) -> Dict[str, float]:
        """Calculate feature importance using permutation"""
        
        if y is None:
            logger.warning("Permutation importance requires target variable, skipping")
            return {}
        
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.inspection import permutation_importance
            from sklearn.model_selection import train_test_split
            
            # Split data for unbiased evaluation
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            
            # Train model
            if len(np.unique(y)) <= 10:
                model = RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=42)
            
            model.fit(X_train, y_train)
            
            # Calculate permutation importance
            perm_importance = permutation_importance(
                model, X_test, y_test, n_repeats=10, random_state=42
            )
            
            importance_scores = dict(zip(X.columns, perm_importance.importances_mean))
            return importance_scores
        
        except ImportError:
            logger.warning("scikit-learn not available for permutation importance")
            return {}
    
    def univariate_importance(self, X: pd.DataFrame, y: pd.Series = None) -> Dict[str, float]:
        """Calculate feature importance using univariate statistical tests"""
        
        if y is None:
            logger.warning("Univariate tests require target variable, skipping")
            return {}
        
        try:
            from sklearn.feature_selection import f_classif, f_regression
            
            # Determine if classification or regression
            if len(np.unique(y)) <= 10:
                scores, _ = f_classif(X, y)
            else:
                scores, _ = f_regression(X, y)
            
            importance_scores = dict(zip(X.columns, scores))
            return importance_scores
        
        except ImportError:
            logger.warning("scikit-learn not available for univariate tests")
            return {}
    
    def analyze_feature_importance(self, X: pd.DataFrame, y: pd.Series = None, 
                                 methods: List[str] = None) -> Dict[str, Any]:
        """Analyze feature importance using multiple methods"""
        
        if methods is None:
            methods = ['correlation', 'variance']
            if y is not None:
                methods.extend(['mutual_info', 'tree_based', 'univariate'])
        
        results = {}
        
        for method in methods:
            if method in self.methods:
                logger.info(f"Calculating {method} importance...")
                try:
                    scores = self.methods[method](X, y)
                    if scores:  # Only add if we got results
                        results[method] = scores
                except Exception as e:
                    logger.error(f"Error calculating {method} importance: {e}")
            else:
                logger.warning(f"Unknown method: {method}")
        
        return results
    
    def rank_features(self, importance_results: Dict[str, Dict[str, float]], 
                     aggregation: str = 'mean') -> Dict[str, Any]:
        """Rank features based on importance scores"""
        
        if not importance_results:
            return {}
        
        # Get all feature names
        all_features = set()
        for scores in importance_results.values():
            all_features.update(scores.keys())
        
        # Calculate aggregated scores
        aggregated_scores = {}
        feature_rankings = {}
        
        for feature in all_features:
            scores = []
            rankings = []
            
            for method, method_scores in importance_results.items():
                if feature in method_scores:
                    scores.append(method_scores[feature])
                    
                    # Calculate rank for this method
                    sorted_features = sorted(method_scores.items(), 
                                           key=lambda x: x[1], reverse=True)
                    rank = next(i for i, (f, _) in enumerate(sorted_features) if f == feature) + 1
                    rankings.append(rank)
            
            if scores:
                if aggregation == 'mean':
                    aggregated_scores[feature] = np.mean(scores)
                elif aggregation == 'median':
                    aggregated_scores[feature] = np.median(scores)
                elif aggregation == 'max':
                    aggregated_scores[feature] = np.max(scores)
                elif aggregation == 'min':
                    aggregated_scores[feature] = np.min(scores)
                else:
                    aggregated_scores[feature] = np.mean(scores)
                
                feature_rankings[feature] = {
                    'scores': scores,
                    'rankings': rankings,
                    'mean_rank': np.mean(rankings),
                    'std_rank': np.std(rankings) if len(rankings) > 1 else 0
                }
        
        # Sort features by aggregated score
        sorted_features = sorted(aggregated_scores.items(), 
                               key=lambda x: x[1], reverse=True)
        
        return {
            'aggregated_scores': aggregated_scores,
            'feature_rankings': feature_rankings,
            'sorted_features': sorted_features,
            'top_features': [f for f, _ in sorted_features[:10]]
        }
    
    def generate_report(self, importance_results: Dict[str, Dict[str, float]], 
                       ranking_results: Dict[str, Any],
                       feature_names: List[str] = None) -> Dict[str, Any]:
        """Generate comprehensive feature importance report"""
        
        report = {
            'summary': {
                'total_features': len(feature_names) if feature_names else 0,
                'methods_used': list(importance_results.keys()),
                'top_features': ranking_results.get('top_features', [])
            },
            'importance_scores': importance_results,
            'rankings': ranking_results,
            'insights': []
        }
        
        # Generate insights
        if ranking_results.get('sorted_features'):
            top_feature, top_score = ranking_results['sorted_features'][0]
            report['insights'].append(f"Most important feature: {top_feature} (score: {top_score:.4f})")
            
            if len(ranking_results['sorted_features']) > 1:
                second_feature, second_score = ranking_results['sorted_features'][1]
                ratio = top_score / second_score if second_score > 0 else float('inf')
                report['insights'].append(f"Top feature is {ratio:.2f}x more important than second feature")
        
        # Consistency analysis
        if len(importance_results) > 1:
            feature_rankings = ranking_results.get('feature_rankings', {})
            consistent_features = []
            inconsistent_features = []
            
            for feature, rank_info in feature_rankings.items():
                std_rank = rank_info.get('std_rank', 0)
                if std_rank < 2:  # Low variance in rankings
                    consistent_features.append(feature)
                elif std_rank > 5:  # High variance in rankings
                    inconsistent_features.append(feature)
            
            if consistent_features:
                report['insights'].append(f"Features with consistent rankings: {len(consistent_features)}")
            if inconsistent_features:
                report['insights'].append(f"Features with inconsistent rankings: {len(inconsistent_features)}")
        
        return report

def create_sample_data(n_samples: int = 1000, n_features: int = 10, 
                      n_informative: int = 5) -> tuple:
    """Create sample data for testing"""
    try:
        from sklearn.datasets import make_classification, make_regression
        
        # Create classification data
        X_class, y_class = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_features - n_informative,
            random_state=42
        )
        
        # Convert to DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        X_df = pd.DataFrame(X_class, columns=feature_names)
        y_series = pd.Series(y_class, name='target')
        
        return X_df, y_series
    
    except ImportError:
        # Create simple synthetic data
        np.random.seed(42)
        
        # Create features with different importance levels
        X_data = {}
        
        # Important features (correlated with target)
        for i in range(n_informative):
            X_data[f'important_feature_{i}'] = np.random.randn(n_samples)
        
        # Less important features
        for i in range(n_features - n_informative):
            X_data[f'noise_feature_{i}'] = np.random.randn(n_samples)
        
        X_df = pd.DataFrame(X_data)
        
        # Create target as combination of important features
        y_data = sum(X_df[col] for col in X_df.columns if 'important' in col)
        y_data += np.random.randn(n_samples) * 0.1  # Add noise
        y_series = pd.Series((y_data > y_data.median()).astype(int), name='target')
        
        return X_df, y_series

def main():
    parser = argparse.ArgumentParser(description="Analyze feature importance in datasets")
    parser.add_argument('--data', help='Path to data file (CSV, JSON, Excel)')
    parser.add_argument('--target', help='Target column name')
    parser.add_argument('--methods', nargs='+', 
                       choices=['correlation', 'mutual_info', 'chi2', 'variance', 
                               'tree_based', 'permutation', 'univariate'],
                       help='Feature importance methods to use')
    parser.add_argument('--output', help='Output file for results')
    parser.add_argument('--aggregation', choices=['mean', 'median', 'max', 'min'],
                       default='mean', help='Score aggregation method')
    parser.add_argument('--top-k', type=int, default=10, help='Number of top features to show')
    parser.add_argument('--sample-data', action='store_true', help='Use sample data for testing')
    parser.add_argument('--n-samples', type=int, default=1000, help='Number of samples for sample data')
    parser.add_argument('--n-features', type=int, default=10, help='Number of features for sample data')
    
    args = parser.parse_args()
    
    analyzer = FeatureImportanceAnalyzer()
    
    # Load or create data
    if args.sample_data:
        print("Creating sample data...")
        X, y = create_sample_data(args.n_samples, args.n_features)
        print(f"Created data with {X.shape[0]} samples and {X.shape[1]} features")
    elif args.data:
        print(f"Loading data from {args.data}...")
        X, y = analyzer.load_data(args.data, args.target)
        print(f"Loaded data with {X.shape[0]} samples and {X.shape[1]} features")
        
        # Preprocess data
        X, y, label_encoders, target_encoder = analyzer.preprocess_data(X, y)
        print("Data preprocessed (missing values filled, categorical variables encoded)")
    else:
        print("Error: Either --data or --sample-data must be specified")
        return
    
    # Analyze feature importance
    print("\nAnalyzing feature importance...")
    importance_results = analyzer.analyze_feature_importance(X, y, args.methods)
    
    if not importance_results:
        print("No importance scores calculated")
        return
    
    # Rank features
    print("Ranking features...")
    ranking_results = analyzer.rank_features(importance_results, args.aggregation)
    
    # Generate report
    report = analyzer.generate_report(importance_results, ranking_results, X.columns.tolist())
    
    # Display results
    print(f"\nFeature Importance Analysis Results:")
    print(f"Total features: {report['summary']['total_features']}")
    print(f"Methods used: {', '.join(report['summary']['methods_used'])}")
    
    print(f"\nTop {args.top_k} Most Important Features:")
    print(f"{'Rank':<6} {'Feature':<25} {'Score':<10} {'Methods'}")
    print("-" * 60)
    
    for i, (feature, score) in enumerate(ranking_results['sorted_features'][:args.top_k]):
        methods_with_feature = [method for method, scores in importance_results.items() 
                               if feature in scores]
        methods_str = ', '.join(methods_with_feature)
        print(f"{i+1:<6} {feature:<25} {score:<10.4f} {methods_str}")
    
    # Show insights
    if report['insights']:
        print(f"\nKey Insights:")
        for insight in report['insights']:
            print(f"  • {insight}")
    
    # Show method-specific rankings
    print(f"\nMethod-Specific Rankings:")
    for method, scores in importance_results.items():
        print(f"\n{method.upper()}:")
        sorted_method_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, score) in enumerate(sorted_method_scores[:5]):
            print(f"  {i+1}. {feature}: {score:.4f}")
    
    # Save results
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()