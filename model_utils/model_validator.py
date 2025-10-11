#!/usr/bin/env python3
"""
model_validator.py

Validate machine learning models with comprehensive testing and quality checks.
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelValidator:
    def __init__(self):
        self.validation_results = {}
        self.thresholds = {
            'accuracy_threshold': 0.7,
            'precision_threshold': 0.7,
            'recall_threshold': 0.7,
            'f1_threshold': 0.7,
            'auc_threshold': 0.7,
            'bias_threshold': 0.1,
            'drift_threshold': 0.1
        }
    
    def set_thresholds(self, thresholds: Dict[str, float]):
        """Set validation thresholds"""
        self.thresholds.update(thresholds)
    
    def validate_model_performance(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 y_pred_proba: np.ndarray = None) -> Dict[str, Any]:
        """Validate model performance metrics"""
        
        try:
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, f1_score,
                confusion_matrix, classification_report, roc_auc_score,
                mean_squared_error, mean_absolute_error, r2_score
            )
            
            # Determine if classification or regression
            if len(np.unique(y_true)) <= 10 and np.all(np.unique(y_true) == np.unique(y_true).astype(int)):
                task_type = 'classification'
            else:
                task_type = 'regression'
            
            performance = {
                'task_type': task_type,
                'sample_size': len(y_true)
            }
            
            if task_type == 'classification':
                # Classification metrics
                performance.update({
                    'accuracy': float(accuracy_score(y_true, y_pred)),
                    'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
                    'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
                    'f1_score': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
                })
                
                # AUC if probabilities available
                if y_pred_proba is not None:
                    try:
                        if len(np.unique(y_true)) == 2:  # Binary classification
                            performance['auc'] = float(roc_auc_score(y_true, y_pred_proba))
                        else:  # Multi-class
                            performance['auc'] = float(roc_auc_score(y_true, y_pred_proba, multi_class='ovr'))
                    except Exception as e:
                        logger.warning(f"Could not calculate AUC: {e}")
                        performance['auc'] = None
                
                # Confusion matrix
                cm = confusion_matrix(y_true, y_pred)
                performance['confusion_matrix'] = cm.tolist()
                
                # Class-wise metrics
                try:
                    class_report = classification_report(y_true, y_pred, output_dict=True)
                    performance['class_metrics'] = class_report
                except Exception as e:
                    logger.warning(f"Could not generate classification report: {e}")
            
            else:
                # Regression metrics
                performance.update({
                    'mse': float(mean_squared_error(y_true, y_pred)),
                    'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                    'mae': float(mean_absolute_error(y_true, y_pred)),
                    'r2_score': float(r2_score(y_true, y_pred))
                })
                
                # Additional regression metrics
                residuals = y_true - y_pred
                performance.update({
                    'mean_residual': float(np.mean(residuals)),
                    'std_residual': float(np.std(residuals)),
                    'max_residual': float(np.max(np.abs(residuals)))
                })
            
            # Check if performance meets thresholds
            performance['passes_thresholds'] = self.check_performance_thresholds(performance)
            
            return performance
        
        except ImportError:
            logger.error("scikit-learn not available for performance validation")
            return {'error': 'scikit-learn not available'}
        except Exception as e:
            logger.error(f"Error in performance validation: {e}")
            return {'error': str(e)}
    
    def check_performance_thresholds(self, performance: Dict[str, Any]) -> Dict[str, bool]:
        """Check if performance metrics meet thresholds"""
        
        passes = {}
        
        if performance['task_type'] == 'classification':
            passes['accuracy'] = performance.get('accuracy', 0) >= self.thresholds['accuracy_threshold']
            passes['precision'] = performance.get('precision', 0) >= self.thresholds['precision_threshold']
            passes['recall'] = performance.get('recall', 0) >= self.thresholds['recall_threshold']
            passes['f1_score'] = performance.get('f1_score', 0) >= self.thresholds['f1_threshold']
            
            if 'auc' in performance and performance['auc'] is not None:
                passes['auc'] = performance['auc'] >= self.thresholds['auc_threshold']
        
        else:  # regression
            # For regression, we use R² score threshold
            passes['r2_score'] = performance.get('r2_score', -float('inf')) >= self.thresholds.get('r2_threshold', 0.5)
        
        passes['overall'] = all(passes.values())
        return passes
    
    def validate_data_quality(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Validate data quality"""
        
        data_quality = {
            'sample_size': len(X),
            'feature_count': X.shape[1] if len(X.shape) > 1 else 1,
            'missing_values': {}
        }
        
        # Check for missing values
        if hasattr(X, 'isnull'):  # pandas DataFrame
            missing_X = X.isnull().sum().sum()
            missing_y = y.isnull().sum() if hasattr(y, 'isnull') else 0
        else:  # numpy array
            missing_X = np.isnan(X).sum() if X.dtype.kind in 'fc' else 0
            missing_y = np.isnan(y).sum() if y.dtype.kind in 'fc' else 0
        
        data_quality['missing_values'] = {
            'features': int(missing_X),
            'target': int(missing_y),
            'total': int(missing_X + missing_y)
        }
        
        # Check for infinite values
        if np.issubdtype(X.dtype, np.number):
            inf_X = np.isinf(X).sum()
            inf_y = np.isinf(y).sum() if np.issubdtype(y.dtype, np.number) else 0
            
            data_quality['infinite_values'] = {
                'features': int(inf_X),
                'target': int(inf_y),
                'total': int(inf_X + inf_y)
            }
        
        # Check class balance (for classification)
        if len(np.unique(y)) <= 10:  # Likely classification
            unique_values, counts = np.unique(y, return_counts=True)
            class_distribution = dict(zip(unique_values.astype(str), counts.astype(int)))
            
            # Calculate imbalance ratio
            max_count = max(counts)
            min_count = min(counts)
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            data_quality['class_distribution'] = class_distribution
            data_quality['imbalance_ratio'] = float(imbalance_ratio)
            data_quality['is_balanced'] = imbalance_ratio <= 3.0  # Threshold for balance
        
        # Feature correlation check (for numerical features)
        if X.shape[1] > 1 and np.issubdtype(X.dtype, np.number):
            try:
                if hasattr(X, 'corr'):  # pandas DataFrame
                    corr_matrix = X.corr()
                else:
                    corr_matrix = np.corrcoef(X.T)
                
                # Find highly correlated feature pairs
                high_corr_pairs = []
                n_features = corr_matrix.shape[0]
                
                for i in range(n_features):
                    for j in range(i + 1, n_features):
                        corr_val = corr_matrix.iloc[i, j] if hasattr(corr_matrix, 'iloc') else corr_matrix[i, j]
                        if abs(corr_val) > 0.9:
                            high_corr_pairs.append({
                                'feature1': i,
                                'feature2': j,
                                'correlation': float(corr_val)
                            })
                
                data_quality['high_correlation_pairs'] = high_corr_pairs
                data_quality['multicollinearity_issues'] = len(high_corr_pairs) > 0
            
            except Exception as e:
                logger.warning(f"Could not calculate feature correlations: {e}")
        
        # Data quality score
        quality_score = 100.0
        
        if data_quality['missing_values']['total'] > 0:
            missing_ratio = data_quality['missing_values']['total'] / (len(X) * X.shape[1] + len(y))
            quality_score -= missing_ratio * 50
        
        if 'infinite_values' in data_quality and data_quality['infinite_values']['total'] > 0:
            quality_score -= 20
        
        if 'is_balanced' in data_quality and not data_quality['is_balanced']:
            quality_score -= 15
        
        if 'multicollinearity_issues' in data_quality and data_quality['multicollinearity_issues']:
            quality_score -= 10
        
        data_quality['quality_score'] = max(0, quality_score)
        
        return data_quality
    
    def validate_model_bias(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           sensitive_features: np.ndarray = None) -> Dict[str, Any]:
        """Validate model for bias and fairness"""
        
        bias_validation = {
            'bias_detected': False,
            'bias_metrics': {}
        }
        
        if sensitive_features is None:
            bias_validation['message'] = 'No sensitive features provided for bias validation'
            return bias_validation
        
        try:
            # Calculate metrics for each group
            unique_groups = np.unique(sensitive_features)
            group_metrics = {}
            
            for group in unique_groups:
                group_mask = sensitive_features == group
                group_y_true = y_true[group_mask]
                group_y_pred = y_pred[group_mask]
                
                if len(group_y_true) == 0:
                    continue
                
                # Calculate performance for this group
                group_perf = self.validate_model_performance(group_y_true, group_y_pred)
                group_metrics[str(group)] = {
                    'sample_size': len(group_y_true),
                    'metrics': group_perf
                }
            
            bias_validation['group_metrics'] = group_metrics
            
            # Calculate bias metrics
            if len(group_metrics) >= 2:
                # Get primary metric based on task type
                primary_metric = 'accuracy' if group_metrics[list(group_metrics.keys())[0]]['metrics']['task_type'] == 'classification' else 'r2_score'
                
                metric_values = []
                for group_data in group_metrics.values():
                    if primary_metric in group_data['metrics']:
                        metric_values.append(group_data['metrics'][primary_metric])
                
                if len(metric_values) >= 2:
                    max_metric = max(metric_values)
                    min_metric = min(metric_values)
                    bias_ratio = (max_metric - min_metric) / max_metric if max_metric > 0 else 0
                    
                    bias_validation['bias_metrics'] = {
                        'primary_metric': primary_metric,
                        'max_group_performance': float(max_metric),
                        'min_group_performance': float(min_metric),
                        'bias_ratio': float(bias_ratio),
                        'bias_threshold': self.thresholds['bias_threshold']
                    }
                    
                    bias_validation['bias_detected'] = bias_ratio > self.thresholds['bias_threshold']
        
        except Exception as e:
            logger.error(f"Error in bias validation: {e}")
            bias_validation['error'] = str(e)
        
        return bias_validation
    
    def validate_model_drift(self, X_train: np.ndarray, X_test: np.ndarray,
                           y_train: np.ndarray = None, y_test: np.ndarray = None) -> Dict[str, Any]:
        """Validate model for data drift"""
        
        drift_validation = {
            'data_drift_detected': False,
            'concept_drift_detected': False,
            'drift_metrics': {}
        }
        
        try:
            # Feature drift detection using statistical tests
            feature_drift_scores = []
            
            for i in range(X_train.shape[1]):
                train_feature = X_train[:, i] if len(X_train.shape) > 1 else X_train
                test_feature = X_test[:, i] if len(X_test.shape) > 1 else X_test
                
                # Use Kolmogorov-Smirnov test for continuous features
                try:
                    from scipy.stats import ks_2samp
                    statistic, p_value = ks_2samp(train_feature, test_feature)
                    
                    feature_drift_scores.append({
                        'feature_index': i,
                        'ks_statistic': float(statistic),
                        'p_value': float(p_value),
                        'drift_detected': p_value < 0.05
                    })
                
                except ImportError:
                    # Fallback: simple distribution comparison
                    train_mean = np.mean(train_feature)
                    test_mean = np.mean(test_feature)
                    train_std = np.std(train_feature)
                    test_std = np.std(test_feature)
                    
                    mean_diff = abs(train_mean - test_mean) / (train_std + 1e-8)
                    std_diff = abs(train_std - test_std) / (train_std + 1e-8)
                    
                    drift_score = max(mean_diff, std_diff)
                    
                    feature_drift_scores.append({
                        'feature_index': i,
                        'drift_score': float(drift_score),
                        'drift_detected': drift_score > 0.5
                    })
            
            drift_validation['feature_drift'] = feature_drift_scores
            
            # Overall data drift
            drift_detected_features = sum(1 for score in feature_drift_scores if score['drift_detected'])
            drift_ratio = drift_detected_features / len(feature_drift_scores)
            
            drift_validation['data_drift_detected'] = drift_ratio > self.thresholds['drift_threshold']
            drift_validation['drift_metrics']['feature_drift_ratio'] = float(drift_ratio)
            
            # Concept drift (if target variables provided)
            if y_train is not None and y_test is not None:
                if len(np.unique(y_train)) <= 10:  # Classification
                    # Compare class distributions
                    train_dist = np.bincount(y_train.astype(int)) / len(y_train)
                    test_dist = np.bincount(y_test.astype(int), minlength=len(train_dist)) / len(y_test)
                    
                    # KL divergence as concept drift measure
                    kl_div = np.sum(train_dist * np.log((train_dist + 1e-8) / (test_dist + 1e-8)))
                    
                    drift_validation['concept_drift_detected'] = kl_div > 0.1
                    drift_validation['drift_metrics']['kl_divergence'] = float(kl_div)
                
                else:  # Regression
                    # Compare target distributions
                    train_mean = np.mean(y_train)
                    test_mean = np.mean(y_test)
                    train_std = np.std(y_train)
                    
                    target_drift = abs(train_mean - test_mean) / (train_std + 1e-8)
                    
                    drift_validation['concept_drift_detected'] = target_drift > 0.5
                    drift_validation['drift_metrics']['target_drift_score'] = float(target_drift)
        
        except Exception as e:
            logger.error(f"Error in drift validation: {e}")
            drift_validation['error'] = str(e)
        
        return drift_validation
    
    def validate_model_robustness(self, model, X_test: np.ndarray, 
                                y_test: np.ndarray, noise_levels: List[float] = None) -> Dict[str, Any]:
        """Validate model robustness to input perturbations"""
        
        if noise_levels is None:
            noise_levels = [0.01, 0.05, 0.1, 0.2]
        
        robustness_validation = {
            'noise_tests': [],
            'robustness_score': 0.0
        }
        
        try:
            # Get baseline performance
            y_pred_baseline = model.predict(X_test)
            baseline_perf = self.validate_model_performance(y_test, y_pred_baseline)
            baseline_metric = baseline_perf['accuracy'] if baseline_perf['task_type'] == 'classification' else baseline_perf['r2_score']
            
            robustness_scores = []
            
            for noise_level in noise_levels:
                # Add noise to features
                noise = np.random.normal(0, noise_level, X_test.shape)
                X_noisy = X_test + noise
                
                # Get predictions on noisy data
                y_pred_noisy = model.predict(X_noisy)
                noisy_perf = self.validate_model_performance(y_test, y_pred_noisy)
                noisy_metric = noisy_perf['accuracy'] if noisy_perf['task_type'] == 'classification' else noisy_perf['r2_score']
                
                # Calculate performance degradation
                performance_degradation = (baseline_metric - noisy_metric) / baseline_metric if baseline_metric > 0 else 1.0
                robustness_score = max(0, 1 - performance_degradation)
                robustness_scores.append(robustness_score)
                
                robustness_validation['noise_tests'].append({
                    'noise_level': noise_level,
                    'baseline_metric': float(baseline_metric),
                    'noisy_metric': float(noisy_metric),
                    'performance_degradation': float(performance_degradation),
                    'robustness_score': float(robustness_score)
                })
            
            # Overall robustness score
            robustness_validation['robustness_score'] = float(np.mean(robustness_scores))
            robustness_validation['passes_robustness'] = robustness_validation['robustness_score'] > 0.7
        
        except Exception as e:
            logger.error(f"Error in robustness validation: {e}")
            robustness_validation['error'] = str(e)
        
        return robustness_validation
    
    def generate_validation_report(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        
        report = {
            'validation_summary': {
                'timestamp': datetime.now().isoformat(),
                'overall_passed': True,
                'critical_issues': [],
                'warnings': [],
                'recommendations': []
            },
            'detailed_results': validation_results
        }
        
        # Check each validation component
        if 'performance' in validation_results:
            perf = validation_results['performance']
            if 'passes_thresholds' in perf and not perf['passes_thresholds'].get('overall', False):
                report['validation_summary']['overall_passed'] = False
                report['validation_summary']['critical_issues'].append('Model performance below thresholds')
        
        if 'data_quality' in validation_results:
            quality = validation_results['data_quality']
            if quality.get('quality_score', 100) < 70:
                report['validation_summary']['warnings'].append('Data quality score below 70%')
            
            if quality.get('missing_values', {}).get('total', 0) > 0:
                report['validation_summary']['warnings'].append('Missing values detected in data')
        
        if 'bias' in validation_results:
            bias = validation_results['bias']
            if bias.get('bias_detected', False):
                report['validation_summary']['critical_issues'].append('Model bias detected across sensitive groups')
        
        if 'drift' in validation_results:
            drift = validation_results['drift']
            if drift.get('data_drift_detected', False):
                report['validation_summary']['warnings'].append('Data drift detected')
            if drift.get('concept_drift_detected', False):
                report['validation_summary']['critical_issues'].append('Concept drift detected')
        
        if 'robustness' in validation_results:
            robustness = validation_results['robustness']
            if not robustness.get('passes_robustness', True):
                report['validation_summary']['warnings'].append('Model robustness below threshold')
        
        # Generate recommendations
        if report['validation_summary']['critical_issues']:
            report['validation_summary']['recommendations'].append('Address critical issues before deployment')
        
        if report['validation_summary']['warnings']:
            report['validation_summary']['recommendations'].append('Consider addressing warnings for improved model quality')
        
        if not report['validation_summary']['critical_issues'] and not report['validation_summary']['warnings']:
            report['validation_summary']['recommendations'].append('Model passed all validation checks')
        
        return report

def main():
    parser = argparse.ArgumentParser(description="Comprehensive model validation")
    parser.add_argument('--predictions', required=True, help='File with predictions (JSON/CSV)')
    parser.add_argument('--ground-truth', required=True, help='File with ground truth labels')
    parser.add_argument('--features', help='File with feature data for drift detection')
    parser.add_argument('--train-features', help='Training features for drift comparison')
    parser.add_argument('--sensitive-features', help='Sensitive features for bias detection')
    parser.add_argument('--probabilities', help='Prediction probabilities (for classification)')
    parser.add_argument('--output', help='Output file for validation report')
    parser.add_argument('--thresholds', help='JSON file with custom validation thresholds')
    parser.add_argument('--validate', nargs='+', 
                       choices=['performance', 'quality', 'bias', 'drift', 'robustness'],
                       default=['performance', 'quality'],
                       help='Validation components to run')
    
    args = parser.parse_args()
    
    validator = ModelValidator()
    
    # Load custom thresholds if provided
    if args.thresholds:
        with open(args.thresholds, 'r') as f:
            custom_thresholds = json.load(f)
        validator.set_thresholds(custom_thresholds)
        print(f"Loaded custom thresholds from {args.thresholds}")
    
    # Load data
    print("Loading validation data...")
    
    # Load predictions and ground truth
    if args.predictions.endswith('.json'):
        with open(args.predictions, 'r') as f:
            predictions = np.array(json.load(f))
    else:
        predictions = pd.read_csv(args.predictions).values.flatten()
    
    if args.ground_truth.endswith('.json'):
        with open(args.ground_truth, 'r') as f:
            ground_truth = np.array(json.load(f))
    else:
        ground_truth = pd.read_csv(args.ground_truth).values.flatten()
    
    print(f"Loaded {len(predictions)} predictions and {len(ground_truth)} ground truth labels")
    
    # Load optional data
    probabilities = None
    if args.probabilities:
        if args.probabilities.endswith('.json'):
            with open(args.probabilities, 'r') as f:
                probabilities = np.array(json.load(f))
        else:
            probabilities = pd.read_csv(args.probabilities).values
    
    features = None
    if args.features:
        if args.features.endswith('.json'):
            with open(args.features, 'r') as f:
                features = np.array(json.load(f))
        else:
            features = pd.read_csv(args.features).values
    
    sensitive_features = None
    if args.sensitive_features:
        if args.sensitive_features.endswith('.json'):
            with open(args.sensitive_features, 'r') as f:
                sensitive_features = np.array(json.load(f))
        else:
            sensitive_features = pd.read_csv(args.sensitive_features).values.flatten()
    
    train_features = None
    if args.train_features:
        if args.train_features.endswith('.json'):
            with open(args.train_features, 'r') as f:
                train_features = np.array(json.load(f))
        else:
            train_features = pd.read_csv(args.train_features).values
    
    # Run validations
    validation_results = {}
    
    if 'performance' in args.validate:
        print("Validating model performance...")
        validation_results['performance'] = validator.validate_model_performance(
            ground_truth, predictions, probabilities
        )
    
    if 'quality' in args.validate and features is not None:
        print("Validating data quality...")
        validation_results['data_quality'] = validator.validate_data_quality(
            features, ground_truth
        )
    
    if 'bias' in args.validate and sensitive_features is not None:
        print("Validating model bias...")
        validation_results['bias'] = validator.validate_model_bias(
            ground_truth, predictions, sensitive_features
        )
    
    if 'drift' in args.validate and features is not None and train_features is not None:
        print("Validating model drift...")
        validation_results['drift'] = validator.validate_model_drift(
            train_features, features, None, ground_truth
        )
    
    # Generate report
    print("Generating validation report...")
    report = validator.generate_validation_report(validation_results)
    
    # Display results
    print(f"\nValidation Results Summary:")
    print(f"Overall Passed: {report['validation_summary']['overall_passed']}")
    
    if report['validation_summary']['critical_issues']:
        print(f"Critical Issues:")
        for issue in report['validation_summary']['critical_issues']:
            print(f"  ❌ {issue}")
    
    if report['validation_summary']['warnings']:
        print(f"Warnings:")
        for warning in report['validation_summary']['warnings']:
            print(f"  ⚠️  {warning}")
    
    if report['validation_summary']['recommendations']:
        print(f"Recommendations:")
        for rec in report['validation_summary']['recommendations']:
            print(f"  💡 {rec}")
    
    # Show key metrics
    if 'performance' in validation_results:
        perf = validation_results['performance']
        print(f"\nKey Performance Metrics:")
        if perf['task_type'] == 'classification':
            print(f"  Accuracy: {perf.get('accuracy', 0):.3f}")
            print(f"  F1 Score: {perf.get('f1_score', 0):.3f}")
            if 'auc' in perf and perf['auc'] is not None:
                print(f"  AUC: {perf['auc']:.3f}")
        else:
            print(f"  R² Score: {perf.get('r2_score', 0):.3f}")
            print(f"  RMSE: {perf.get('rmse', 0):.3f}")
    
    # Save report
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nValidation report saved to {args.output}")
    
    print(f"\nValidation completed!")

if __name__ == "__main__":
    main()