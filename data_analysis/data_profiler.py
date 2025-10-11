#!/usr/bin/env python3
"""
data_profiler.py

Comprehensive data profiling and quality assessment.
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProfiler:
    def __init__(self):
        self.profile = {}
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load data from various file formats"""
        path = Path(data_path)
        
        if path.suffix.lower() == '.csv':
            df = pd.read_csv(path)
        elif path.suffix.lower() == '.json':
            df = pd.read_json(path)
        elif path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(path)
        elif path.suffix.lower() == '.parquet':
            df = pd.read_parquet(path)
        elif path.suffix.lower() == '.tsv':
            df = pd.read_csv(path, sep='\t')
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        return df
    
    def basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset information"""
        
        return {
            'shape': {
                'rows': len(df),
                'columns': len(df.columns)
            },
            'memory_usage': {
                'total_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'per_column': df.memory_usage(deep=True).to_dict()
            },
            'dtypes': df.dtypes.astype(str).to_dict(),
            'column_names': df.columns.tolist()
        }
    
    def missing_values_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing values"""
        
        missing_count = df.isnull().sum()
        missing_percent = (missing_count / len(df)) * 100
        
        missing_info = {}
        for col in df.columns:
            missing_info[col] = {
                'count': int(missing_count[col]),
                'percentage': float(missing_percent[col])
            }
        
        # Overall statistics
        total_missing = missing_count.sum()
        total_cells = df.shape[0] * df.shape[1]
        
        return {
            'by_column': missing_info,
            'summary': {
                'total_missing': int(total_missing),
                'total_cells': int(total_cells),
                'overall_percentage': float(total_missing / total_cells * 100),
                'columns_with_missing': int((missing_count > 0).sum()),
                'complete_rows': int(df.dropna().shape[0])
            }
        }
    
    def numeric_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze numeric columns"""
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        numeric_analysis = {}
        
        for col in numeric_columns:
            series = df[col].dropna()
            
            if len(series) == 0:
                continue
            
            # Basic statistics
            stats = {
                'count': int(len(series)),
                'mean': float(series.mean()),
                'std': float(series.std()),
                'min': float(series.min()),
                'max': float(series.max()),
                'median': float(series.median()),
                'q25': float(series.quantile(0.25)),
                'q75': float(series.quantile(0.75)),
                'skewness': float(series.skew()),
                'kurtosis': float(series.kurtosis())
            }
            
            # Outlier detection using IQR
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            
            stats['outliers'] = {
                'count': int(len(outliers)),
                'percentage': float(len(outliers) / len(series) * 100),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound)
            }
            
            # Zero values
            zero_count = (series == 0).sum()
            stats['zeros'] = {
                'count': int(zero_count),
                'percentage': float(zero_count / len(series) * 100)
            }
            
            # Distribution characteristics
            stats['distribution'] = {
                'is_normal': self.test_normality(series),
                'unique_values': int(series.nunique()),
                'unique_percentage': float(series.nunique() / len(series) * 100)
            }
            
            numeric_analysis[col] = stats
        
        return numeric_analysis
    
    def categorical_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze categorical columns"""
        
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        categorical_analysis = {}
        
        for col in categorical_columns:
            series = df[col].dropna()
            
            if len(series) == 0:
                continue
            
            # Value counts
            value_counts = series.value_counts()
            
            # Basic statistics
            stats = {
                'count': int(len(series)),
                'unique_values': int(series.nunique()),
                'unique_percentage': float(series.nunique() / len(series) * 100),
                'most_frequent': {
                    'value': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    'count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    'percentage': float(value_counts.iloc[0] / len(series) * 100) if len(value_counts) > 0 else 0
                }
            }
            
            # Top values
            top_values = []
            for i, (value, count) in enumerate(value_counts.head(10).items()):
                top_values.append({
                    'value': str(value),
                    'count': int(count),
                    'percentage': float(count / len(series) * 100)
                })
            
            stats['top_values'] = top_values
            
            # Length statistics for string columns
            if series.dtype == 'object':
                lengths = series.astype(str).str.len()
                stats['length_stats'] = {
                    'min_length': int(lengths.min()),
                    'max_length': int(lengths.max()),
                    'avg_length': float(lengths.mean()),
                    'std_length': float(lengths.std())
                }
            
            # Cardinality assessment
            cardinality = series.nunique()
            total_count = len(series)
            
            if cardinality == total_count:
                cardinality_type = 'unique_identifier'
            elif cardinality / total_count > 0.95:
                cardinality_type = 'high_cardinality'
            elif cardinality / total_count < 0.05:
                cardinality_type = 'low_cardinality'
            else:
                cardinality_type = 'medium_cardinality'
            
            stats['cardinality_type'] = cardinality_type
            
            categorical_analysis[col] = stats
        
        return categorical_analysis
    
    def test_normality(self, series: pd.Series) -> bool:
        """Test if a series follows normal distribution"""
        try:
            from scipy.stats import shapiro
            
            # Use Shapiro-Wilk test for small samples
            if len(series) <= 5000:
                _, p_value = shapiro(series.sample(min(len(series), 5000)))
                return p_value > 0.05
            else:
                # For larger samples, use simpler heuristics
                skew = abs(series.skew())
                kurtosis = abs(series.kurtosis())
                return skew < 1 and kurtosis < 3
        
        except:
            # Fallback: simple heuristics
            skew = abs(series.skew())
            kurtosis = abs(series.kurtosis())
            return skew < 1 and kurtosis < 3
    
    def correlation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between numeric columns"""
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            return {'message': 'Not enough numeric columns for correlation analysis'}
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Find highly correlated pairs
        high_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                
                if abs(corr_value) > 0.7:  # High correlation threshold
                    high_correlations.append({
                        'column1': col1,
                        'column2': col2,
                        'correlation': float(corr_value),
                        'strength': 'very_strong' if abs(corr_value) > 0.9 else 'strong'
                    })
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'high_correlations': high_correlations,
            'summary': {
                'total_pairs': int(len(corr_matrix.columns) * (len(corr_matrix.columns) - 1) / 2),
                'high_correlation_pairs': len(high_correlations)
            }
        }
    
    def data_quality_assessment(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess overall data quality"""
        
        quality_issues = []
        score = 100.0
        
        # Check missing values
        missing_analysis = self.missing_values_analysis(df)
        missing_percentage = missing_analysis['summary']['overall_percentage']
        
        if missing_percentage > 50:
            quality_issues.append('High percentage of missing values (>50%)')
            score -= 30
        elif missing_percentage > 20:
            quality_issues.append('Moderate percentage of missing values (>20%)')
            score -= 15
        elif missing_percentage > 5:
            quality_issues.append('Some missing values (>5%)')
            score -= 5
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        duplicate_percentage = duplicate_count / len(df) * 100
        
        if duplicate_percentage > 10:
            quality_issues.append('High percentage of duplicate rows (>10%)')
            score -= 20
        elif duplicate_percentage > 1:
            quality_issues.append('Some duplicate rows (>1%)')
            score -= 10
        
        # Check for constant columns
        constant_columns = []
        for col in df.columns:
            if df[col].nunique() <= 1:
                constant_columns.append(col)
        
        if constant_columns:
            quality_issues.append(f'Constant columns found: {", ".join(constant_columns)}')
            score -= len(constant_columns) * 5
        
        # Check for high cardinality categorical columns
        high_cardinality_cols = []
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_columns:
            cardinality_ratio = df[col].nunique() / len(df)
            if cardinality_ratio > 0.9:
                high_cardinality_cols.append(col)
        
        if high_cardinality_cols:
            quality_issues.append(f'High cardinality categorical columns: {", ".join(high_cardinality_cols)}')
            score -= len(high_cardinality_cols) * 5
        
        # Overall quality rating
        if score >= 90:
            quality_rating = 'excellent'
        elif score >= 75:
            quality_rating = 'good'
        elif score >= 60:
            quality_rating = 'fair'
        elif score >= 40:
            quality_rating = 'poor'
        else:
            quality_rating = 'very_poor'
        
        return {
            'quality_score': max(0, score),
            'quality_rating': quality_rating,
            'issues': quality_issues,
            'duplicate_rows': {
                'count': int(duplicate_count),
                'percentage': float(duplicate_percentage)
            },
            'constant_columns': constant_columns,
            'high_cardinality_columns': high_cardinality_cols
        }
    
    def generate_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data profile"""
        
        logger.info("Generating data profile...")
        
        profile = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'profiler_version': '1.0.0'
            },
            'basic_info': self.basic_info(df),
            'missing_values': self.missing_values_analysis(df),
            'numeric_analysis': self.numeric_analysis(df),
            'categorical_analysis': self.categorical_analysis(df),
            'correlation_analysis': self.correlation_analysis(df),
            'data_quality': self.data_quality_assessment(df)
        }
        
        return profile
    
    def generate_recommendations(self, profile: Dict[str, Any]) -> List[str]:
        """Generate data improvement recommendations"""
        
        recommendations = []
        
        # Missing values recommendations
        missing_summary = profile.get('missing_values', {}).get('summary', {})
        if missing_summary.get('overall_percentage', 0) > 10:
            recommendations.append(
                "Consider imputing missing values or removing columns/rows with high missing percentages"
            )
        
        # Data quality recommendations
        quality_issues = profile.get('data_quality', {}).get('issues', [])
        if 'High percentage of duplicate rows' in str(quality_issues):
            recommendations.append("Remove duplicate rows to improve data quality")
        
        constant_columns = profile.get('data_quality', {}).get('constant_columns', [])
        if constant_columns:
            recommendations.append(f"Consider removing constant columns: {', '.join(constant_columns)}")
        
        # Correlation recommendations
        high_correlations = profile.get('correlation_analysis', {}).get('high_correlations', [])
        if len(high_correlations) > 0:
            recommendations.append(
                "High correlations detected between numeric columns - consider feature selection or dimensionality reduction"
            )
        
        # Outlier recommendations
        numeric_analysis = profile.get('numeric_analysis', {})
        high_outlier_columns = []
        for col, stats in numeric_analysis.items():
            outlier_percentage = stats.get('outliers', {}).get('percentage', 0)
            if outlier_percentage > 10:
                high_outlier_columns.append(col)
        
        if high_outlier_columns:
            recommendations.append(
                f"Consider investigating outliers in columns: {', '.join(high_outlier_columns)}"
            )
        
        # Cardinality recommendations
        categorical_analysis = profile.get('categorical_analysis', {})
        high_cardinality_cols = [
            col for col, stats in categorical_analysis.items()
            if stats.get('cardinality_type') == 'high_cardinality'
        ]
        
        if high_cardinality_cols:
            recommendations.append(
                f"Consider encoding or grouping high cardinality categorical columns: {', '.join(high_cardinality_cols)}"
            )
        
        return recommendations

def main():
    parser = argparse.ArgumentParser(description="Comprehensive data profiling and quality assessment")
    parser.add_argument('data_file', help='Path to data file')
    parser.add_argument('--output', help='Output file for profile report')
    parser.add_argument('--format', choices=['json', 'html'], default='json',
                       help='Output format')
    parser.add_argument('--sample', type=int, help='Sample size for large datasets')
    parser.add_argument('--recommendations', action='store_true',
                       help='Include recommendations in the report')
    
    args = parser.parse_args()
    
    profiler = DataProfiler()
    
    # Load data
    print(f"Loading data from {args.data_file}...")
    df = profiler.load_data(args.data_file)
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Sample data if requested
    if args.sample and len(df) > args.sample:
        print(f"Sampling {args.sample} rows from dataset...")
        df = df.sample(n=args.sample, random_state=42)
    
    # Generate profile
    profile = profiler.generate_profile(df)
    
    # Add recommendations if requested
    if args.recommendations:
        profile['recommendations'] = profiler.generate_recommendations(profile)
    
    # Display summary
    print(f"\nData Profile Summary:")
    print(f"  Dataset shape: {profile['basic_info']['shape']['rows']} rows × {profile['basic_info']['shape']['columns']} columns")
    print(f"  Memory usage: {profile['basic_info']['memory_usage']['total_mb']:.2f} MB")
    print(f"  Missing values: {profile['missing_values']['summary']['overall_percentage']:.2f}%")
    print(f"  Data quality score: {profile['data_quality']['quality_score']:.1f}/100 ({profile['data_quality']['quality_rating']})")
    
    # Show column type breakdown
    numeric_cols = len(profile['numeric_analysis'])
    categorical_cols = len(profile['categorical_analysis'])
    print(f"  Column types: {numeric_cols} numeric, {categorical_cols} categorical")
    
    # Show top data quality issues
    quality_issues = profile['data_quality']['issues']
    if quality_issues:
        print(f"\nTop Data Quality Issues:")
        for issue in quality_issues[:3]:
            print(f"  • {issue}")
    
    # Show recommendations
    if args.recommendations and 'recommendations' in profile:
        print(f"\nRecommendations:")
        for rec in profile['recommendations']:
            print(f"  • {rec}")
    
    # Save profile
    if args.output:
        if args.format == 'json':
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(profile, f, indent=2, ensure_ascii=False)
            print(f"\nProfile saved to {args.output}")
        
        elif args.format == 'html':
            # Generate HTML report (simple version)
            html_content = generate_html_report(profile)
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"\nHTML report saved to {args.output}")

def generate_html_report(profile: Dict[str, Any]) -> str:
    """Generate simple HTML report"""
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Profile Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
            .metric {{ margin: 10px 0; }}
            .quality-score {{ font-size: 24px; font-weight: bold; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Data Profile Report</h1>
        <p>Generated: {profile['metadata']['generated_at']}</p>
        
        <div class="section">
            <h2>Dataset Overview</h2>
            <div class="metric">Shape: {profile['basic_info']['shape']['rows']} rows × {profile['basic_info']['shape']['columns']} columns</div>
            <div class="metric">Memory Usage: {profile['basic_info']['memory_usage']['total_mb']:.2f} MB</div>
            <div class="metric">Missing Values: {profile['missing_values']['summary']['overall_percentage']:.2f}%</div>
            <div class="metric">Data Quality Score: <span class="quality-score">{profile['data_quality']['quality_score']:.1f}/100</span> ({profile['data_quality']['quality_rating']})</div>
        </div>
        
        <div class="section">
            <h2>Column Analysis</h2>
            <p>Numeric columns: {len(profile['numeric_analysis'])}</p>
            <p>Categorical columns: {len(profile['categorical_analysis'])}</p>
        </div>
        
        <div class="section">
            <h2>Data Quality Issues</h2>
            <ul>
    """
    
    for issue in profile['data_quality']['issues']:
        html += f"<li>{issue}</li>"
    
    html += """
            </ul>
        </div>
        
        <div class="section">
            <h2>Recommendations</h2>
            <ul>
    """
    
    if 'recommendations' in profile:
        for rec in profile['recommendations']:
            html += f"<li>{rec}</li>"
    
    html += """
            </ul>
        </div>
    </body>
    </html>
    """
    
    return html

if __name__ == "__main__":
    main()