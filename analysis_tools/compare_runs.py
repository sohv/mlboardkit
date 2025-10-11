#!/usr/bin/env python3
"""
compare_runs.py

Compare metrics and results across multiple experiment runs.
"""

import argparse
import json
import glob
from pathlib import Path
from typing import Dict, List, Any
import statistics


class RunComparator:
    def __init__(self):
        pass
    
    def load_run_data(self, pattern: str) -> List[Dict[str, Any]]:
        """Load experiment run data from files matching pattern."""
        files = glob.glob(pattern)
        runs = []
        
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    data['source_file'] = Path(file_path).name
                    runs.append(data)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return runs
    
    def extract_metrics(self, runs: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """Extract numeric metrics from runs."""
        all_metrics = {}
        
        for run in runs:
            metrics = run.get('metrics', {})
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)
        
        return all_metrics
    
    def compute_statistics(self, metrics: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """Compute statistics for each metric."""
        stats = {}
        
        for metric_name, values in metrics.items():
            if len(values) > 0:
                stats[metric_name] = {
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
        
        return stats
    
    def find_best_runs(self, runs: List[Dict[str, Any]], metric: str, 
                      higher_better: bool = True) -> List[Dict[str, Any]]:
        """Find best performing runs based on a specific metric."""
        valid_runs = []
        
        for run in runs:
            metrics = run.get('metrics', {})
            if metric in metrics and isinstance(metrics[metric], (int, float)):
                valid_runs.append(run)
        
        if not valid_runs:
            return []
        
        # Sort by metric
        valid_runs.sort(
            key=lambda x: x['metrics'][metric], 
            reverse=higher_better
        )
        
        return valid_runs[:5]  # Top 5
    
    def generate_comparison_report(self, runs: List[Dict[str, Any]], 
                                 stats: Dict[str, Dict[str, float]]) -> str:
        """Generate comparison report."""
        report = f"# Experiment Runs Comparison\\n\\n"
        report += f"**Total Runs**: {len(runs)}\\n\\n"
        
        # Statistics table
        report += "## Metrics Statistics\\n\\n"
        report += "| Metric | Mean | Median | Std Dev | Min | Max | Count |\\n"
        report += "|--------|------|--------|---------|-----|-----|-------|\\n"
        
        for metric, stat in stats.items():
            report += f"| {metric} | {stat['mean']:.4f} | {stat['median']:.4f} | "
            report += f"{stat['std']:.4f} | {stat['min']:.4f} | {stat['max']:.4f} | "
            report += f"{stat['count']} |\\n"
        
        # Best runs section
        if stats:
            main_metric = list(stats.keys())[0]  # Use first metric
            best_runs = self.find_best_runs(runs, main_metric, higher_better=True)
            
            report += f"\\n## Top Runs (by {main_metric})\\n\\n"
            for i, run in enumerate(best_runs[:3]):
                report += f"### Run {i+1}: {run.get('source_file', 'Unknown')}\\n"
                metrics = run.get('metrics', {})
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        report += f"- **{k}**: {v:.4f}\\n"
                report += "\\n"
        
        return report


def main():
    parser = argparse.ArgumentParser(description='Compare experiment runs')
    parser.add_argument('--pattern', required=True, 
                       help='File pattern for run results (e.g., "results/*.json")')
    parser.add_argument('--output', default='run_comparison.md', help='Output report file')
    parser.add_argument('--metric', help='Primary metric for ranking runs')
    
    args = parser.parse_args()
    
    comparator = RunComparator()
    
    # Load run data
    runs = comparator.load_run_data(args.pattern)
    print(f"Loaded {len(runs)} experiment runs")
    
    if not runs:
        print("No valid runs found!")
        return
    
    # Extract and analyze metrics
    metrics = comparator.extract_metrics(runs)
    stats = comparator.compute_statistics(metrics)
    
    # Generate report
    report = comparator.generate_comparison_report(runs, stats)
    
    # Save report
    with open(args.output, 'w') as f:
        f.write(report)
    
    print(f"Comparison report saved to: {args.output}")
    print(f"Analyzed metrics: {list(metrics.keys())}")


if __name__ == '__main__':
    main()