#!/usr/bin/env python3
"""
Experiment Tracking and Monitoring Utility

Track ML experiments, log metrics, monitor system resources, and manage
experiment lifecycles with support for multiple backends and real-time monitoring.

Usage:
    python3 experiment_tracker.py init --project-name my-experiments
    python3 experiment_tracker.py start --name exp-001 --description "Baseline model"
    python3 experiment_tracker.py log --metric accuracy 0.95 --metric loss 0.15
    python3 experiment_tracker.py monitor --duration 3600 --interval 30
    python3 experiment_tracker.py compare --experiments exp-001 exp-002 exp-003
"""

import argparse
import json
import time
import psutil
import os
import sys
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
import sqlite3
import yaml
import subprocess
import signal
import csv
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class Metric:
    """Represents a single metric value."""
    name: str
    value: Union[float, int, str]
    timestamp: str
    step: Optional[int] = None
    epoch: Optional[int] = None
    tags: Optional[Dict[str, str]] = None


@dataclass
class SystemStats:
    """System resource statistics."""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_usage_percent: float
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None
    network_bytes_sent: Optional[int] = None
    network_bytes_recv: Optional[int] = None


@dataclass
class Experiment:
    """Experiment metadata."""
    name: str
    description: str
    created_at: str
    status: str = "running"
    tags: Optional[Dict[str, str]] = None
    parameters: Optional[Dict[str, Any]] = None
    artifacts: Optional[List[str]] = None
    finished_at: Optional[str] = None
    duration_seconds: Optional[float] = None


class ExperimentTracker:
    """Main experiment tracking class."""
    
    def __init__(self, project_path: str = ".", backend: str = "sqlite"):
        self.project_path = Path(project_path)
        self.experiments_dir = self.project_path / "experiments"
        self.experiments_dir.mkdir(exist_ok=True)
        
        self.backend = backend
        self.db_path = self.experiments_dir / "experiments.db"
        self.config_path = self.project_path / "experiment_config.yaml"
        
        self._init_database()
        self._monitoring_active = False
        self._monitoring_thread = None
    
    def _init_database(self):
        """Initialize SQLite database for experiment tracking."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Experiments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                name TEXT PRIMARY KEY,
                description TEXT,
                created_at TEXT,
                finished_at TEXT,
                status TEXT,
                duration_seconds REAL,
                tags TEXT,
                parameters TEXT,
                artifacts TEXT
            )
        """)
        
        # Metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_name TEXT,
                metric_name TEXT,
                value REAL,
                timestamp TEXT,
                step INTEGER,
                epoch INTEGER,
                tags TEXT,
                FOREIGN KEY (experiment_name) REFERENCES experiments (name)
            )
        """)
        
        # System stats table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_name TEXT,
                timestamp TEXT,
                cpu_percent REAL,
                memory_percent REAL,
                memory_used_gb REAL,
                memory_total_gb REAL,
                disk_usage_percent REAL,
                gpu_memory_used_mb REAL,
                gpu_memory_total_mb REAL,
                gpu_utilization REAL,
                network_bytes_sent INTEGER,
                network_bytes_recv INTEGER,
                FOREIGN KEY (experiment_name) REFERENCES experiments (name)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def init_project(self, project_name: str, description: str = "") -> Dict[str, str]:
        """Initialize experiment tracking for a project."""
        print(f"🚀 Initializing experiment tracking: {project_name}")
        
        config = {
            "project_name": project_name,
            "description": description,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "backend": self.backend,
            "current_experiment": None,
            "settings": {
                "auto_log_system_stats": True,
                "log_interval_seconds": 60,
                "keep_history_days": 30
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        print(f"✅ Project initialized: {self.config_path}")
        return config
    
    def start_experiment(self, name: str, description: str = "", 
                        tags: Optional[Dict[str, str]] = None,
                        parameters: Optional[Dict[str, Any]] = None) -> Experiment:
        """Start a new experiment."""
        print(f"🧪 Starting experiment: {name}")
        
        # Check if experiment already exists
        if self._experiment_exists(name):
            raise ValueError(f"Experiment '{name}' already exists")
        
        experiment = Experiment(
            name=name,
            description=description,
            created_at=datetime.now(timezone.utc).isoformat(),
            tags=tags or {},
            parameters=parameters or {},
            artifacts=[]
        )
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO experiments 
            (name, description, created_at, status, tags, parameters, artifacts)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            experiment.name,
            experiment.description,
            experiment.created_at,
            experiment.status,
            json.dumps(experiment.tags),
            json.dumps(experiment.parameters),
            json.dumps(experiment.artifacts)
        ))
        
        conn.commit()
        conn.close()
        
        # Update current experiment in config
        self._update_current_experiment(name)
        
        # Create experiment directory
        exp_dir = self.experiments_dir / name
        exp_dir.mkdir(exist_ok=True)
        
        # Save experiment metadata
        with open(exp_dir / "metadata.yaml", 'w') as f:
            yaml.dump(asdict(experiment), f, default_flow_style=False, indent=2)
        
        print(f"✅ Experiment '{name}' started")
        return experiment
    
    def log_metric(self, experiment_name: str, metric_name: str, value: Union[float, int, str],
                  step: Optional[int] = None, epoch: Optional[int] = None,
                  tags: Optional[Dict[str, str]] = None) -> Metric:
        """Log a metric for an experiment."""
        metric = Metric(
            name=metric_name,
            value=value,
            timestamp=datetime.now(timezone.utc).isoformat(),
            step=step,
            epoch=epoch,
            tags=tags or {}
        )
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO metrics 
            (experiment_name, metric_name, value, timestamp, step, epoch, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            experiment_name,
            metric.name,
            float(metric.value) if isinstance(metric.value, (int, float)) else 0,
            metric.timestamp,
            metric.step,
            metric.epoch,
            json.dumps(metric.tags)
        ))
        
        conn.commit()
        conn.close()
        
        print(f"📊 Logged metric: {metric_name} = {value}")
        return metric
    
    def log_metrics(self, experiment_name: str, metrics: Dict[str, Union[float, int, str]],
                   step: Optional[int] = None, epoch: Optional[int] = None) -> List[Metric]:
        """Log multiple metrics at once."""
        logged_metrics = []
        for name, value in metrics.items():
            metric = self.log_metric(experiment_name, name, value, step, epoch)
            logged_metrics.append(metric)
        
        return logged_metrics
    
    def log_parameters(self, experiment_name: str, parameters: Dict[str, Any]):
        """Log or update experiment parameters."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get existing parameters
        cursor.execute("SELECT parameters FROM experiments WHERE name = ?", (experiment_name,))
        result = cursor.fetchone()
        
        if result:
            existing_params = json.loads(result[0] or "{}")
            existing_params.update(parameters)
            
            cursor.execute(
                "UPDATE experiments SET parameters = ? WHERE name = ?",
                (json.dumps(existing_params), experiment_name)
            )
            
            conn.commit()
            print(f"📝 Updated parameters for experiment: {experiment_name}")
        
        conn.close()
    
    def finish_experiment(self, experiment_name: str, status: str = "completed") -> Experiment:
        """Finish an experiment."""
        print(f"🏁 Finishing experiment: {experiment_name}")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get experiment start time
        cursor.execute("SELECT created_at FROM experiments WHERE name = ?", (experiment_name,))
        result = cursor.fetchone()
        
        if not result:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        
        start_time = datetime.fromisoformat(result[0])
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()
        
        # Update experiment
        cursor.execute("""
            UPDATE experiments 
            SET status = ?, finished_at = ?, duration_seconds = ?
            WHERE name = ?
        """, (status, end_time.isoformat(), duration, experiment_name))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Experiment '{experiment_name}' finished ({duration:.1f}s)")
        
        # Stop monitoring if this was the current experiment
        if self._get_current_experiment() == experiment_name:
            self.stop_monitoring()
        
        return self.get_experiment(experiment_name)
    
    def get_experiment(self, experiment_name: str) -> Optional[Experiment]:
        """Get experiment by name."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM experiments WHERE name = ?", (experiment_name,))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return None
        
        return Experiment(
            name=result[0],
            description=result[1],
            created_at=result[2],
            finished_at=result[3],
            status=result[4],
            duration_seconds=result[5],
            tags=json.loads(result[6] or "{}"),
            parameters=json.loads(result[7] or "{}"),
            artifacts=json.loads(result[8] or "[]")
        )
    
    def list_experiments(self) -> List[Experiment]:
        """List all experiments."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM experiments ORDER BY created_at DESC")
        results = cursor.fetchall()
        conn.close()
        
        experiments = []
        for result in results:
            experiment = Experiment(
                name=result[0],
                description=result[1],
                created_at=result[2],
                finished_at=result[3],
                status=result[4],
                duration_seconds=result[5],
                tags=json.loads(result[6] or "{}"),
                parameters=json.loads(result[7] or "{}"),
                artifacts=json.loads(result[8] or "[]")
            )
            experiments.append(experiment)
        
        return experiments
    
    def get_metrics(self, experiment_name: str, metric_name: Optional[str] = None) -> List[Metric]:
        """Get metrics for an experiment."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if metric_name:
            cursor.execute("""
                SELECT metric_name, value, timestamp, step, epoch, tags
                FROM metrics 
                WHERE experiment_name = ? AND metric_name = ?
                ORDER BY timestamp
            """, (experiment_name, metric_name))
        else:
            cursor.execute("""
                SELECT metric_name, value, timestamp, step, epoch, tags
                FROM metrics 
                WHERE experiment_name = ?
                ORDER BY timestamp
            """, (experiment_name,))
        
        results = cursor.fetchall()
        conn.close()
        
        metrics = []
        for result in results:
            metric = Metric(
                name=result[0],
                value=result[1],
                timestamp=result[2],
                step=result[3],
                epoch=result[4],
                tags=json.loads(result[5] or "{}")
            )
            metrics.append(metric)
        
        return metrics
    
    def start_monitoring(self, experiment_name: str, interval: int = 60):
        """Start system monitoring for an experiment."""
        if self._monitoring_active:
            print("⚠️  Monitoring already active")
            return
        
        print(f"📈 Starting system monitoring for: {experiment_name}")
        print(f"⏱️  Monitoring interval: {interval}s")
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._monitor_loop,
            args=(experiment_name, interval),
            daemon=True
        )
        self._monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        if self._monitoring_active:
            print("🛑 Stopping system monitoring")
            self._monitoring_active = False
            if self._monitoring_thread:
                self._monitoring_thread.join(timeout=5)
    
    def _monitor_loop(self, experiment_name: str, interval: int):
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                stats = self._collect_system_stats()
                self._log_system_stats(experiment_name, stats)
                time.sleep(interval)
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                break
    
    def _collect_system_stats(self) -> SystemStats:
        """Collect current system statistics."""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Network
        network = psutil.net_io_counters()
        
        stats = SystemStats(
            timestamp=datetime.now(timezone.utc).isoformat(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / 1024**3,
            memory_total_gb=memory.total / 1024**3,
            disk_usage_percent=disk.percent,
            network_bytes_sent=network.bytes_sent,
            network_bytes_recv=network.bytes_recv
        )
        
        # GPU stats (if available)
        try:
            gpu_stats = self._get_gpu_stats()
            if gpu_stats:
                stats.gpu_memory_used_mb = gpu_stats.get('memory_used_mb')
                stats.gpu_memory_total_mb = gpu_stats.get('memory_total_mb')
                stats.gpu_utilization = gpu_stats.get('utilization')
        except:
            pass
        
        return stats
    
    def _get_gpu_stats(self) -> Optional[Dict[str, float]]:
        """Get GPU statistics using nvidia-smi."""
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines and lines[0]:
                    values = lines[0].split(', ')
                    return {
                        'memory_used_mb': float(values[0]),
                        'memory_total_mb': float(values[1]),
                        'utilization': float(values[2])
                    }
        except:
            pass
        
        return None
    
    def _log_system_stats(self, experiment_name: str, stats: SystemStats):
        """Log system stats to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO system_stats 
            (experiment_name, timestamp, cpu_percent, memory_percent, memory_used_gb,
             memory_total_gb, disk_usage_percent, gpu_memory_used_mb, gpu_memory_total_mb,
             gpu_utilization, network_bytes_sent, network_bytes_recv)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            experiment_name,
            stats.timestamp,
            stats.cpu_percent,
            stats.memory_percent,
            stats.memory_used_gb,
            stats.memory_total_gb,
            stats.disk_usage_percent,
            stats.gpu_memory_used_mb,
            stats.gpu_memory_total_mb,
            stats.gpu_utilization,
            stats.network_bytes_sent,
            stats.network_bytes_recv
        ))
        
        conn.commit()
        conn.close()
    
    def compare_experiments(self, experiment_names: List[str], 
                          metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compare metrics across multiple experiments."""
        print(f"📊 Comparing experiments: {', '.join(experiment_names)}")
        
        comparison = {
            "experiments": {},
            "metric_comparison": {},
            "summary": {}
        }
        
        for exp_name in experiment_names:
            experiment = self.get_experiment(exp_name)
            if not experiment:
                print(f"⚠️  Experiment '{exp_name}' not found")
                continue
            
            comparison["experiments"][exp_name] = asdict(experiment)
            
            # Get metrics
            exp_metrics = self.get_metrics(exp_name)
            
            # Group by metric name
            metric_groups = {}
            for metric in exp_metrics:
                if metric.name not in metric_groups:
                    metric_groups[metric.name] = []
                metric_groups[metric.name].append(metric.value)
            
            # Calculate statistics for each metric
            for metric_name, values in metric_groups.items():
                if metrics is None or metric_name in metrics:
                    if metric_name not in comparison["metric_comparison"]:
                        comparison["metric_comparison"][metric_name] = {}
                    
                    comparison["metric_comparison"][metric_name][exp_name] = {
                        "values": values,
                        "count": len(values),
                        "latest": values[-1] if values else None,
                        "best": max(values) if values else None,
                        "worst": min(values) if values else None,
                        "avg": sum(values) / len(values) if values else None
                    }
        
        return comparison
    
    def export_experiment_data(self, experiment_name: str, output_path: str, format: str = "csv"):
        """Export experiment data to file."""
        print(f"📤 Exporting experiment data: {experiment_name}")
        
        # Get experiment and metrics
        experiment = self.get_experiment(experiment_name)
        if not experiment:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        
        metrics = self.get_metrics(experiment_name)
        
        output_path = Path(output_path)
        
        if format == "csv":
            # Export metrics to CSV
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['metric_name', 'value', 'timestamp', 'step', 'epoch'])
                
                for metric in metrics:
                    writer.writerow([
                        metric.name,
                        metric.value,
                        metric.timestamp,
                        metric.step,
                        metric.epoch
                    ])
        
        elif format == "json":
            # Export everything to JSON
            data = {
                "experiment": asdict(experiment),
                "metrics": [asdict(metric) for metric in metrics]
            }
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        print(f"✅ Data exported to: {output_path}")
    
    def _experiment_exists(self, name: str) -> bool:
        """Check if experiment exists."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM experiments WHERE name = ?", (name,))
        result = cursor.fetchone()
        conn.close()
        return result is not None
    
    def _get_current_experiment(self) -> Optional[str]:
        """Get current experiment from config."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get("current_experiment")
        return None
    
    def _update_current_experiment(self, experiment_name: str):
        """Update current experiment in config."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {}
        
        config["current_experiment"] = experiment_name
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Experiment tracking and monitoring")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize experiment tracking')
    init_parser.add_argument('--project-name', required=True, help='Project name')
    init_parser.add_argument('--description', default='', help='Project description')

    # Start experiment
    start_parser = subparsers.add_parser('start', help='Start new experiment')
    start_parser.add_argument('--name', required=True, help='Experiment name')
    start_parser.add_argument('--description', default='', help='Experiment description')
    start_parser.add_argument('--tags', help='Tags in key=value format', nargs='*')
    start_parser.add_argument('--params', help='Parameters in key=value format', nargs='*')

    # Log metrics
    log_parser = subparsers.add_parser('log', help='Log metrics')
    log_parser.add_argument('--experiment', help='Experiment name (defaults to current)')
    log_parser.add_argument('--metric', action='append', nargs=2, metavar=('NAME', 'VALUE'),
                           help='Metric name and value')
    log_parser.add_argument('--step', type=int, help='Step number')
    log_parser.add_argument('--epoch', type=int, help='Epoch number')

    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Start system monitoring')
    monitor_parser.add_argument('--experiment', help='Experiment name (defaults to current)')
    monitor_parser.add_argument('--duration', type=int, help='Monitoring duration in seconds')
    monitor_parser.add_argument('--interval', type=int, default=60, help='Monitoring interval')

    # List experiments
    list_parser = subparsers.add_parser('list', help='List experiments')

    # Compare experiments
    compare_parser = subparsers.add_parser('compare', help='Compare experiments')
    compare_parser.add_argument('--experiments', nargs='+', required=True, help='Experiment names')
    compare_parser.add_argument('--metrics', nargs='*', help='Specific metrics to compare')

    # Finish experiment
    finish_parser = subparsers.add_parser('finish', help='Finish experiment')
    finish_parser.add_argument('--experiment', help='Experiment name (defaults to current)')
    finish_parser.add_argument('--status', default='completed', help='Final status')

    # Export data
    export_parser = subparsers.add_parser('export', help='Export experiment data')
    export_parser.add_argument('--experiment', required=True, help='Experiment name')
    export_parser.add_argument('--output', required=True, help='Output file path')
    export_parser.add_argument('--format', choices=['csv', 'json'], default='csv', help='Export format')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    tracker = ExperimentTracker()

    try:
        if args.command == 'init':
            tracker.init_project(args.project_name, args.description)

        elif args.command == 'start':
            # Parse tags and parameters
            tags = {}
            if args.tags:
                for tag in args.tags:
                    if '=' in tag:
                        key, value = tag.split('=', 1)
                        tags[key] = value

            params = {}
            if args.params:
                for param in args.params:
                    if '=' in param:
                        key, value = param.split('=', 1)
                        # Try to convert to appropriate type
                        try:
                            params[key] = float(value) if '.' in value else int(value)
                        except ValueError:
                            params[key] = value

            tracker.start_experiment(args.name, args.description, tags, params)

        elif args.command == 'log':
            experiment_name = args.experiment or tracker._get_current_experiment()
            if not experiment_name:
                print("❌ No experiment specified and no current experiment set")
                return

            if args.metric:
                for name, value in args.metric:
                    try:
                        numeric_value = float(value)
                    except ValueError:
                        numeric_value = value
                    
                    tracker.log_metric(experiment_name, name, numeric_value, args.step, args.epoch)

        elif args.command == 'monitor':
            experiment_name = args.experiment or tracker._get_current_experiment()
            if not experiment_name:
                print("❌ No experiment specified and no current experiment set")
                return

            tracker.start_monitoring(experiment_name, args.interval)
            
            if args.duration:
                print(f"⏱️  Monitoring for {args.duration} seconds...")
                time.sleep(args.duration)
                tracker.stop_monitoring()
            else:
                print("🔄 Monitoring started. Press Ctrl+C to stop.")
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    tracker.stop_monitoring()

        elif args.command == 'list':
            experiments = tracker.list_experiments()
            if experiments:
                print("🧪 Experiments:")
                for exp in experiments:
                    status_icon = "✅" if exp.status == "completed" else "🔄" if exp.status == "running" else "❌"
                    duration = f" ({exp.duration_seconds:.1f}s)" if exp.duration_seconds else ""
                    print(f"  {status_icon} {exp.name}: {exp.status}{duration}")
                    if exp.description:
                        print(f"    {exp.description}")
            else:
                print("No experiments found")

        elif args.command == 'compare':
            comparison = tracker.compare_experiments(args.experiments, args.metrics)
            
            print("📊 Experiment Comparison:")
            for exp_name, exp_data in comparison["experiments"].items():
                print(f"\n{exp_name}:")
                print(f"  Status: {exp_data['status']}")
                print(f"  Duration: {exp_data.get('duration_seconds', 'N/A')}s")
            
            print("\n📈 Metric Comparison:")
            for metric_name, metric_data in comparison["metric_comparison"].items():
                print(f"\n{metric_name}:")
                for exp_name, stats in metric_data.items():
                    latest = stats.get('latest', 'N/A')
                    best = stats.get('best', 'N/A')
                    print(f"  {exp_name}: latest={latest}, best={best}")

        elif args.command == 'finish':
            experiment_name = args.experiment or tracker._get_current_experiment()
            if not experiment_name:
                print("❌ No experiment specified and no current experiment set")
                return

            tracker.finish_experiment(experiment_name, args.status)

        elif args.command == 'export':
            tracker.export_experiment_data(args.experiment, args.output, args.format)

    except KeyboardInterrupt:
        if tracker._monitoring_active:
            tracker.stop_monitoring()
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()