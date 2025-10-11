#!/usr/bin/env python3
"""
timer.py

Comprehensive timing and profiling utilities for ML workflows.
Includes context managers, decorators, benchmarking, and performance analysis.
"""

import argparse
import time
import json
import functools
import statistics
import threading
from collections import defaultdict, deque
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, asdict

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import memory_profiler
    HAS_MEMORY_PROFILER = True
except ImportError:
    HAS_MEMORY_PROFILER = False


@dataclass
class TimingResult:
    """Result of a timing measurement"""
    name: str
    duration: float
    start_time: datetime
    end_time: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BenchmarkResult:
    """Result of a benchmark run"""
    name: str
    iterations: int
    times: List[float]
    mean: float
    median: float
    std: float
    min_time: float
    max_time: float
    percentiles: Dict[str, float]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Timer:
    """High-precision timer with context manager and decorator support"""
    
    def __init__(self, name: str = "Timer", precision: int = 6):
        self.name = name
        self.precision = precision
        self.start_time = None
        self.end_time = None
        self.duration = None
        self.is_running = False
    
    def start(self):
        """Start the timer"""
        if self.is_running:
            raise RuntimeError("Timer is already running")
        
        self.start_time = time.perf_counter()
        self.is_running = True
        return self
    
    def stop(self):
        """Stop the timer and return duration"""
        if not self.is_running:
            raise RuntimeError("Timer is not running")
        
        self.end_time = time.perf_counter()
        self.duration = round(self.end_time - self.start_time, self.precision)
        self.is_running = False
        return self.duration
    
    def elapsed(self) -> float:
        """Get elapsed time without stopping"""
        if not self.is_running:
            return self.duration or 0.0
        
        return round(time.perf_counter() - self.start_time, self.precision)
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    def __call__(self, func):
        """Use as decorator"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with Timer(f"{func.__name__}") as timer:
                result = func(*args, **kwargs)
            print(f"{timer.name} took {timer.duration:.{self.precision}f} seconds")
            return result
        return wrapper


class TimingManager:
    """Advanced timing manager with statistics and reporting"""
    
    def __init__(self, auto_save: bool = False, save_file: str = "timing_results.json"):
        self.timings: List[TimingResult] = []
        self.active_timers: Dict[str, Timer] = {}
        self.auto_save = auto_save
        self.save_file = save_file
        self._lock = threading.Lock()
    
    @contextmanager
    def time_block(self, name: str, metadata: Dict[str, Any] = None):
        """Context manager for timing code blocks"""
        timer = Timer(name)
        start_dt = datetime.now()
        
        try:
            timer.start()
            yield timer
        finally:
            timer.stop()
            end_dt = datetime.now()
            
            result = TimingResult(
                name=name,
                duration=timer.duration,
                start_time=start_dt,
                end_time=end_dt,
                metadata=metadata or {}
            )
            
            with self._lock:
                self.timings.append(result)
            
            if self.auto_save:
                self.save_results()
    
    def start_timer(self, name: str, metadata: Dict[str, Any] = None):
        """Start a named timer"""
        if name in self.active_timers:
            raise ValueError(f"Timer '{name}' is already running")
        
        timer = Timer(name)
        timer.start()
        timer.metadata = metadata or {}
        timer.start_dt = datetime.now()
        
        self.active_timers[name] = timer
        return timer
    
    def stop_timer(self, name: str) -> TimingResult:
        """Stop a named timer and record result"""
        if name not in self.active_timers:
            raise ValueError(f"Timer '{name}' is not running")
        
        timer = self.active_timers.pop(name)
        timer.stop()
        end_dt = datetime.now()
        
        result = TimingResult(
            name=name,
            duration=timer.duration,
            start_time=timer.start_dt,
            end_time=end_dt,
            metadata=getattr(timer, 'metadata', {})
        )
        
        with self._lock:
            self.timings.append(result)
        
        if self.auto_save:
            self.save_results()
        
        return result
    
    def get_statistics(self, name: str = None) -> Dict[str, Any]:
        """Get timing statistics for all or specific timers"""
        if name:
            durations = [t.duration for t in self.timings if t.name == name]
            if not durations:
                return {}
        else:
            durations = [t.duration for t in self.timings]
        
        if not durations:
            return {}
        
        return {
            'count': len(durations),
            'total': sum(durations),
            'mean': statistics.mean(durations),
            'median': statistics.median(durations),
            'std': statistics.stdev(durations) if len(durations) > 1 else 0.0,
            'min': min(durations),
            'max': max(durations),
            'percentiles': {
                'p25': self._percentile(durations, 25),
                'p75': self._percentile(durations, 75),
                'p90': self._percentile(durations, 90),
                'p95': self._percentile(durations, 95),
                'p99': self._percentile(durations, 99)
            }
        }
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile"""
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def get_summary_report(self) -> str:
        """Generate a comprehensive timing report"""
        if not self.timings:
            return "No timing data available"
        
        report = ["=== Timing Summary Report ===\n"]
        
        # Overall statistics
        all_stats = self.get_statistics()
        report.append(f"Total measurements: {all_stats['count']}")
        report.append(f"Total time: {all_stats['total']:.6f}s")
        report.append(f"Average time: {all_stats['mean']:.6f}s")
        report.append(f"Median time: {all_stats['median']:.6f}s")
        report.append("")
        
        # Per-timer statistics
        timer_names = list(set(t.name for t in self.timings))
        for name in sorted(timer_names):
            stats = self.get_statistics(name)
            report.append(f"Timer: {name}")
            report.append(f"  Count: {stats['count']}")
            report.append(f"  Mean: {stats['mean']:.6f}s")
            report.append(f"  Std: {stats['std']:.6f}s")
            report.append(f"  Range: {stats['min']:.6f}s - {stats['max']:.6f}s")
            report.append("")
        
        return "\n".join(report)
    
    def save_results(self, filename: str = None):
        """Save timing results to JSON file"""
        filename = filename or self.save_file
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'total_measurements': len(self.timings),
            'timings': [asdict(timing) for timing in self.timings],
            'statistics': self.get_statistics()
        }
        
        # Convert datetime objects to strings
        for timing in data['timings']:
            timing['start_time'] = timing['start_time'].isoformat()
            timing['end_time'] = timing['end_time'].isoformat()
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def clear(self):
        """Clear all timing data"""
        with self._lock:
            self.timings.clear()
            self.active_timers.clear()


class Benchmark:
    """Advanced benchmarking utilities"""
    
    def __init__(self, warmup_runs: int = 1, precision: int = 6):
        self.warmup_runs = warmup_runs
        self.precision = precision
        self.results: List[BenchmarkResult] = []
    
    def benchmark_function(self, 
                         func: Callable, 
                         iterations: int = 100,
                         name: str = None,
                         *args, **kwargs) -> BenchmarkResult:
        """Benchmark a function with multiple iterations"""
        name = name or func.__name__
        
        # Warmup runs
        for _ in range(self.warmup_runs):
            func(*args, **kwargs)
        
        # Actual benchmark runs
        times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            func(*args, **kwargs)
            end_time = time.perf_counter()
            times.append(round(end_time - start_time, self.precision))
        
        # Calculate statistics
        mean_time = statistics.mean(times)
        median_time = statistics.median(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0.0
        min_time = min(times)
        max_time = max(times)
        
        percentiles = {
            'p25': self._percentile(times, 25),
            'p50': self._percentile(times, 50),
            'p75': self._percentile(times, 75),
            'p90': self._percentile(times, 90),
            'p95': self._percentile(times, 95),
            'p99': self._percentile(times, 99)
        }
        
        result = BenchmarkResult(
            name=name,
            iterations=iterations,
            times=times,
            mean=mean_time,
            median=median_time,
            std=std_time,
            min_time=min_time,
            max_time=max_time,
            percentiles=percentiles
        )
        
        self.results.append(result)
        return result
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile"""
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def compare_functions(self, functions: List[Callable], 
                         iterations: int = 100,
                         *args, **kwargs) -> Dict[str, BenchmarkResult]:
        """Compare multiple functions"""
        results = {}
        
        for func in functions:
            result = self.benchmark_function(func, iterations, *args, **kwargs)
            results[func.__name__] = result
        
        return results
    
    def generate_report(self) -> str:
        """Generate benchmark report"""
        if not self.results:
            return "No benchmark results available"
        
        report = ["=== Benchmark Report ===\n"]
        
        for result in self.results:
            report.append(f"Function: {result.name}")
            report.append(f"  Iterations: {result.iterations}")
            report.append(f"  Mean: {result.mean:.6f}s")
            report.append(f"  Median: {result.median:.6f}s")
            report.append(f"  Std: {result.std:.6f}s")
            report.append(f"  Min: {result.min_time:.6f}s")
            report.append(f"  Max: {result.max_time:.6f}s")
            report.append(f"  P95: {result.percentiles['p95']:.6f}s")
            report.append("")
        
        return "\n".join(report)


class PerformanceProfiler:
    """Performance profiler with memory and CPU monitoring"""
    
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.monitoring = False
        self.metrics: List[Dict[str, Any]] = []
        self._thread = None
    
    def start_monitoring(self):
        """Start performance monitoring"""
        if not HAS_PSUTIL:
            print("Warning: psutil not available, performance monitoring disabled")
            return
        
        self.monitoring = True
        self.metrics.clear()
        self._thread = threading.Thread(target=self._monitor_loop)
        self._thread.daemon = True
        self._thread.start()
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return summary"""
        self.monitoring = False
        if self._thread:
            self._thread.join()
        
        if not self.metrics:
            return {}
        
        # Calculate summary statistics
        cpu_values = [m['cpu_percent'] for m in self.metrics]
        memory_values = [m['memory_mb'] for m in self.metrics]
        
        return {
            'duration': len(self.metrics) * self.interval,
            'samples': len(self.metrics),
            'cpu': {
                'mean': statistics.mean(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values)
            },
            'memory': {
                'mean': statistics.mean(memory_values),
                'max': max(memory_values),
                'min': min(memory_values)
            },
            'raw_data': self.metrics
        }
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                metric = {
                    'timestamp': time.time(),
                    'cpu_percent': process.cpu_percent(),
                    'memory_mb': process.memory_info().rss / (1024 * 1024)
                }
                self.metrics.append(metric)
                time.sleep(self.interval)
            except Exception:
                break
    
    @contextmanager
    def profile_block(self, name: str = "code_block"):
        """Context manager for profiling code blocks"""
        self.start_monitoring()
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            summary = self.stop_monitoring()
            
            if summary:
                summary['name'] = name
                summary['wall_time'] = end_time - start_time
                print(f"\nProfile Summary for '{name}':")
                print(f"  Wall time: {summary['wall_time']:.3f}s")
                print(f"  CPU mean: {summary['cpu']['mean']:.1f}%")
                print(f"  Memory mean: {summary['memory']['mean']:.1f} MB")


# Decorator functions
def time_it(name: str = None, precision: int = 6):
    """Decorator to time function execution"""
    def decorator(func):
        timer_name = name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            duration = round(end_time - start_time, precision)
            print(f"{timer_name} took {duration:.{precision}f} seconds")
            return result
        return wrapper
    return decorator


def benchmark_it(iterations: int = 10, warmup: int = 1):
    """Decorator to benchmark function execution"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            benchmark = Benchmark(warmup_runs=warmup)
            result = benchmark.benchmark_function(func, iterations, *args, **kwargs)
            print(f"\nBenchmark results for {func.__name__}:")
            print(f"  Iterations: {result.iterations}")
            print(f"  Mean: {result.mean:.6f}s")
            print(f"  Std: {result.std:.6f}s")
            print(f"  Min/Max: {result.min_time:.6f}s / {result.max_time:.6f}s")
            
            # Return the actual function result from the last iteration
            return func(*args, **kwargs)
        return wrapper
    return decorator


def main():
    parser = argparse.ArgumentParser(description="Comprehensive timing utilities")
    parser.add_argument('--demo', action='store_true', help='Run demonstration')
    parser.add_argument('--benchmark', help='Python file with functions to benchmark')
    parser.add_argument('--iterations', type=int, default=100, help='Benchmark iterations')
    parser.add_argument('--profile', action='store_true', help='Enable performance profiling')
    parser.add_argument('--output', help='Output file for results')
    
    args = parser.parse_args()
    
    if args.demo:
        # Demonstration of timing features
        timing_manager = TimingManager(auto_save=True, save_file="demo_timing.json")
        
        # Time some operations
        with timing_manager.time_block("sleep_operation"):
            time.sleep(0.1)
        
        with timing_manager.time_block("computation"):
            result = sum(range(1000000))
        
        # Start/stop timer example
        timing_manager.start_timer("manual_timer")
        time.sleep(0.05)
        timing_manager.stop_timer("manual_timer")
        
        # Print summary
        print(timing_manager.get_summary_report())
        
        # Benchmark example
        benchmark = Benchmark()
        
        def test_function():
            return sum(range(10000))
        
        result = benchmark.benchmark_function(test_function, iterations=50)
        print("\nBenchmark example:")
        print(f"Mean execution time: {result.mean:.6f}s")
        
        # Performance profiling example
        if HAS_PSUTIL:
            profiler = PerformanceProfiler()
            with profiler.profile_block("heavy_computation"):
                # Simulate heavy computation
                result = sum(i**2 for i in range(100000))


if __name__ == "__main__":
    main()