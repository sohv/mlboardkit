#!/usr/bin/env python3
"""
scheduler_job.py

Comprehensive job scheduler for ML workflows with cron-like scheduling, 
job dependencies, resource monitoring, and advanced retry mechanisms.
"""

import argparse
import time
import subprocess
import shlex
import json
import signal
import threading
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

try:
    from croniter import croniter
    HAS_CRONITER = True
except ImportError:
    HAS_CRONITER = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@dataclass
class JobConfig:
    """Configuration for a scheduled job"""
    name: str
    command: str
    schedule: str  # cron expression or "interval:seconds"
    max_retries: int = 3
    timeout: int = 3600  # seconds
    depends_on: List[str] = None
    env_vars: Dict[str, str] = None
    working_dir: str = None
    resource_limits: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.depends_on is None:
            self.depends_on = []
        if self.env_vars is None:
            self.env_vars = {}
        if self.resource_limits is None:
            self.resource_limits = {"max_memory_mb": 4096, "max_cpu_percent": 80}


@dataclass
class JobRun:
    """Record of a job execution"""
    job_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"  # running, completed, failed, timeout
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    retry_count: int = 0
    resource_usage: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.resource_usage is None:
            self.resource_usage = {}


class JobScheduler:
    """Advanced job scheduler with dependency management and monitoring"""
    
    def __init__(self, config_file: str = None, log_dir: str = "logs"):
        self.jobs: Dict[str, JobConfig] = {}
        self.job_history: List[JobRun] = []
        self.running_jobs: Dict[str, subprocess.Popen] = {}
        self.lock = threading.Lock()
        self.shutdown_event = threading.Event()
        
        # Setup logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / "scheduler.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Load configuration if provided
        if config_file and Path(config_file).exists():
            self.load_config(config_file)
            
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown_event.set()
        self.stop_all_jobs()
    
    def load_config(self, config_file: str):
        """Load job configurations from JSON file"""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            for job_data in config_data.get('jobs', []):
                job = JobConfig(**job_data)
                self.add_job(job)
            
            self.logger.info(f"Loaded {len(self.jobs)} jobs from {config_file}")
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
    
    def save_config(self, config_file: str):
        """Save current job configurations to JSON file"""
        config_data = {
            'jobs': [asdict(job) for job in self.jobs.values()]
        }
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        self.logger.info(f"Saved configuration to {config_file}")
    
    def add_job(self, job: JobConfig):
        """Add a job to the scheduler"""
        self.jobs[job.name] = job
        self.logger.info(f"Added job: {job.name}")
    
    def remove_job(self, job_name: str):
        """Remove a job from the scheduler"""
        if job_name in self.jobs:
            del self.jobs[job_name]
            self.logger.info(f"Removed job: {job_name}")
    
    def get_next_run_time(self, job: JobConfig, base_time: datetime = None) -> datetime:
        """Calculate next run time for a job"""
        if base_time is None:
            base_time = datetime.now()
        
        if job.schedule.startswith("interval:"):
            seconds = int(job.schedule.split(":")[1])
            return base_time + timedelta(seconds=seconds)
        else:
            # Cron expression
            if not HAS_CRONITER:
                self.logger.error(f"croniter not available for job {job.name}, using 1 hour interval")
                return base_time + timedelta(hours=1)
            
            try:
                cron = croniter(job.schedule, base_time)
                return cron.get_next(datetime)
            except:
                self.logger.error(f"Invalid cron expression for {job.name}: {job.schedule}")
                return base_time + timedelta(hours=1)  # Default to 1 hour
    
    def check_dependencies(self, job: JobConfig) -> bool:
        """Check if job dependencies are satisfied"""
        if not job.depends_on:
            return True
        
        # Check if all dependent jobs completed successfully recently (within last hour)
        recent_time = datetime.now() - timedelta(hours=1)
        
        for dep_job in job.depends_on:
            dep_completed = False
            for run in reversed(self.job_history):
                if (run.job_name == dep_job and 
                    run.start_time >= recent_time and 
                    run.status == "completed"):
                    dep_completed = True
                    break
            
            if not dep_completed:
                return False
        
        return True
    
    def check_resource_limits(self, job: JobConfig) -> bool:
        """Check if system resources are available for job"""
        if not HAS_PSUTIL:
            self.logger.warning("psutil not available, skipping resource checks")
            return True
        
        limits = job.resource_limits
        
        # Check memory
        memory = psutil.virtual_memory()
        available_mb = memory.available / (1024 * 1024)
        if available_mb < limits.get("max_memory_mb", 4096):
            return False
        
        # Check CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > (100 - limits.get("max_cpu_percent", 80)):
            return False
        
        return True
    
    def monitor_job_resources(self, process: subprocess.Popen, job_run: JobRun):
        """Monitor resource usage of running job"""
        if not HAS_PSUTIL:
            return
        
        try:
            psutil_process = psutil.Process(process.pid)
            while process.poll() is None:
                try:
                    memory_info = psutil_process.memory_info()
                    cpu_percent = psutil_process.cpu_percent()
                    
                    job_run.resource_usage = {
                        "memory_mb": memory_info.rss / (1024 * 1024),
                        "cpu_percent": cpu_percent,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    time.sleep(5)  # Monitor every 5 seconds
                except psutil.NoSuchProcess:
                    break
        except Exception as e:
            self.logger.warning(f"Resource monitoring failed: {e}")
    
    def run_job(self, job: JobConfig) -> JobRun:
        """Execute a single job"""
        job_run = JobRun(
            job_name=job.name,
            start_time=datetime.now()
        )
        
        try:
            # Setup environment
            env = dict(os.environ)
            env.update(job.env_vars)
            
            # Create log file for this run
            log_file = self.log_dir / f"{job.name}_{job_run.start_time.strftime('%Y%m%d_%H%M%S')}.log"
            
            with open(log_file, 'w') as f:
                # Start process
                process = subprocess.Popen(
                    shlex.split(job.command),
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=env,
                    cwd=job.working_dir
                )
                
                # Start resource monitoring in background
                monitor_thread = threading.Thread(
                    target=self.monitor_job_resources,
                    args=(process, job_run)
                )
                monitor_thread.daemon = True
                monitor_thread.start()
                
                # Store running job
                with self.lock:
                    self.running_jobs[job.name] = process
                
                # Wait for completion with timeout
                try:
                    process.wait(timeout=job.timeout)
                    job_run.exit_code = process.returncode
                    job_run.status = "completed" if process.returncode == 0 else "failed"
                except subprocess.TimeoutExpired:
                    process.kill()
                    job_run.status = "timeout"
                    job_run.exit_code = -1
                
                # Read output
                with open(log_file, 'r') as f:
                    output = f.read()
                    job_run.stdout = output
                
        except Exception as e:
            job_run.status = "failed"
            job_run.stderr = str(e)
            self.logger.error(f"Job {job.name} failed: {e}")
        
        finally:
            job_run.end_time = datetime.now()
            with self.lock:
                if job.name in self.running_jobs:
                    del self.running_jobs[job.name]
            
            self.job_history.append(job_run)
            self.logger.info(f"Job {job.name} finished with status: {job_run.status}")
        
        return job_run
    
    def run_job_with_retries(self, job: JobConfig):
        """Run job with retry logic"""
        for attempt in range(job.max_retries + 1):
            if not self.check_dependencies(job):
                self.logger.warning(f"Dependencies not met for {job.name}, skipping")
                return
            
            if not self.check_resource_limits(job):
                self.logger.warning(f"Resource limits exceeded for {job.name}, skipping")
                return
            
            job_run = self.run_job(job)
            job_run.retry_count = attempt
            
            if job_run.status == "completed":
                return
            
            if attempt < job.max_retries:
                wait_time = 2 ** attempt  # Exponential backoff
                self.logger.info(f"Retrying {job.name} in {wait_time} seconds (attempt {attempt + 1})")
                time.sleep(wait_time)
    
    def stop_all_jobs(self):
        """Stop all running jobs"""
        with self.lock:
            for job_name, process in self.running_jobs.items():
                try:
                    process.terminate()
                    self.logger.info(f"Terminated job: {job_name}")
                except:
                    pass
    
    def run_scheduler(self):
        """Main scheduler loop"""
        self.logger.info("Starting job scheduler")
        job_timers = {}
        
        # Initialize next run times
        for job in self.jobs.values():
            job_timers[job.name] = self.get_next_run_time(job)
        
        while not self.shutdown_event.is_set():
            current_time = datetime.now()
            
            # Check each job
            for job in self.jobs.values():
                if current_time >= job_timers[job.name]:
                    # Run job in separate thread
                    job_thread = threading.Thread(
                        target=self.run_job_with_retries,
                        args=(job,)
                    )
                    job_thread.daemon = True
                    job_thread.start()
                    
                    # Schedule next run
                    job_timers[job.name] = self.get_next_run_time(job, current_time)
                    self.logger.info(f"Scheduled next run of {job.name} at {job_timers[job.name]}")
            
            # Sleep for a short interval
            time.sleep(10)
        
        self.logger.info("Scheduler stopped")
    
    def status_report(self) -> str:
        """Generate status report"""
        report = ["=== Job Scheduler Status ===\n"]
        
        # Running jobs
        with self.lock:
            if self.running_jobs:
                report.append("Running Jobs:")
                for job_name in self.running_jobs:
                    report.append(f"  - {job_name}")
            else:
                report.append("No jobs currently running")
        
        # Recent job history
        recent_runs = [run for run in self.job_history if run.start_time >= datetime.now() - timedelta(hours=24)]
        if recent_runs:
            report.append(f"\nRecent Job Runs (last 24 hours): {len(recent_runs)}")
            for run in recent_runs[-5:]:  # Show last 5
                duration = (run.end_time - run.start_time).total_seconds() if run.end_time else 0
                report.append(f"  - {run.job_name}: {run.status} ({duration:.1f}s)")
        
        return "\n".join(report)


def create_sample_config():
    """Create a sample configuration file"""
    sample_jobs = [
        {
            "name": "data_backup",
            "command": "rsync -av /data/ /backup/",
            "schedule": "0 2 * * *",  # Daily at 2 AM
            "max_retries": 2,
            "timeout": 7200,
            "resource_limits": {"max_memory_mb": 2048, "max_cpu_percent": 50}
        },
        {
            "name": "model_training",
            "command": "python train_model.py --config config.yaml",
            "schedule": "interval:3600",  # Every hour
            "depends_on": ["data_backup"],
            "max_retries": 1,
            "timeout": 14400,
            "env_vars": {"CUDA_VISIBLE_DEVICES": "0"},
            "resource_limits": {"max_memory_mb": 8192, "max_cpu_percent": 90}
        },
        {
            "name": "cleanup_logs",
            "command": "find /logs -name '*.log' -mtime +7 -delete",
            "schedule": "0 0 * * 0",  # Weekly on Sunday
            "max_retries": 1,
            "timeout": 600
        }
    ]
    
    config = {"jobs": sample_jobs}
    with open("scheduler_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("Sample configuration created: scheduler_config.json")


def main():
    parser = argparse.ArgumentParser(description="Advanced ML Job Scheduler")
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--log-dir', default='logs', help='Log directory')
    parser.add_argument('--create-sample-config', action='store_true', 
                       help='Create sample configuration file')
    parser.add_argument('--status', action='store_true', help='Show scheduler status')
    
    # Simple mode for backward compatibility
    parser.add_argument('--cmd', help='Single command to run')
    parser.add_argument('--interval', type=int, default=60, help='Seconds between runs')
    parser.add_argument('--iterations', type=int, default=1, help='Number of iterations (0 = infinite)')
    
    args = parser.parse_args()
    
    if args.create_sample_config:
        create_sample_config()
        return
    
    scheduler = JobScheduler(config_file=args.config, log_dir=args.log_dir)
    
    if args.status:
        print(scheduler.status_report())
        return
    
    # Simple mode for single command
    if args.cmd:
        job = JobConfig(
            name="simple_job",
            command=args.cmd,
            schedule=f"interval:{args.interval}",
            max_retries=0
        )
        scheduler.add_job(job)
        
        if args.iterations == 1:
            scheduler.run_job_with_retries(job)
        else:
            scheduler.run_scheduler()
    else:
        scheduler.run_scheduler()

if __name__ == '__main__':
    main()
