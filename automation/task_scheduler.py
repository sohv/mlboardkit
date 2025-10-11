#!/usr/bin/env python3
"""
task_scheduler.py

Schedule and execute automated tasks.
"""

import argparse
import json
import time
import subprocess
import schedule
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import threading
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TaskScheduler:
    def __init__(self, config_file: str = 'scheduler_config.json'):
        self.config_file = config_file
        self.tasks = []
        self.running = False
        self.scheduler_thread = None
        self.load_config()
    
    def load_config(self):
        """Load scheduler configuration"""
        config_path = Path(self.config_file)
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.tasks = config.get('tasks', [])
                logger.info(f"Loaded {len(self.tasks)} tasks from {self.config_file}")
        else:
            self.tasks = []
            logger.info("No existing config found, starting with empty task list")
    
    def save_config(self):
        """Save scheduler configuration"""
        config = {
            'tasks': self.tasks,
            'last_updated': datetime.now().isoformat()
        }
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logger.info(f"Configuration saved to {self.config_file}")
    
    def add_task(self, task: Dict[str, Any]) -> str:
        """Add a new task to the scheduler"""
        
        # Validate task
        required_fields = ['name', 'command', 'schedule']
        for field in required_fields:
            if field not in task:
                raise ValueError(f"Task missing required field: {field}")
        
        # Add metadata
        task['id'] = f"task_{len(self.tasks) + 1}_{int(time.time())}"
        task['created'] = datetime.now().isoformat()
        task['status'] = 'active'
        task['last_run'] = None
        task['next_run'] = None
        task['run_count'] = 0
        task['success_count'] = 0
        task['error_count'] = 0
        
        self.tasks.append(task)
        self.save_config()
        
        logger.info(f"Added task: {task['name']} ({task['id']})")
        return task['id']
    
    def remove_task(self, task_id: str) -> bool:
        """Remove a task from the scheduler"""
        for i, task in enumerate(self.tasks):
            if task['id'] == task_id:
                removed_task = self.tasks.pop(i)
                self.save_config()
                logger.info(f"Removed task: {removed_task['name']} ({task_id})")
                return True
        return False
    
    def update_task(self, task_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing task"""
        for task in self.tasks:
            if task['id'] == task_id:
                task.update(updates)
                task['updated'] = datetime.now().isoformat()
                self.save_config()
                logger.info(f"Updated task: {task['name']} ({task_id})")
                return True
        return False
    
    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single task"""
        
        logger.info(f"Executing task: {task['name']}")
        
        start_time = datetime.now()
        result = {
            'task_id': task['id'],
            'task_name': task['name'],
            'start_time': start_time.isoformat(),
            'success': False,
            'output': '',
            'error': '',
            'duration': 0
        }
        
        try:
            # Update task metadata
            task['last_run'] = start_time.isoformat()
            task['run_count'] += 1
            
            # Execute command
            command = task['command']
            working_dir = task.get('working_directory', None)
            timeout = task.get('timeout', 300)  # 5 minute default timeout
            
            if isinstance(command, str):
                # Shell command
                process = subprocess.run(
                    command,
                    shell=True,
                    cwd=working_dir,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                
                result['output'] = process.stdout
                result['error'] = process.stderr
                result['success'] = process.returncode == 0
                
            elif isinstance(command, list):
                # Command list
                process = subprocess.run(
                    command,
                    cwd=working_dir,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                
                result['output'] = process.stdout
                result['error'] = process.stderr
                result['success'] = process.returncode == 0
            
            else:
                raise ValueError(f"Invalid command type: {type(command)}")
            
            # Update success/error counts
            if result['success']:
                task['success_count'] += 1
                logger.info(f"Task completed successfully: {task['name']}")
            else:
                task['error_count'] += 1
                logger.error(f"Task failed: {task['name']} - {result['error']}")
        
        except subprocess.TimeoutExpired:
            result['error'] = f"Task timed out after {timeout} seconds"
            task['error_count'] += 1
            logger.error(f"Task timed out: {task['name']}")
        
        except Exception as e:
            result['error'] = str(e)
            task['error_count'] += 1
            logger.error(f"Task execution error: {task['name']} - {e}")
        
        finally:
            end_time = datetime.now()
            result['end_time'] = end_time.isoformat()
            result['duration'] = (end_time - start_time).total_seconds()
            
            # Save execution log
            self.log_execution(result)
            self.save_config()
        
        return result
    
    def log_execution(self, result: Dict[str, Any]):
        """Log task execution results"""
        log_file = Path('task_execution.log')
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"{json.dumps(result)}\n")
    
    def parse_schedule(self, schedule_str: str) -> Any:
        """Parse schedule string and return schedule object"""
        
        # Parse different schedule formats
        if schedule_str.startswith('every '):
            # Format: "every 5 minutes", "every 1 hour", "every day at 09:00"
            parts = schedule_str.split()
            
            if len(parts) >= 3:
                interval = int(parts[1])
                unit = parts[2].lower()
                
                if unit.startswith('second'):
                    return schedule.every(interval).seconds
                elif unit.startswith('minute'):
                    return schedule.every(interval).minutes
                elif unit.startswith('hour'):
                    return schedule.every(interval).hours
                elif unit.startswith('day'):
                    if 'at' in parts:
                        time_str = parts[parts.index('at') + 1]
                        return schedule.every(interval).days.at(time_str)
                    else:
                        return schedule.every(interval).days
                elif unit.startswith('week'):
                    return schedule.every(interval).weeks
        
        elif schedule_str.startswith('daily at '):
            # Format: "daily at 09:00"
            time_str = schedule_str.split('at ')[1]
            return schedule.every().day.at(time_str)
        
        elif schedule_str.startswith('weekly on '):
            # Format: "weekly on monday at 09:00"
            parts = schedule_str.split()
            day = parts[2].lower()
            if 'at' in parts:
                time_str = parts[parts.index('at') + 1]
                return getattr(schedule.every(), day).at(time_str)
            else:
                return getattr(schedule.every(), day)
        
        elif schedule_str == 'once':
            # Run once immediately
            return None
        
        else:
            raise ValueError(f"Unsupported schedule format: {schedule_str}")
    
    def setup_schedules(self):
        """Set up all task schedules"""
        schedule.clear()  # Clear existing schedules
        
        for task in self.tasks:
            if task['status'] != 'active':
                continue
            
            schedule_str = task['schedule']
            
            try:
                if schedule_str == 'once':
                    # Execute once immediately
                    self.execute_task(task)
                else:
                    # Set up recurring schedule
                    scheduled_job = self.parse_schedule(schedule_str)
                    if scheduled_job:
                        scheduled_job.do(self.execute_task, task)
                        
                        # Calculate next run time
                        next_run = scheduled_job.next_run
                        if next_run:
                            task['next_run'] = next_run.isoformat()
                
            except Exception as e:
                logger.error(f"Error setting up schedule for task {task['name']}: {e}")
                task['status'] = 'error'
    
    def start_scheduler(self):
        """Start the task scheduler"""
        if self.running:
            logger.warning("Scheduler is already running")
            return
        
        self.running = True
        self.setup_schedules()
        
        def run_scheduler():
            logger.info("Task scheduler started")
            while self.running:
                schedule.run_pending()
                time.sleep(1)
            logger.info("Task scheduler stopped")
        
        self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()
    
    def stop_scheduler(self):
        """Stop the task scheduler"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        logger.info("Scheduler stop requested")
    
    def list_tasks(self) -> List[Dict[str, Any]]:
        """List all tasks"""
        return self.tasks
    
    def get_task_status(self, task_id: str = None) -> Dict[str, Any]:
        """Get status of tasks"""
        if task_id:
            # Get specific task status
            for task in self.tasks:
                if task['id'] == task_id:
                    return {
                        'task': task,
                        'scheduled_jobs': len(schedule.jobs)
                    }
            return {'error': 'Task not found'}
        
        else:
            # Get overall status
            active_tasks = sum(1 for task in self.tasks if task['status'] == 'active')
            total_runs = sum(task.get('run_count', 0) for task in self.tasks)
            total_successes = sum(task.get('success_count', 0) for task in self.tasks)
            total_errors = sum(task.get('error_count', 0) for task in self.tasks)
            
            return {
                'total_tasks': len(self.tasks),
                'active_tasks': active_tasks,
                'scheduled_jobs': len(schedule.jobs),
                'total_runs': total_runs,
                'total_successes': total_successes,
                'total_errors': total_errors,
                'success_rate': total_successes / max(total_runs, 1),
                'running': self.running
            }

def create_sample_tasks() -> List[Dict[str, Any]]:
    """Create sample tasks for demonstration"""
    return [
        {
            'name': 'Daily Backup',
            'description': 'Backup important files daily',
            'command': 'rsync -av --delete /home/user/important/ /backup/daily/',
            'schedule': 'daily at 02:00',
            'timeout': 1800,
            'working_directory': '/home/user'
        },
        {
            'name': 'System Update Check',
            'description': 'Check for system updates',
            'command': ['apt', 'list', '--upgradable'],
            'schedule': 'weekly on sunday at 08:00',
            'timeout': 300
        },
        {
            'name': 'Log Cleanup',
            'description': 'Clean old log files',
            'command': 'find /var/log -name "*.log" -mtime +30 -delete',
            'schedule': 'weekly on monday at 01:00',
            'timeout': 600
        },
        {
            'name': 'Health Check',
            'description': 'Check system health',
            'command': 'python3 health_check.py',
            'schedule': 'every 1 hour',
            'timeout': 120
        }
    ]

def main():
    parser = argparse.ArgumentParser(description="Task scheduler for automated execution")
    parser.add_argument('command', choices=['start', 'stop', 'add', 'remove', 'list', 'status', 'run'],
                       help='Command to execute')
    parser.add_argument('--config', default='scheduler_config.json',
                       help='Configuration file path')
    parser.add_argument('--task-id', help='Task ID for remove/status commands')
    parser.add_argument('--name', help='Task name for add command')
    parser.add_argument('--command-str', help='Command to execute for add command')
    parser.add_argument('--schedule', help='Schedule string for add command')
    parser.add_argument('--description', help='Task description for add command')
    parser.add_argument('--timeout', type=int, default=300, help='Task timeout in seconds')
    parser.add_argument('--working-dir', help='Working directory for task execution')
    parser.add_argument('--sample-tasks', action='store_true', help='Add sample tasks')
    parser.add_argument('--daemon', action='store_true', help='Run scheduler as daemon')
    
    args = parser.parse_args()
    
    scheduler = TaskScheduler(args.config)
    
    if args.command == 'start':
        if args.sample_tasks:
            print("Adding sample tasks...")
            sample_tasks = create_sample_tasks()
            for task in sample_tasks:
                scheduler.add_task(task)
            print(f"Added {len(sample_tasks)} sample tasks")
        
        print("Starting task scheduler...")
        scheduler.start_scheduler()
        
        if args.daemon:
            print("Running as daemon (Ctrl+C to stop)")
            try:
                while True:
                    time.sleep(60)
                    status = scheduler.get_task_status()
                    print(f"Status: {status['active_tasks']} active tasks, "
                          f"{status['total_runs']} total runs, "
                          f"{status['success_rate']:.1%} success rate")
            except KeyboardInterrupt:
                print("\nStopping scheduler...")
                scheduler.stop_scheduler()
        else:
            print("Scheduler started in background")
    
    elif args.command == 'stop':
        print("Stopping task scheduler...")
        scheduler.stop_scheduler()
    
    elif args.command == 'add':
        if not all([args.name, args.command_str, args.schedule]):
            print("Error: --name, --command-str, and --schedule are required for add command")
            return
        
        task = {
            'name': args.name,
            'command': args.command_str,
            'schedule': args.schedule,
            'description': args.description or '',
            'timeout': args.timeout
        }
        
        if args.working_dir:
            task['working_directory'] = args.working_dir
        
        task_id = scheduler.add_task(task)
        print(f"Added task: {args.name} ({task_id})")
    
    elif args.command == 'remove':
        if not args.task_id:
            print("Error: --task-id is required for remove command")
            return
        
        if scheduler.remove_task(args.task_id):
            print(f"Removed task: {args.task_id}")
        else:
            print(f"Task not found: {args.task_id}")
    
    elif args.command == 'list':
        tasks = scheduler.list_tasks()
        
        if not tasks:
            print("No tasks configured")
            return
        
        print(f"{'ID':<20} {'Name':<25} {'Schedule':<20} {'Status':<10} {'Runs':<6} {'Success':<7}")
        print("-" * 88)
        
        for task in tasks:
            runs = task.get('run_count', 0)
            successes = task.get('success_count', 0)
            success_rate = f"{successes}/{runs}" if runs > 0 else "0/0"
            
            print(f"{task['id']:<20} {task['name']:<25} {task['schedule']:<20} "
                  f"{task['status']:<10} {runs:<6} {success_rate:<7}")
        
        # Show overall status
        status = scheduler.get_task_status()
        print(f"\nTotal: {status['total_tasks']} tasks, "
              f"{status['active_tasks']} active, "
              f"Success rate: {status['success_rate']:.1%}")
    
    elif args.command == 'status':
        if args.task_id:
            status = scheduler.get_task_status(args.task_id)
            if 'error' in status:
                print(status['error'])
            else:
                task = status['task']
                print(f"Task: {task['name']} ({task['id']})")
                print(f"Status: {task['status']}")
                print(f"Schedule: {task['schedule']}")
                print(f"Last run: {task.get('last_run', 'Never')}")
                print(f"Next run: {task.get('next_run', 'Not scheduled')}")
                print(f"Runs: {task.get('run_count', 0)}")
                print(f"Successes: {task.get('success_count', 0)}")
                print(f"Errors: {task.get('error_count', 0)}")
        else:
            status = scheduler.get_task_status()
            print(f"Scheduler Status:")
            print(f"  Running: {status['running']}")
            print(f"  Total tasks: {status['total_tasks']}")
            print(f"  Active tasks: {status['active_tasks']}")
            print(f"  Scheduled jobs: {status['scheduled_jobs']}")
            print(f"  Total runs: {status['total_runs']}")
            print(f"  Total successes: {status['total_successes']}")
            print(f"  Total errors: {status['total_errors']}")
            print(f"  Success rate: {status['success_rate']:.1%}")
    
    elif args.command == 'run':
        if not args.task_id:
            print("Error: --task-id is required for run command")
            return
        
        # Find and run specific task
        for task in scheduler.tasks:
            if task['id'] == args.task_id:
                print(f"Running task: {task['name']}")
                result = scheduler.execute_task(task)
                
                print(f"Execution completed:")
                print(f"  Success: {result['success']}")
                print(f"  Duration: {result['duration']:.2f} seconds")
                
                if result['output']:
                    print(f"  Output: {result['output'][:200]}...")
                
                if result['error']:
                    print(f"  Error: {result['error']}")
                
                return
        
        print(f"Task not found: {args.task_id}")

if __name__ == "__main__":
    main()