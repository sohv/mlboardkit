#!/usr/bin/env python3
"""
workflow_engine.py

Execute complex workflows with dependencies and error handling.
"""

import argparse
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import threading
import logging
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"

class WorkflowEngine:
    def __init__(self):
        self.workflows = {}
        self.execution_history = []
    
    def load_workflow(self, workflow_file: str) -> str:
        """Load workflow from JSON file"""
        with open(workflow_file, 'r', encoding='utf-8') as f:
            workflow = json.load(f)
        
        # Validate workflow
        self.validate_workflow(workflow)
        
        workflow_id = workflow['id']
        self.workflows[workflow_id] = workflow
        
        logger.info(f"Loaded workflow: {workflow['name']} ({workflow_id})")
        return workflow_id
    
    def validate_workflow(self, workflow: Dict[str, Any]):
        """Validate workflow structure"""
        required_fields = ['id', 'name', 'steps']
        for field in required_fields:
            if field not in workflow:
                raise ValueError(f"Workflow missing required field: {field}")
        
        # Validate steps
        for i, step in enumerate(workflow['steps']):
            step_required = ['id', 'name', 'action']
            for field in step_required:
                if field not in step:
                    raise ValueError(f"Step {i} missing required field: {field}")
        
        # Check for circular dependencies
        self.check_circular_dependencies(workflow['steps'])
    
    def check_circular_dependencies(self, steps: List[Dict[str, Any]]):
        """Check for circular dependencies in workflow steps"""
        step_ids = {step['id'] for step in steps}
        
        def has_cycle(step_id: str, visited: Set[str], path: Set[str]) -> bool:
            if step_id in path:
                return True
            if step_id in visited:
                return False
            
            visited.add(step_id)
            path.add(step_id)
            
            # Find step and check dependencies
            for step in steps:
                if step['id'] == step_id:
                    dependencies = step.get('depends_on', [])
                    for dep in dependencies:
                        if dep in step_ids and has_cycle(dep, visited, path):
                            return True
                    break
            
            path.remove(step_id)
            return False
        
        visited = set()
        for step in steps:
            if step['id'] not in visited:
                if has_cycle(step['id'], visited, set()):
                    raise ValueError(f"Circular dependency detected involving step: {step['id']}")
    
    def get_execution_order(self, steps: List[Dict[str, Any]]) -> List[str]:
        """Determine execution order based on dependencies"""
        
        # Build dependency graph
        step_map = {step['id']: step for step in steps}
        in_degree = {step['id']: 0 for step in steps}
        
        # Calculate in-degrees
        for step in steps:
            dependencies = step.get('depends_on', [])
            in_degree[step['id']] = len(dependencies)
        
        # Topological sort
        queue = [step_id for step_id, degree in in_degree.items() if degree == 0]
        execution_order = []
        
        while queue:
            current = queue.pop(0)
            execution_order.append(current)
            
            # Update in-degrees of dependent steps
            for step in steps:
                dependencies = step.get('depends_on', [])
                if current in dependencies:
                    in_degree[step['id']] -= 1
                    if in_degree[step['id']] == 0:
                        queue.append(step['id'])
        
        if len(execution_order) != len(steps):
            raise ValueError("Cannot determine execution order - possible circular dependencies")
        
        return execution_order
    
    def execute_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow step"""
        
        step_id = step['id']
        step_name = step['name']
        action = step['action']
        
        logger.info(f"Executing step: {step_name} ({step_id})")
        
        start_time = datetime.now()
        result = {
            'step_id': step_id,
            'step_name': step_name,
            'status': StepStatus.RUNNING.value,
            'start_time': start_time.isoformat(),
            'output': '',
            'error': '',
            'duration': 0
        }
        
        try:
            if action['type'] == 'command':
                result.update(self.execute_command_action(action, context))
            
            elif action['type'] == 'script':
                result.update(self.execute_script_action(action, context))
            
            elif action['type'] == 'http':
                result.update(self.execute_http_action(action, context))
            
            elif action['type'] == 'python':
                result.update(self.execute_python_action(action, context))
            
            elif action['type'] == 'condition':
                result.update(self.execute_condition_action(action, context))
            
            else:
                raise ValueError(f"Unknown action type: {action['type']}")
            
            # Check if step should be considered successful
            if result.get('success', False):
                result['status'] = StepStatus.SUCCESS.value
            else:
                result['status'] = StepStatus.FAILED.value
        
        except Exception as e:
            result['status'] = StepStatus.FAILED.value
            result['error'] = str(e)
            logger.error(f"Step failed: {step_name} - {e}")
        
        finally:
            end_time = datetime.now()
            result['end_time'] = end_time.isoformat()
            result['duration'] = (end_time - start_time).total_seconds()
        
        # Update context with step results
        context[f'step_{step_id}'] = result
        
        return result
    
    def execute_command_action(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute command action"""
        command = action['command']
        working_dir = action.get('working_directory')
        timeout = action.get('timeout', 300)
        
        # Substitute variables in command
        command = self.substitute_variables(command, context)
        
        try:
            if isinstance(command, str):
                process = subprocess.run(
                    command,
                    shell=True,
                    cwd=working_dir,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
            else:
                process = subprocess.run(
                    command,
                    cwd=working_dir,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
            
            return {
                'success': process.returncode == 0,
                'output': process.stdout,
                'error': process.stderr,
                'return_code': process.returncode
            }
        
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': f"Command timed out after {timeout} seconds"
            }
    
    def execute_script_action(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute script action"""
        script_path = action['script']
        args = action.get('args', [])
        working_dir = action.get('working_directory')
        timeout = action.get('timeout', 300)
        
        # Substitute variables in args
        args = [self.substitute_variables(arg, context) for arg in args]
        
        command = [script_path] + args
        
        try:
            process = subprocess.run(
                command,
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return {
                'success': process.returncode == 0,
                'output': process.stdout,
                'error': process.stderr,
                'return_code': process.returncode
            }
        
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': f"Script timed out after {timeout} seconds"
            }
    
    def execute_http_action(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute HTTP action"""
        try:
            import requests
            
            url = self.substitute_variables(action['url'], context)
            method = action.get('method', 'GET').upper()
            headers = action.get('headers', {})
            data = action.get('data')
            timeout = action.get('timeout', 30)
            
            # Substitute variables in headers and data
            headers = {k: self.substitute_variables(v, context) for k, v in headers.items()}
            if data:
                data = self.substitute_variables(data, context)
            
            response = requests.request(
                method=method,
                url=url,
                headers=headers,
                json=data if isinstance(data, dict) else None,
                data=data if isinstance(data, str) else None,
                timeout=timeout
            )
            
            return {
                'success': 200 <= response.status_code < 300,
                'output': response.text,
                'status_code': response.status_code,
                'headers': dict(response.headers)
            }
        
        except ImportError:
            return {
                'success': False,
                'error': "requests library not available for HTTP actions"
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def execute_python_action(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Python code action"""
        code = action['code']
        
        # Substitute variables in code
        code = self.substitute_variables(code, context)
        
        try:
            # Create execution environment
            exec_globals = {'context': context}
            exec_locals = {}
            
            # Execute code
            exec(code, exec_globals, exec_locals)
            
            # Get result
            result_value = exec_locals.get('result', 'Code executed successfully')
            success = exec_locals.get('success', True)
            
            return {
                'success': success,
                'output': str(result_value)
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def execute_condition_action(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute condition action"""
        condition = action['condition']
        
        # Substitute variables in condition
        condition = self.substitute_variables(condition, context)
        
        try:
            # Evaluate condition
            result = eval(condition, {"__builtins__": {}}, context)
            
            return {
                'success': True,
                'output': f"Condition result: {result}",
                'condition_result': bool(result)
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': f"Condition evaluation failed: {e}"
            }
    
    def substitute_variables(self, text: str, context: Dict[str, Any]) -> str:
        """Substitute variables in text using context"""
        if not isinstance(text, str):
            return text
        
        # Simple variable substitution: ${variable_name}
        import re
        
        def replacer(match):
            var_name = match.group(1)
            return str(context.get(var_name, match.group(0)))
        
        return re.sub(r'\$\{([^}]+)\}', replacer, text)
    
    def should_skip_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check if step should be skipped based on conditions"""
        
        # Check if condition
        if_condition = step.get('if')
        if if_condition:
            try:
                condition = self.substitute_variables(if_condition, context)
                should_run = eval(condition, {"__builtins__": {}}, context)
                return not should_run
            except Exception as e:
                logger.warning(f"Error evaluating if condition for step {step['id']}: {e}")
                return False
        
        return False
    
    def execute_workflow(self, workflow_id: str, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a workflow"""
        
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        workflow = self.workflows[workflow_id]
        inputs = inputs or {}
        
        logger.info(f"Starting workflow execution: {workflow['name']} ({workflow_id})")
        
        start_time = datetime.now()
        execution_id = f"{workflow_id}_{int(start_time.timestamp())}"
        
        # Initialize execution context
        context = {
            'workflow_id': workflow_id,
            'execution_id': execution_id,
            'start_time': start_time.isoformat(),
            **inputs
        }
        
        execution_result = {
            'execution_id': execution_id,
            'workflow_id': workflow_id,
            'workflow_name': workflow['name'],
            'start_time': start_time.isoformat(),
            'status': 'running',
            'steps': [],
            'context': context
        }
        
        try:
            # Get execution order
            execution_order = self.get_execution_order(workflow['steps'])
            step_map = {step['id']: step for step in workflow['steps']}
            
            # Execute steps in order
            for step_id in execution_order:
                step = step_map[step_id]
                
                # Check dependencies
                dependencies = step.get('depends_on', [])
                failed_dependencies = []
                
                for dep_id in dependencies:
                    dep_result = context.get(f'step_{dep_id}')
                    if not dep_result or dep_result['status'] != StepStatus.SUCCESS.value:
                        failed_dependencies.append(dep_id)
                
                if failed_dependencies:
                    # Skip step due to failed dependencies
                    step_result = {
                        'step_id': step_id,
                        'step_name': step['name'],
                        'status': StepStatus.SKIPPED.value,
                        'start_time': datetime.now().isoformat(),
                        'error': f"Skipped due to failed dependencies: {failed_dependencies}",
                        'duration': 0
                    }
                    context[f'step_{step_id}'] = step_result
                    execution_result['steps'].append(step_result)
                    continue
                
                # Check if step should be skipped based on conditions
                if self.should_skip_step(step, context):
                    step_result = {
                        'step_id': step_id,
                        'step_name': step['name'],
                        'status': StepStatus.SKIPPED.value,
                        'start_time': datetime.now().isoformat(),
                        'error': 'Skipped due to condition',
                        'duration': 0
                    }
                    context[f'step_{step_id}'] = step_result
                    execution_result['steps'].append(step_result)
                    continue
                
                # Execute step
                step_result = self.execute_step(step, context)
                execution_result['steps'].append(step_result)
                
                # Check if we should stop on failure
                if (step_result['status'] == StepStatus.FAILED.value and 
                    step.get('continue_on_failure', False) == False):
                    logger.error(f"Workflow stopped due to step failure: {step['name']}")
                    break
            
            # Determine overall workflow status
            step_statuses = [step['status'] for step in execution_result['steps']]
            
            if any(status == StepStatus.FAILED.value for status in step_statuses):
                execution_result['status'] = 'failed'
            elif any(status == StepStatus.RUNNING.value for status in step_statuses):
                execution_result['status'] = 'running'
            else:
                execution_result['status'] = 'completed'
        
        except Exception as e:
            execution_result['status'] = 'error'
            execution_result['error'] = str(e)
            logger.error(f"Workflow execution error: {e}")
        
        finally:
            end_time = datetime.now()
            execution_result['end_time'] = end_time.isoformat()
            execution_result['duration'] = (end_time - start_time).total_seconds()
            
            # Save execution history
            self.execution_history.append(execution_result)
            
            logger.info(f"Workflow execution completed: {execution_result['status']} "
                       f"({execution_result['duration']:.2f}s)")
        
        return execution_result

def create_sample_workflow() -> Dict[str, Any]:
    """Create a sample workflow for demonstration"""
    return {
        "id": "data_processing_pipeline",
        "name": "Data Processing Pipeline",
        "description": "Process data files with validation and cleanup",
        "version": "1.0",
        "steps": [
            {
                "id": "validate_input",
                "name": "Validate Input Files",
                "action": {
                    "type": "command",
                    "command": "ls -la ${input_directory}",
                    "timeout": 30
                },
                "continue_on_failure": False
            },
            {
                "id": "create_backup",
                "name": "Create Backup",
                "depends_on": ["validate_input"],
                "action": {
                    "type": "command",
                    "command": "cp -r ${input_directory} ${backup_directory}",
                    "timeout": 300
                }
            },
            {
                "id": "process_data",
                "name": "Process Data Files",
                "depends_on": ["validate_input"],
                "action": {
                    "type": "script",
                    "script": "python3",
                    "args": ["process_data.py", "${input_directory}", "${output_directory}"],
                    "timeout": 600
                }
            },
            {
                "id": "validate_output",
                "name": "Validate Output",
                "depends_on": ["process_data"],
                "action": {
                    "type": "python",
                    "code": """
import os
output_dir = context.get('output_directory', '/tmp/output')
if os.path.exists(output_dir) and os.listdir(output_dir):
    result = f"Output directory contains {len(os.listdir(output_dir))} files"
    success = True
else:
    result = "Output directory is empty or missing"
    success = False
"""
                }
            },
            {
                "id": "cleanup",
                "name": "Cleanup Temporary Files",
                "depends_on": ["process_data"],
                "action": {
                    "type": "command",
                    "command": "rm -rf /tmp/temp_*",
                    "timeout": 60
                },
                "continue_on_failure": True
            },
            {
                "id": "send_notification",
                "name": "Send Completion Notification",
                "depends_on": ["validate_output"],
                "if": "${notify_on_completion}",
                "action": {
                    "type": "http",
                    "url": "${notification_url}",
                    "method": "POST",
                    "headers": {
                        "Content-Type": "application/json"
                    },
                    "data": {
                        "message": "Data processing pipeline completed successfully",
                        "execution_id": "${execution_id}"
                    },
                    "timeout": 30
                }
            }
        ]
    }

def main():
    parser = argparse.ArgumentParser(description="Workflow execution engine")
    parser.add_argument('command', choices=['load', 'execute', 'list', 'history'],
                       help='Command to execute')
    parser.add_argument('--workflow', help='Workflow file or ID')
    parser.add_argument('--inputs', help='JSON file with workflow inputs')
    parser.add_argument('--input-dir', help='Input directory for sample workflow')
    parser.add_argument('--output-dir', help='Output directory for sample workflow')
    parser.add_argument('--backup-dir', help='Backup directory for sample workflow')
    parser.add_argument('--sample', action='store_true', help='Create sample workflow')
    parser.add_argument('--save', help='Save workflow to file')
    
    args = parser.parse_args()
    
    engine = WorkflowEngine()
    
    if args.command == 'load':
        if args.sample:
            workflow = create_sample_workflow()
            if args.save:
                with open(args.save, 'w', encoding='utf-8') as f:
                    json.dump(workflow, f, indent=2, ensure_ascii=False)
                print(f"Sample workflow saved to {args.save}")
            else:
                workflow_id = workflow['id']
                engine.workflows[workflow_id] = workflow
                print(f"Loaded sample workflow: {workflow['name']} ({workflow_id})")
        elif args.workflow:
            workflow_id = engine.load_workflow(args.workflow)
            print(f"Loaded workflow: {workflow_id}")
        else:
            print("Error: --workflow or --sample required for load command")
    
    elif args.command == 'execute':
        if not args.workflow:
            print("Error: --workflow required for execute command")
            return
        
        # Load inputs
        inputs = {}
        if args.inputs:
            with open(args.inputs, 'r', encoding='utf-8') as f:
                inputs = json.load(f)
        
        # Add command line inputs for sample workflow
        if args.input_dir:
            inputs['input_directory'] = args.input_dir
        if args.output_dir:
            inputs['output_directory'] = args.output_dir
        if args.backup_dir:
            inputs['backup_directory'] = args.backup_dir
        
        # Set default notification
        inputs.setdefault('notify_on_completion', False)
        
        # Load workflow if it's a file
        if args.workflow.endswith('.json'):
            workflow_id = engine.load_workflow(args.workflow)
        else:
            workflow_id = args.workflow
        
        # Execute workflow
        result = engine.execute_workflow(workflow_id, inputs)
        
        print(f"Workflow Execution Result:")
        print(f"  Execution ID: {result['execution_id']}")
        print(f"  Status: {result['status']}")
        print(f"  Duration: {result['duration']:.2f} seconds")
        print(f"  Steps executed: {len(result['steps'])}")
        
        # Show step results
        for step in result['steps']:
            status_icon = "✓" if step['status'] == 'success' else "✗" if step['status'] == 'failed' else "⏭"
            print(f"    {status_icon} {step['step_name']} ({step['status']})")
            if step.get('error'):
                print(f"      Error: {step['error']}")
    
    elif args.command == 'list':
        workflows = engine.workflows
        
        if not workflows:
            print("No workflows loaded")
            return
        
        print(f"{'ID':<30} {'Name':<40} {'Steps':<6}")
        print("-" * 76)
        
        for workflow_id, workflow in workflows.items():
            print(f"{workflow_id:<30} {workflow['name']:<40} {len(workflow['steps']):<6}")
    
    elif args.command == 'history':
        history = engine.execution_history
        
        if not history:
            print("No execution history")
            return
        
        print(f"{'Execution ID':<25} {'Workflow':<30} {'Status':<10} {'Duration':<10}")
        print("-" * 75)
        
        for execution in history:
            duration = f"{execution['duration']:.1f}s"
            print(f"{execution['execution_id']:<25} {execution['workflow_name']:<30} "
                  f"{execution['status']:<10} {duration:<10}")

if __name__ == "__main__":
    main()