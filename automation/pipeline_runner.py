#!/usr/bin/env python3
"""
pipeline_runner.py

Compose and run multi-stage ML pipelines (data → model → eval).
"""

import argparse
import json
import subprocess
import shlex
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


class PipelineRunner:
    def __init__(self, config_file: str):
        self.config = self.load_config(config_file)
        self.results = {}
    
    def load_config(self, config_file: str) -> Dict[str, Any]:
        """Load pipeline configuration."""
        with open(config_file, 'r') as f:
            return json.load(f)
    
    def run_command(self, command: str, stage_name: str) -> Dict[str, Any]:
        """Run a command and capture results."""
        print(f"\\n[{stage_name}] Running: {command}")
        start_time = time.time()
        
        try:
            result = subprocess.run(
                shlex.split(command), 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            duration = time.time() - start_time
            
            return {
                'status': 'success',
                'duration': duration,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
            
        except subprocess.CalledProcessError as e:
            duration = time.time() - start_time
            
            return {
                'status': 'failed',
                'duration': duration,
                'stdout': e.stdout,
                'stderr': e.stderr,
                'returncode': e.returncode,
                'error': str(e)
            }
    
    def run_stage(self, stage: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single pipeline stage."""
        stage_name = stage.get('name', 'unnamed')
        commands = stage.get('commands', [])
        
        if isinstance(commands, str):
            commands = [commands]
        
        stage_results = {
            'name': stage_name,
            'start_time': datetime.now().isoformat(),
            'commands': []
        }
        
        for cmd in commands:
            cmd_result = self.run_command(cmd, stage_name)
            stage_results['commands'].append({
                'command': cmd,
                'result': cmd_result
            })
            
            # Stop stage if command failed and fail_fast is enabled
            if cmd_result['status'] == 'failed' and stage.get('fail_fast', True):
                stage_results['status'] = 'failed'
                return stage_results
        
        stage_results['status'] = 'success'
        stage_results['end_time'] = datetime.now().isoformat()
        return stage_results
    
    def run_pipeline(self) -> Dict[str, Any]:
        """Run the complete pipeline."""
        pipeline_results = {
            'pipeline_name': self.config.get('name', 'unnamed'),
            'start_time': datetime.now().isoformat(),
            'stages': []
        }
        
        stages = self.config.get('stages', [])
        
        for stage in stages:
            print(f"\\n{'='*50}")
            print(f"Starting stage: {stage.get('name', 'unnamed')}")
            print(f"{'='*50}")
            
            stage_result = self.run_stage(stage)
            pipeline_results['stages'].append(stage_result)
            
            # Stop pipeline if stage failed and global fail_fast is enabled
            if (stage_result['status'] == 'failed' and 
                self.config.get('fail_fast', True)):
                pipeline_results['status'] = 'failed'
                break
        else:
            pipeline_results['status'] = 'success'
        
        pipeline_results['end_time'] = datetime.now().isoformat()
        return pipeline_results
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save pipeline results to file."""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\\nPipeline results saved to: {output_file}")


def create_sample_config():
    """Create a sample pipeline configuration."""
    sample = {
        "name": "ML Training Pipeline",
        "fail_fast": True,
        "stages": [
            {
                "name": "data_prep",
                "commands": [
                    "python data_converter.py convert raw_data.json clean_data.csv",
                    "python dataset_processor.py quality-check clean_data.csv --report quality.json"
                ],
                "fail_fast": True
            },
            {
                "name": "training",
                "commands": [
                    "python train_model.py --data clean_data.csv --output model.pkl"
                ]
            },
            {
                "name": "evaluation",
                "commands": [
                    "python evaluate_model.py --model model.pkl --test test_data.csv --output results.json"
                ]
            }
        ]
    }
    
    with open('sample_pipeline.json', 'w') as f:
        json.dump(sample, f, indent=2)
    
    print("Sample pipeline configuration saved to: sample_pipeline.json")


def main():
    parser = argparse.ArgumentParser(description='Run ML pipelines')
    parser.add_argument('--config', help='Pipeline configuration JSON file')
    parser.add_argument('--output', default='pipeline_results.json', 
                       help='Results output file')
    parser.add_argument('--create-sample', action='store_true', 
                       help='Create sample pipeline config')
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_config()
        return
    
    if not args.config:
        print("Error: --config required (or use --create-sample)")
        return
    
    # Run pipeline
    runner = PipelineRunner(args.config)
    results = runner.run_pipeline()
    
    # Save results
    runner.save_results(results, args.output)
    
    # Print summary
    print(f"\\n{'='*50}")
    print(f"Pipeline Status: {results['status'].upper()}")
    print(f"Total Stages: {len(results['stages'])}")
    
    for stage in results['stages']:
        status_icon = "✅" if stage['status'] == 'success' else "❌"
        print(f"  {status_icon} {stage['name']}: {stage['status']}")


if __name__ == '__main__':
    main()