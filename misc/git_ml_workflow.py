#!/usr/bin/env python3
"""
Git Workflow Automation for ML Projects

Automates Git operations for machine learning projects including experiment tracking,
model versioning, automated commits, and branch management for ML workflows.

Usage:
    python3 git_ml_workflow.py init --project-name my-ml-project
    python3 git_ml_workflow.py experiment start --name exp-001 --description "Baseline model"
    python3 git_ml_workflow.py commit --auto --include-metrics
    python3 git_ml_workflow.py tag-model --version v1.0.0 --model-path models/best_model.pkl
    python3 git_ml_workflow.py sync-artifacts --remote origin
"""

import argparse
import subprocess
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import yaml
import hashlib
import shutil


class GitMLWorkflow:
    """Git workflow automation specifically designed for ML projects."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.ml_config_file = self.repo_path / ".ml-project.yaml"
        self.experiments_dir = self.repo_path / "experiments"
        self.models_dir = self.repo_path / "models"
        self.artifacts_dir = self.repo_path / "artifacts"
        
    def run_git_command(self, command: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run git command in the repository directory."""
        full_command = ["git", "-C", str(self.repo_path)] + command
        try:
            result = subprocess.run(full_command, capture_output=True, text=True, check=check)
            return result
        except subprocess.CalledProcessError as e:
            print(f"Git command failed: {' '.join(full_command)}")
            print(f"Error: {e.stderr}")
            raise
    
    def init_ml_project(self, project_name: str, remote_url: Optional[str] = None) -> Dict[str, Any]:
        """Initialize a new ML project with Git and ML-specific structure."""
        print(f"🚀 Initializing ML project: {project_name}")
        
        # Initialize git if not already done
        if not (self.repo_path / ".git").exists():
            self.run_git_command(["init"])
            print("✅ Git repository initialized")
        
        # Create ML project structure
        directories = [
            "data/raw",
            "data/processed", 
            "data/external",
            "notebooks",
            "src",
            "models",
            "experiments",
            "artifacts",
            "configs",
            "scripts",
            "tests",
            "docs"
        ]
        
        for dir_path in directories:
            (self.repo_path / dir_path).mkdir(parents=True, exist_ok=True)
            # Create .gitkeep for empty directories
            gitkeep = self.repo_path / dir_path / ".gitkeep"
            if not any((self.repo_path / dir_path).iterdir()):
                gitkeep.touch()
        
        # Create ML project configuration
        ml_config = {
            "project_name": project_name,
            "created_at": datetime.now().isoformat(),
            "ml_framework": "pytorch",  # default
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "experiments": {},
            "models": {},
            "remote_url": remote_url
        }
        
        with open(self.ml_config_file, 'w') as f:
            yaml.dump(ml_config, f, default_flow_style=False, indent=2)
        
        # Create essential files
        self._create_gitignore()
        self._create_readme(project_name)
        self._create_requirements_txt()
        
        # Add remote if provided
        if remote_url:
            try:
                self.run_git_command(["remote", "add", "origin", remote_url])
                print(f"✅ Remote origin added: {remote_url}")
            except subprocess.CalledProcessError:
                print(f"⚠️  Could not add remote (may already exist): {remote_url}")
        
        # Initial commit
        self.run_git_command(["add", "."])
        self.run_git_command(["commit", "-m", f"Initial commit for ML project: {project_name}"])
        
        print("✅ ML project structure created")
        return ml_config
    
    def _create_gitignore(self):
        """Create comprehensive .gitignore for ML projects."""
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/
.venv/

# Jupyter Notebook
.ipynb_checkpoints

# PyCharm
.idea/

# VS Code
.vscode/

# ML/Data Science specific
# Large data files
*.csv
*.tsv
*.parquet
*.h5
*.hdf5
*.pkl
*.pickle
data/raw/*
!data/raw/.gitkeep
data/processed/*
!data/processed/.gitkeep

# Model files (large)
models/*.pt
models/*.pth
models/*.h5
models/*.pkl
models/*.joblib
models/*.onnx
*.model
*.weights

# Experiment outputs
experiments/*/outputs/
experiments/*/logs/
experiments/*/checkpoints/
mlruns/
.mlflow/

# Logs
logs/
*.log

# Temporary files
.tmp/
temp/
*.tmp

# OS
.DS_Store
Thumbs.db

# DVC (Data Version Control)
.dvc/
*.dvc

# Weights & Biases
wandb/

# MLflow
mlruns/

# TensorBoard
runs/
"""
        with open(self.repo_path / ".gitignore", 'w') as f:
            f.write(gitignore_content)
    
    def _create_readme(self, project_name: str):
        """Create README template for ML project."""
        readme_content = f"""# {project_name}

## Project Structure

```
├── data/
│   ├── raw/          # Original, immutable data
│   ├── processed/    # Cleaned and preprocessed data
│   └── external/     # External data sources
├── notebooks/        # Jupyter notebooks for exploration
├── src/             # Source code for the project
├── models/          # Trained models and model artifacts
├── experiments/     # Experiment tracking and results
├── configs/         # Configuration files
├── scripts/         # Utility scripts
├── tests/           # Unit tests
└── docs/            # Documentation

```

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start a new experiment:
   ```bash
   python3 git_ml_workflow.py experiment start --name exp-001
   ```

3. Track your work:
   ```bash
   python3 git_ml_workflow.py commit --auto --include-metrics
   ```

## Experiments

<!-- Experiment tracking will be automatically updated -->

## Models

<!-- Model registry will be automatically updated -->
"""
        with open(self.repo_path / "README.md", 'w') as f:
            f.write(readme_content)
    
    def _create_requirements_txt(self):
        """Create basic requirements.txt for ML projects."""
        requirements = """# Core ML libraries
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0

# Deep Learning (choose one)
# torch>=1.11.0
# tensorflow>=2.8.0

# Data processing
matplotlib>=3.5.0
seaborn>=0.11.0

# Experiment tracking
# mlflow>=1.24.0
# wandb>=0.12.0

# Utilities
pyyaml>=6.0
tqdm>=4.62.0
click>=8.0.0
"""
        with open(self.repo_path / "requirements.txt", 'w') as f:
            f.write(requirements)
    
    def start_experiment(self, name: str, description: str = "", branch: bool = True) -> Dict[str, str]:
        """Start a new ML experiment with optional Git branch."""
        print(f"🧪 Starting experiment: {name}")
        
        # Load ML config
        if not self.ml_config_file.exists():
            raise FileNotFoundError("Not an ML project. Run 'init' first.")
        
        with open(self.ml_config_file, 'r') as f:
            ml_config = yaml.safe_load(f)
        
        # Create experiment directory
        exp_dir = self.experiments_dir / name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create experiment subdirectories
        for subdir in ["configs", "outputs", "logs", "notebooks"]:
            (exp_dir / subdir).mkdir(exist_ok=True)
        
        # Create experiment metadata
        exp_metadata = {
            "name": name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "status": "running",
            "branch": f"experiment/{name}" if branch else None,
            "metrics": {},
            "artifacts": []
        }
        
        # Save experiment metadata
        with open(exp_dir / "metadata.yaml", 'w') as f:
            yaml.dump(exp_metadata, f, default_flow_style=False, indent=2)
        
        # Update ML config
        ml_config["experiments"][name] = exp_metadata
        with open(self.ml_config_file, 'w') as f:
            yaml.dump(ml_config, f, default_flow_style=False, indent=2)
        
        # Create experiment branch
        if branch:
            try:
                current_branch = self.run_git_command(["branch", "--show-current"]).stdout.strip()
                branch_name = f"experiment/{name}"
                self.run_git_command(["checkout", "-b", branch_name])
                exp_metadata["parent_branch"] = current_branch
                print(f"✅ Created and switched to branch: {branch_name}")
            except subprocess.CalledProcessError:
                print("⚠️  Could not create experiment branch")
        
        # Commit experiment setup
        self.run_git_command(["add", str(exp_dir), str(self.ml_config_file)])
        self.run_git_command(["commit", "-m", f"Start experiment: {name}"])
        
        print(f"✅ Experiment {name} started")
        return exp_metadata
    
    def auto_commit(self, include_metrics: bool = True, message: Optional[str] = None) -> str:
        """Automatically commit changes with ML-aware commit message."""
        # Check if there are changes to commit
        status = self.run_git_command(["status", "--porcelain"]).stdout.strip()
        if not status:
            print("No changes to commit")
            return "No changes"
        
        # Generate commit message if not provided
        if not message:
            message = self._generate_commit_message(include_metrics)
        
        # Add all tracked files and new files in specific directories
        ml_dirs = ["src", "configs", "experiments", "notebooks", "scripts"]
        for dir_name in ml_dirs:
            dir_path = self.repo_path / dir_name
            if dir_path.exists():
                self.run_git_command(["add", str(dir_path)])
        
        # Add other important files
        important_files = [".ml-project.yaml", "requirements.txt", "README.md"]
        for file_name in important_files:
            file_path = self.repo_path / file_name
            if file_path.exists():
                self.run_git_command(["add", str(file_path)])
        
        # Commit
        self.run_git_command(["commit", "-m", message])
        commit_hash = self.run_git_command(["rev-parse", "HEAD"]).stdout.strip()[:8]
        
        print(f"✅ Committed changes: {commit_hash}")
        print(f"📝 Message: {message}")
        
        return commit_hash
    
    def _generate_commit_message(self, include_metrics: bool = True) -> str:
        """Generate intelligent commit message based on changes."""
        # Get changed files
        changed_files = self.run_git_command(["diff", "--cached", "--name-only"]).stdout.strip().split('\n')
        changed_files = [f for f in changed_files if f]
        
        # Categorize changes
        categories = {
            "config": ["configs/", ".yaml", ".json"],
            "model": ["models/", ".pkl", ".pt", ".pth", ".h5"],
            "data": ["data/", ".csv", ".parquet"],
            "notebook": [".ipynb"],
            "code": [".py"],
            "experiment": ["experiments/"]
        }
        
        change_types = []
        for category, patterns in categories.items():
            if any(any(pattern in f for pattern in patterns) for f in changed_files):
                change_types.append(category)
        
        # Generate message
        if not change_types:
            return "Update project files"
        
        primary_type = change_types[0]
        
        if primary_type == "experiment":
            return "Update experiment tracking and results"
        elif primary_type == "model":
            return "Update model artifacts and weights"
        elif primary_type == "config":
            return "Update configuration files"
        elif primary_type == "code":
            return "Update source code and scripts"
        elif primary_type == "notebook":
            return "Update analysis notebooks"
        elif primary_type == "data":
            return "Update data processing and datasets"
        else:
            return f"Update {', '.join(change_types)} components"
    
    def tag_model(self, version: str, model_path: str, description: str = "") -> Dict[str, str]:
        """Tag a model version in Git and update model registry."""
        print(f"🏷️  Tagging model version: {version}")
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Calculate model hash for integrity
        model_hash = self._calculate_file_hash(model_path)
        
        # Create tag
        tag_name = f"model-{version}"
        tag_message = f"Model {version}: {description}" if description else f"Model {version}"
        
        try:
            self.run_git_command(["tag", "-a", tag_name, "-m", tag_message])
            print(f"✅ Created tag: {tag_name}")
        except subprocess.CalledProcessError:
            print(f"⚠️  Tag {tag_name} may already exist")
        
        # Update ML config with model info
        if self.ml_config_file.exists():
            with open(self.ml_config_file, 'r') as f:
                ml_config = yaml.safe_load(f)
            
            ml_config.setdefault("models", {})[version] = {
                "path": str(model_path),
                "hash": model_hash,
                "tag": tag_name,
                "description": description,
                "created_at": datetime.now().isoformat(),
                "size_mb": model_path.stat().st_size / 1024 / 1024
            }
            
            with open(self.ml_config_file, 'w') as f:
                yaml.dump(ml_config, f, default_flow_style=False, indent=2)
        
        return {
            "version": version,
            "tag": tag_name,
            "hash": model_hash,
            "path": str(model_path)
        }
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()[:16]  # First 16 characters
    
    def sync_artifacts(self, remote: str = "origin", force: bool = False) -> bool:
        """Sync code changes while respecting large file handling."""
        print(f"🔄 Syncing with remote: {remote}")
        
        try:
            # Fetch latest changes
            self.run_git_command(["fetch", remote])
            
            # Push current branch
            current_branch = self.run_git_command(["branch", "--show-current"]).stdout.strip()
            if current_branch:
                push_cmd = ["push", remote, current_branch]
                if force:
                    push_cmd.append("--force")
                
                self.run_git_command(push_cmd)
                print(f"✅ Pushed branch: {current_branch}")
            
            # Push tags
            self.run_git_command(["push", remote, "--tags"])
            print("✅ Pushed tags")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Sync failed: {e}")
            return False
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments in the project."""
        if not self.ml_config_file.exists():
            return []
        
        with open(self.ml_config_file, 'r') as f:
            ml_config = yaml.safe_load(f)
        
        experiments = ml_config.get("experiments", {})
        return list(experiments.values())
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models."""
        if not self.ml_config_file.exists():
            return []
        
        with open(self.ml_config_file, 'r') as f:
            ml_config = yaml.safe_load(f)
        
        models = ml_config.get("models", {})
        return list(models.values())
    
    def cleanup_experiments(self, days_old: int = 30) -> List[str]:
        """Clean up old experiment branches."""
        print(f"🧹 Cleaning up experiments older than {days_old} days")
        
        # Get all branches
        branches_output = self.run_git_command(["branch", "-a"]).stdout
        experiment_branches = [
            line.strip().replace("* ", "").replace("origin/", "")
            for line in branches_output.split('\n')
            if 'experiment/' in line and 'origin/' not in line
        ]
        
        cleaned_branches = []
        cutoff_date = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        
        for branch in experiment_branches:
            try:
                # Get branch creation date
                commit_date = self.run_git_command([
                    "log", "-1", "--format=%ct", branch
                ]).stdout.strip()
                
                if commit_date and int(commit_date) < cutoff_date:
                    # Delete branch
                    self.run_git_command(["branch", "-D", branch])
                    cleaned_branches.append(branch)
                    print(f"🗑️  Deleted old branch: {branch}")
                    
            except subprocess.CalledProcessError:
                continue
        
        return cleaned_branches


def main():
    parser = argparse.ArgumentParser(description="Git workflow automation for ML projects")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Init command
    init_parser = subparsers.add_parser('init', help='Initialize ML project')
    init_parser.add_argument('--project-name', required=True, help='Project name')
    init_parser.add_argument('--remote-url', help='Git remote URL')

    # Experiment commands
    exp_parser = subparsers.add_parser('experiment', help='Experiment management')
    exp_subparsers = exp_parser.add_subparsers(dest='exp_command')
    
    start_exp_parser = exp_subparsers.add_parser('start', help='Start new experiment')
    start_exp_parser.add_argument('--name', required=True, help='Experiment name')
    start_exp_parser.add_argument('--description', default='', help='Experiment description')
    start_exp_parser.add_argument('--no-branch', action='store_true', help='Skip creating branch')
    
    list_exp_parser = exp_subparsers.add_parser('list', help='List experiments')
    
    # Commit command
    commit_parser = subparsers.add_parser('commit', help='Auto-commit changes')
    commit_parser.add_argument('--auto', action='store_true', help='Generate commit message automatically')
    commit_parser.add_argument('--message', '-m', help='Custom commit message')
    commit_parser.add_argument('--include-metrics', action='store_true', help='Include metrics in commit')

    # Model commands
    model_parser = subparsers.add_parser('tag-model', help='Tag model version')
    model_parser.add_argument('--version', required=True, help='Model version')
    model_parser.add_argument('--model-path', required=True, help='Path to model file')
    model_parser.add_argument('--description', default='', help='Model description')

    list_models_parser = subparsers.add_parser('list-models', help='List registered models')

    # Sync command
    sync_parser = subparsers.add_parser('sync-artifacts', help='Sync with remote')
    sync_parser.add_argument('--remote', default='origin', help='Remote name')
    sync_parser.add_argument('--force', action='store_true', help='Force push')

    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old experiments')
    cleanup_parser.add_argument('--days', type=int, default=30, help='Days old to clean up')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    git_ml = GitMLWorkflow()

    try:
        if args.command == 'init':
            git_ml.init_ml_project(args.project_name, args.remote_url)

        elif args.command == 'experiment':
            if args.exp_command == 'start':
                git_ml.start_experiment(args.name, args.description, not args.no_branch)
            elif args.exp_command == 'list':
                experiments = git_ml.list_experiments()
                if experiments:
                    print("📊 Experiments:")
                    for exp in experiments:
                        status = exp.get('status', 'unknown')
                        print(f"  {exp['name']}: {status} ({exp.get('created_at', 'unknown')})")
                        if exp.get('description'):
                            print(f"    {exp['description']}")
                else:
                    print("No experiments found")

        elif args.command == 'commit':
            if args.auto or not args.message:
                git_ml.auto_commit(args.include_metrics, args.message)
            else:
                git_ml.auto_commit(False, args.message)

        elif args.command == 'tag-model':
            result = git_ml.tag_model(args.version, args.model_path, args.description)
            print(f"✅ Model tagged: {result['version']} (hash: {result['hash']})")

        elif args.command == 'list-models':
            models = git_ml.list_models()
            if models:
                print("🤖 Registered Models:")
                for model in models:
                    size_mb = model.get('size_mb', 0)
                    print(f"  {model.get('tag', 'unknown')}: {model.get('path', 'unknown')} ({size_mb:.1f} MB)")
                    if model.get('description'):
                        print(f"    {model['description']}")
            else:
                print("No models registered")

        elif args.command == 'sync-artifacts':
            success = git_ml.sync_artifacts(args.remote, args.force)
            if not success:
                sys.exit(1)

        elif args.command == 'cleanup':
            cleaned = git_ml.cleanup_experiments(args.days)
            if cleaned:
                print(f"✅ Cleaned up {len(cleaned)} old experiment branches")
            else:
                print("No old experiments to clean up")

    except Exception as e:
        print(f" Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()