#!/usr/bin/env python3
"""
Project Organization and File Management Utility

Automates project scaffolding, file organization, artifact management,
and workspace cleanup for ML/AI projects and general development workflows.

Usage:
    python3 project_organizer.py scaffold --template ml-research --name my-project
    python3 project_organizer.py organize --directory ./messy_project --dry-run
    python3 project_organizer.py archive --project ./old_project --compression zip
    python3 project_organizer.py dedupe --directory ./data --strategy hash
    python3 project_organizer.py backup --source ./important_files --destination ./backups
"""

import argparse
import os
import shutil
import json
import yaml
from pathlib import Path
import sys
from typing import Dict, List, Optional, Any, Set
import hashlib
import zipfile
import tarfile
from datetime import datetime
import tempfile
import fnmatch
import re


class ProjectScaffolder:
    """Create project structures from templates."""
    
    TEMPLATES = {
        "ml-research": {
            "description": "Machine Learning Research Project",
            "directories": [
                "data/raw",
                "data/processed",
                "data/external",
                "src/models",
                "src/data",
                "src/features",
                "src/visualization",
                "notebooks/exploratory",
                "notebooks/modeling",
                "experiments",
                "models/trained",
                "models/checkpoints",
                "configs",
                "scripts",
                "tests",
                "docs",
                "reports/figures",
                "references"
            ],
            "files": {
                "README.md": "# {project_name}\n\n## Project Structure\n\nThis project follows the ML research template structure.\n",
                "requirements.txt": "# Core ML Libraries\nnumpy>=1.21.0\npandas>=1.3.0\nscikit-learn>=1.0.0\nmatplotlib>=3.5.0\njupyter>=1.0.0\n",
                ".gitignore": "# Python\n__pycache__/\n*.pyc\n*.pyo\n\n# Data\ndata/raw/*\n!data/raw/.gitkeep\n\n# Models\nmodels/trained/*\n!models/trained/.gitkeep\n\n# Notebooks\n.ipynb_checkpoints/\n",
                "src/__init__.py": "",
                "src/models/__init__.py": "",
                "src/data/__init__.py": "",
                "Makefile": "# Project Makefile\n\ninstall:\n\tpip install -r requirements.txt\n\ndata:\n\t# Add data download commands\n\ntrain:\n\t# Add training commands\n\ntest:\n\tpytest tests/\n"
            }
        },
        
        "web-app": {
            "description": "Web Application Project",
            "directories": [
                "src/components",
                "src/pages",
                "src/utils",
                "src/styles",
                "src/assets",
                "public",
                "tests/unit",
                "tests/integration",
                "docs",
                "scripts",
                "config"
            ],
            "files": {
                "README.md": "# {project_name}\n\n## Web Application\n\nA modern web application built with best practices.\n",
                "package.json": '{\n  "name": "{project_name}",\n  "version": "1.0.0",\n  "description": "",\n  "main": "index.js",\n  "scripts": {\n    "start": "node src/index.js",\n    "test": "jest",\n    "build": "webpack --mode production"\n  }\n}',
                ".gitignore": "node_modules/\nbuild/\ndist/\n.env\n*.log\n",
                "src/index.js": "// Main application entry point\nconsole.log('Hello, {project_name}!');\n"
            }
        },
        
        "python-package": {
            "description": "Python Package/Library",
            "directories": [
                "src/{project_name}",
                "tests",
                "docs/source",
                "examples",
                "scripts"
            ],
            "files": {
                "README.md": "# {project_name}\n\nA Python package for...\n\n## Installation\n\n```bash\npip install {project_name}\n```\n",
                "setup.py": "from setuptools import setup, find_packages\n\nsetup(\n    name='{project_name}',\n    version='0.1.0',\n    packages=find_packages(where='src'),\n    package_dir={{'': 'src'}},\n    install_requires=[],\n    python_requires='>=3.7'\n)",
                "pyproject.toml": "[build-system]\nrequires = ['setuptools>=45', 'wheel']\nbuild-backend = 'setuptools.build_meta'\n",
                "src/{project_name}/__init__.py": "__version__ = '0.1.0'\n",
                "requirements-dev.txt": "pytest>=6.0.0\nblack>=21.0.0\nflake8>=3.9.0\nmypy>=0.812\n",
                ".gitignore": "__pycache__/\n*.pyc\nbuild/\ndist/\n*.egg-info/\n.pytest_cache/\n.mypy_cache/\n"
            }
        },
        
        "data-science": {
            "description": "Data Science Analysis Project",
            "directories": [
                "data/raw",
                "data/interim",
                "data/processed",
                "notebooks/01-data-exploration",
                "notebooks/02-data-cleaning",
                "notebooks/03-modeling",
                "notebooks/04-visualization",
                "src/data",
                "src/features",
                "src/models",
                "src/visualization",
                "reports/figures",
                "references"
            ],
            "files": {
                "README.md": "# {project_name}\n\n## Data Science Project\n\nAnalysis and modeling for...\n\n## Project Organization\n\n- `data/`: Data files\n- `notebooks/`: Jupyter notebooks\n- `src/`: Source code\n- `reports/`: Generated reports\n",
                "requirements.txt": "pandas>=1.3.0\nnumpy>=1.21.0\nmatplotlib>=3.5.0\nseaborn>=0.11.0\njupyter>=1.0.0\nscikit-learn>=1.0.0\nplotly>=5.0.0\n",
                "environment.yml": "name: {project_name}\nchannels:\n  - conda-forge\ndependencies:\n  - python=3.9\n  - pandas\n  - numpy\n  - matplotlib\n  - jupyter\n  - scikit-learn\n"
            }
        }
    }
    
    @classmethod
    def create_project(cls, template: str, project_name: str, output_dir: str = ".") -> Path:
        """Create a new project from template."""
        if template not in cls.TEMPLATES:
            available = ", ".join(cls.TEMPLATES.keys())
            raise ValueError(f"Unknown template: {template}. Available: {available}")
        
        template_config = cls.TEMPLATES[template]
        project_path = Path(output_dir) / project_name
        
        print(f"🏗️  Creating {template_config['description']}: {project_name}")
        
        # Create project directory
        project_path.mkdir(parents=True, exist_ok=True)
        
        # Create directories
        for dir_path in template_config["directories"]:
            dir_path = dir_path.format(project_name=project_name)
            full_path = project_path / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            
            # Create .gitkeep for empty directories
            if not any(full_path.iterdir()):
                (full_path / ".gitkeep").touch()
        
        # Create files
        for file_path, content in template_config["files"].items():
            file_path = file_path.format(project_name=project_name)
            full_path = project_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Format content
            formatted_content = content.format(project_name=project_name)
            with open(full_path, 'w') as f:
                f.write(formatted_content)
        
        print(f"✅ Project created at: {project_path}")
        return project_path


class FileOrganizer:
    """Organize files based on patterns and rules."""
    
    DEFAULT_RULES = {
        "code": {
            "extensions": [".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".hpp"],
            "directory": "src"
        },
        "data": {
            "extensions": [".csv", ".json", ".xml", ".xlsx", ".parquet", ".h5"],
            "directory": "data"
        },
        "docs": {
            "extensions": [".md", ".txt", ".pdf", ".doc", ".docx"],
            "directory": "docs"
        },
        "images": {
            "extensions": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg"],
            "directory": "assets/images"
        },
        "configs": {
            "extensions": [".yaml", ".yml", ".ini", ".conf", ".cfg"],
            "directory": "configs"
        },
        "notebooks": {
            "extensions": [".ipynb"],
            "directory": "notebooks"
        },
        "models": {
            "extensions": [".pkl", ".pt", ".pth", ".h5", ".model", ".joblib"],
            "directory": "models"
        }
    }
    
    def __init__(self, rules: Optional[Dict] = None):
        self.rules = rules or self.DEFAULT_RULES
    
    def organize_directory(self, directory: Path, dry_run: bool = False) -> Dict[str, List[str]]:
        """Organize files in directory based on rules."""
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        moves = {}
        stats = {"moved": 0, "skipped": 0, "errors": 0}
        
        print(f"🗂️  Organizing directory: {directory}")
        
        for file_path in directory.rglob("*"):
            if file_path.is_file() and not file_path.name.startswith('.'):
                target_dir = self._get_target_directory(file_path)
                
                if target_dir:
                    target_path = directory / target_dir / file_path.name
                    
                    # Avoid moving files that are already in the right place
                    if target_path.parent == file_path.parent:
                        stats["skipped"] += 1
                        continue
                    
                    # Track move
                    if target_dir not in moves:
                        moves[target_dir] = []
                    moves[target_dir].append(str(file_path.relative_to(directory)))
                    
                    if not dry_run:
                        try:
                            target_path.parent.mkdir(parents=True, exist_ok=True)
                            
                            # Handle name conflicts
                            if target_path.exists():
                                base_name = target_path.stem
                                extension = target_path.suffix
                                counter = 1
                                while target_path.exists():
                                    target_path = target_path.parent / f"{base_name}_{counter}{extension}"
                                    counter += 1
                            
                            shutil.move(str(file_path), str(target_path))
                            stats["moved"] += 1
                            
                        except Exception as e:
                            print(f"❌ Error moving {file_path}: {e}")
                            stats["errors"] += 1
                    else:
                        stats["moved"] += 1
        
        # Print summary
        action = "Would move" if dry_run else "Moved"
        print(f"📊 Summary: {action} {stats['moved']} files, skipped {stats['skipped']}, errors {stats['errors']}")
        
        if moves:
            print("\n📁 File movements:")
            for target_dir, files in moves.items():
                print(f"  {target_dir}/:")
                for file_name in files[:5]:  # Show first 5 files
                    print(f"    - {file_name}")
                if len(files) > 5:
                    print(f"    ... and {len(files) - 5} more")
        
        return moves
    
    def _get_target_directory(self, file_path: Path) -> Optional[str]:
        """Determine target directory for a file based on rules."""
        extension = file_path.suffix.lower()
        
        for category, rule in self.rules.items():
            if extension in rule.get("extensions", []):
                return rule["directory"]
        
        return None


class FileDeduplicator:
    """Find and remove duplicate files."""
    
    def __init__(self):
        self.file_hashes = {}
        self.duplicates = {}
    
    def find_duplicates(self, directory: Path, strategy: str = "hash") -> Dict[str, List[Path]]:
        """Find duplicate files using different strategies."""
        directory = Path(directory)
        duplicates = {}
        
        print(f"🔍 Finding duplicates in: {directory}")
        print(f"📋 Strategy: {strategy}")
        
        if strategy == "hash":
            duplicates = self._find_by_hash(directory)
        elif strategy == "name":
            duplicates = self._find_by_name(directory)
        elif strategy == "size":
            duplicates = self._find_by_size(directory)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Print summary
        total_dupes = sum(len(files) - 1 for files in duplicates.values())
        total_size = sum(
            sum(f.stat().st_size for f in files[1:])
            for files in duplicates.values()
        )
        
        print(f"📊 Found {len(duplicates)} groups with {total_dupes} duplicate files")
        print(f"💾 Potential space savings: {total_size / 1024 / 1024:.2f} MB")
        
        return duplicates
    
    def _find_by_hash(self, directory: Path) -> Dict[str, List[Path]]:
        """Find duplicates by file hash."""
        hash_groups = {}
        
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                file_hash = self._calculate_file_hash(file_path)
                if file_hash not in hash_groups:
                    hash_groups[file_hash] = []
                hash_groups[file_hash].append(file_path)
        
        # Return only groups with duplicates
        return {h: files for h, files in hash_groups.items() if len(files) > 1}
    
    def _find_by_name(self, directory: Path) -> Dict[str, List[Path]]:
        """Find duplicates by filename."""
        name_groups = {}
        
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                name = file_path.name
                if name not in name_groups:
                    name_groups[name] = []
                name_groups[name].append(file_path)
        
        return {name: files for name, files in name_groups.items() if len(files) > 1}
    
    def _find_by_size(self, directory: Path) -> Dict[str, List[Path]]:
        """Find duplicates by file size."""
        size_groups = {}
        
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                size = file_path.stat().st_size
                if size not in size_groups:
                    size_groups[size] = []
                size_groups[size].append(file_path)
        
        return {str(size): files for size, files in size_groups.items() if len(files) > 1}
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file."""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except (IOError, OSError):
            return f"error_{file_path.name}"
    
    def remove_duplicates(self, duplicates: Dict[str, List[Path]], 
                         keep: str = "first", dry_run: bool = False) -> int:
        """Remove duplicate files."""
        removed_count = 0
        
        for group_id, files in duplicates.items():
            if len(files) <= 1:
                continue
            
            # Determine which file to keep
            if keep == "first":
                to_keep = files[0]
                to_remove = files[1:]
            elif keep == "largest":
                to_keep = max(files, key=lambda f: f.stat().st_size)
                to_remove = [f for f in files if f != to_keep]
            elif keep == "newest":
                to_keep = max(files, key=lambda f: f.stat().st_mtime)
                to_remove = [f for f in files if f != to_keep]
            else:
                print(f"Unknown keep strategy: {keep}")
                continue
            
            # Remove duplicates
            for file_path in to_remove:
                if not dry_run:
                    try:
                        file_path.unlink()
                        removed_count += 1
                        print(f"🗑️  Removed: {file_path}")
                    except OSError as e:
                        print(f"❌ Error removing {file_path}: {e}")
                else:
                    removed_count += 1
                    print(f"🗑️  Would remove: {file_path}")
        
        action = "Would remove" if dry_run else "Removed"
        print(f"✅ {action} {removed_count} duplicate files")
        return removed_count


class ProjectArchiver:
    """Archive and backup projects."""
    
    def archive_project(self, project_path: Path, output_path: Path, 
                       compression: str = "zip", exclude_patterns: List[str] = None) -> Path:
        """Archive a project directory."""
        project_path = Path(project_path)
        output_path = Path(output_path)
        
        if exclude_patterns is None:
            exclude_patterns = [
                "__pycache__",
                "*.pyc",
                ".git",
                "node_modules",
                ".DS_Store",
                "*.log",
                ".pytest_cache",
                ".mypy_cache"
            ]
        
        print(f"📦 Archiving project: {project_path}")
        print(f"📁 Output: {output_path}")
        
        if compression == "zip":
            return self._create_zip_archive(project_path, output_path, exclude_patterns)
        elif compression == "tar":
            return self._create_tar_archive(project_path, output_path, exclude_patterns, "tar")
        elif compression == "tar.gz":
            return self._create_tar_archive(project_path, output_path, exclude_patterns, "tar.gz")
        else:
            raise ValueError(f"Unsupported compression: {compression}")
    
    def _create_zip_archive(self, project_path: Path, output_path: Path, 
                           exclude_patterns: List[str]) -> Path:
        """Create ZIP archive."""
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in project_path.rglob("*"):
                if file_path.is_file() and not self._should_exclude(file_path, exclude_patterns):
                    arc_name = file_path.relative_to(project_path.parent)
                    zipf.write(file_path, arc_name)
        
        print(f"✅ ZIP archive created: {output_path}")
        return output_path
    
    def _create_tar_archive(self, project_path: Path, output_path: Path, 
                           exclude_patterns: List[str], format: str) -> Path:
        """Create TAR archive."""
        mode = "w:gz" if format == "tar.gz" else "w"
        
        with tarfile.open(output_path, mode) as tarf:
            for file_path in project_path.rglob("*"):
                if not self._should_exclude(file_path, exclude_patterns):
                    arc_name = file_path.relative_to(project_path.parent)
                    tarf.add(file_path, arc_name)
        
        print(f"✅ TAR archive created: {output_path}")
        return output_path
    
    def _should_exclude(self, file_path: Path, exclude_patterns: List[str]) -> bool:
        """Check if file should be excluded based on patterns."""
        for pattern in exclude_patterns:
            if fnmatch.fnmatch(file_path.name, pattern) or \
               fnmatch.fnmatch(str(file_path), pattern) or \
               any(fnmatch.fnmatch(part, pattern) for part in file_path.parts):
                return True
        return False
    
    def backup_files(self, source_paths: List[Path], destination: Path, 
                    incremental: bool = False) -> Path:
        """Backup files to destination."""
        destination = Path(destination)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = destination / f"backup_{timestamp}"
        
        print(f"💾 Creating backup: {backup_dir}")
        
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        for source_path in source_paths:
            source_path = Path(source_path)
            if source_path.exists():
                if source_path.is_file():
                    shutil.copy2(source_path, backup_dir / source_path.name)
                elif source_path.is_dir():
                    shutil.copytree(source_path, backup_dir / source_path.name, 
                                  dirs_exist_ok=True)
                print(f"✅ Backed up: {source_path}")
            else:
                print(f"⚠️  Source not found: {source_path}")
        
        # Create backup manifest
        manifest = {
            "created_at": datetime.now().isoformat(),
            "sources": [str(p) for p in source_paths],
            "destination": str(backup_dir),
            "incremental": incremental
        }
        
        with open(backup_dir / "backup_manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"✅ Backup completed: {backup_dir}")
        return backup_dir


def main():
    parser = argparse.ArgumentParser(description="Project organization and file management")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Scaffold command
    scaffold_parser = subparsers.add_parser('scaffold', help='Create project from template')
    scaffold_parser.add_argument('--template', choices=list(ProjectScaffolder.TEMPLATES.keys()),
                                required=True, help='Project template')
    scaffold_parser.add_argument('--name', required=True, help='Project name')
    scaffold_parser.add_argument('--output-dir', default='.', help='Output directory')

    # List templates command
    templates_parser = subparsers.add_parser('list-templates', help='List available templates')

    # Organize command
    organize_parser = subparsers.add_parser('organize', help='Organize files in directory')
    organize_parser.add_argument('--directory', required=True, help='Directory to organize')
    organize_parser.add_argument('--dry-run', action='store_true', help='Show what would be done')
    organize_parser.add_argument('--rules', help='Custom rules file (JSON)')

    # Dedupe command
    dedupe_parser = subparsers.add_parser('dedupe', help='Find and remove duplicates')
    dedupe_parser.add_argument('--directory', required=True, help='Directory to check')
    dedupe_parser.add_argument('--strategy', choices=['hash', 'name', 'size'], 
                              default='hash', help='Duplicate detection strategy')
    dedupe_parser.add_argument('--remove', action='store_true', help='Remove duplicates')
    dedupe_parser.add_argument('--keep', choices=['first', 'largest', 'newest'], 
                              default='first', help='Which duplicate to keep')
    dedupe_parser.add_argument('--dry-run', action='store_true', help='Show what would be done')

    # Archive command
    archive_parser = subparsers.add_parser('archive', help='Archive project')
    archive_parser.add_argument('--project', required=True, help='Project directory')
    archive_parser.add_argument('--output', required=True, help='Output archive path')
    archive_parser.add_argument('--compression', choices=['zip', 'tar', 'tar.gz'], 
                               default='zip', help='Compression format')
    archive_parser.add_argument('--exclude', nargs='*', help='Exclude patterns')

    # Backup command
    backup_parser = subparsers.add_parser('backup', help='Backup files')
    backup_parser.add_argument('--source', nargs='+', required=True, help='Source paths')
    backup_parser.add_argument('--destination', required=True, help='Backup destination')
    backup_parser.add_argument('--incremental', action='store_true', help='Incremental backup')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == 'scaffold':
            ProjectScaffolder.create_project(args.template, args.name, args.output_dir)

        elif args.command == 'list-templates':
            print("📋 Available project templates:")
            for name, config in ProjectScaffolder.TEMPLATES.items():
                print(f"  {name}: {config['description']}")

        elif args.command == 'organize':
            rules = None
            if args.rules:
                with open(args.rules, 'r') as f:
                    rules = json.load(f)
            
            organizer = FileOrganizer(rules)
            organizer.organize_directory(args.directory, args.dry_run)

        elif args.command == 'dedupe':
            deduplicator = FileDeduplicator()
            duplicates = deduplicator.find_duplicates(args.directory, args.strategy)
            
            if duplicates and args.remove:
                deduplicator.remove_duplicates(duplicates, args.keep, args.dry_run)

        elif args.command == 'archive':
            archiver = ProjectArchiver()
            exclude_patterns = args.exclude or []
            archiver.archive_project(args.project, args.output, args.compression, exclude_patterns)

        elif args.command == 'backup':
            archiver = ProjectArchiver()
            source_paths = [Path(p) for p in args.source]
            archiver.backup_files(source_paths, args.destination, args.incremental)

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()