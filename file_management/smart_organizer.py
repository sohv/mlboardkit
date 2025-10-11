#!/usr/bin/env python3
"""
smart_organizer.py

Intelligently organize files based on content, metadata, and patterns.
"""

import argparse
import json
import os
import shutil
import hashlib
import mimetypes
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SmartFileOrganizer:
    def __init__(self):
        self.organization_rules = self.load_default_rules()
        self.stats = {
            'files_processed': 0,
            'files_moved': 0,
            'files_renamed': 0,
            'duplicates_found': 0,
            'errors': 0
        }
    
    def load_default_rules(self) -> Dict[str, Any]:
        """Load default organization rules"""
        return {
            'by_extension': {
                'images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.svg', '.webp'],
                'documents': ['.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt', '.pages'],
                'spreadsheets': ['.xls', '.xlsx', '.csv', '.ods', '.numbers'],
                'presentations': ['.ppt', '.pptx', '.odp', '.key'],
                'archives': ['.zip', '.rar', '.7z', '.tar', '.gz', '.bz2'],
                'videos': ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', '.webm'],
                'audio': ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma'],
                'code': ['.py', '.js', '.html', '.css', '.cpp', '.java', '.c', '.h'],
                'data': ['.json', '.xml', '.yaml', '.yml', '.sql', '.db', '.sqlite']
            },
            'by_size': {
                'large': 100 * 1024 * 1024,  # 100MB
                'medium': 10 * 1024 * 1024,   # 10MB
                'small': 1 * 1024 * 1024      # 1MB
            },
            'by_date': {
                'recent': 30,    # Last 30 days
                'month': 365,    # Last year
                'old': 365 * 2   # Older than 2 years
            }
        }
    
    def load_custom_rules(self, rules_file: str):
        """Load custom organization rules from file"""
        with open(rules_file, 'r', encoding='utf-8') as f:
            custom_rules = json.load(f)
        
        # Merge with default rules
        for category, rules in custom_rules.items():
            if category in self.organization_rules:
                self.organization_rules[category].update(rules)
            else:
                self.organization_rules[category] = rules
    
    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze file and extract metadata"""
        
        analysis = {
            'path': str(file_path),
            'name': file_path.name,
            'stem': file_path.stem,
            'suffix': file_path.suffix.lower(),
            'size': 0,
            'modified_time': None,
            'created_time': None,
            'mime_type': None,
            'hash': None,
            'category': 'other'
        }
        
        try:
            stat = file_path.stat()
            analysis.update({
                'size': stat.st_size,
                'modified_time': datetime.fromtimestamp(stat.st_mtime),
                'created_time': datetime.fromtimestamp(stat.st_ctime)
            })
            
            # Get MIME type
            mime_type, _ = mimetypes.guess_type(str(file_path))
            analysis['mime_type'] = mime_type
            
            # Calculate file hash for duplicate detection
            if stat.st_size < 100 * 1024 * 1024:  # Only hash files < 100MB
                analysis['hash'] = self.calculate_file_hash(file_path)
            
            # Determine category
            analysis['category'] = self.categorize_file(file_path, analysis)
            
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}")
            self.stats['errors'] += 1
        
        return analysis
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file"""
        hash_sha256 = hashlib.sha256()
        
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return None
    
    def categorize_file(self, file_path: Path, analysis: Dict[str, Any]) -> str:
        """Categorize file based on rules"""
        
        suffix = analysis['suffix']
        
        # Check extension-based categories
        for category, extensions in self.organization_rules['by_extension'].items():
            if suffix in extensions:
                return category
        
        # Check MIME type
        mime_type = analysis['mime_type']
        if mime_type:
            if mime_type.startswith('image/'):
                return 'images'
            elif mime_type.startswith('video/'):
                return 'videos'
            elif mime_type.startswith('audio/'):
                return 'audio'
            elif mime_type.startswith('text/'):
                return 'documents'
        
        return 'other'
    
    def find_duplicates(self, file_analyses: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Find duplicate files based on hash"""
        
        hash_groups = {}
        
        for analysis in file_analyses:
            file_hash = analysis.get('hash')
            if file_hash:
                if file_hash not in hash_groups:
                    hash_groups[file_hash] = []
                hash_groups[file_hash].append(analysis)
        
        # Return only groups with duplicates
        duplicates = {h: files for h, files in hash_groups.items() if len(files) > 1}
        
        return duplicates
    
    def generate_organization_plan(self, file_analyses: List[Dict[str, Any]], 
                                 target_dir: str, strategy: str) -> Dict[str, Any]:
        """Generate file organization plan"""
        
        target_path = Path(target_dir)
        plan = {
            'moves': [],
            'renames': [],
            'duplicates': [],
            'errors': []
        }
        
        # Find duplicates
        duplicates = self.find_duplicates(file_analyses)
        self.stats['duplicates_found'] = len(duplicates)
        
        for file_hash, duplicate_files in duplicates.items():
            # Keep the first file, mark others as duplicates
            for dup_file in duplicate_files[1:]:
                plan['duplicates'].append({
                    'file': dup_file['path'],
                    'original': duplicate_files[0]['path'],
                    'size': dup_file['size']
                })
        
        # Get non-duplicate files
        duplicate_paths = set()
        for dup_list in duplicates.values():
            for dup_file in dup_list[1:]:
                duplicate_paths.add(dup_file['path'])
        
        non_duplicate_files = [f for f in file_analyses if f['path'] not in duplicate_paths]
        
        # Generate moves based on strategy
        if strategy == 'by_type':
            plan['moves'].extend(self.plan_by_type(non_duplicate_files, target_path))
        elif strategy == 'by_date':
            plan['moves'].extend(self.plan_by_date(non_duplicate_files, target_path))
        elif strategy == 'by_size':
            plan['moves'].extend(self.plan_by_size(non_duplicate_files, target_path))
        elif strategy == 'smart':
            plan['moves'].extend(self.plan_smart_organization(non_duplicate_files, target_path))
        
        return plan
    
    def plan_by_type(self, files: List[Dict[str, Any]], target_dir: Path) -> List[Dict[str, str]]:
        """Plan organization by file type"""
        
        moves = []
        
        for file_info in files:
            category = file_info['category']
            source_path = Path(file_info['path'])
            
            # Create target directory structure
            target_subdir = target_dir / category
            target_file = target_subdir / source_path.name
            
            moves.append({
                'source': str(source_path),
                'target': str(target_file),
                'reason': f'File type: {category}'
            })
        
        return moves
    
    def plan_by_date(self, files: List[Dict[str, Any]], target_dir: Path) -> List[Dict[str, str]]:
        """Plan organization by date"""
        
        moves = []
        
        for file_info in files:
            source_path = Path(file_info['path'])
            modified_time = file_info['modified_time']
            
            if modified_time:
                # Create year/month directory structure
                year = modified_time.year
                month = modified_time.strftime('%m-%B')
                
                target_subdir = target_dir / str(year) / month
                target_file = target_subdir / source_path.name
                
                moves.append({
                    'source': str(source_path),
                    'target': str(target_file),
                    'reason': f'Date: {year}/{month}'
                })
        
        return moves
    
    def plan_by_size(self, files: List[Dict[str, Any]], target_dir: Path) -> List[Dict[str, str]]:
        """Plan organization by file size"""
        
        moves = []
        size_rules = self.organization_rules['by_size']
        
        for file_info in files:
            source_path = Path(file_info['path'])
            file_size = file_info['size']
            
            # Determine size category
            if file_size > size_rules['large']:
                size_category = 'large'
            elif file_size > size_rules['medium']:
                size_category = 'medium'
            elif file_size > size_rules['small']:
                size_category = 'small'
            else:
                size_category = 'tiny'
            
            target_subdir = target_dir / size_category
            target_file = target_subdir / source_path.name
            
            moves.append({
                'source': str(source_path),
                'target': str(target_file),
                'reason': f'Size: {size_category} ({file_size:,} bytes)'
            })
        
        return moves
    
    def plan_smart_organization(self, files: List[Dict[str, Any]], target_dir: Path) -> List[Dict[str, str]]:
        """Plan smart organization combining multiple criteria"""
        
        moves = []
        
        for file_info in files:
            source_path = Path(file_info['path'])
            category = file_info['category']
            modified_time = file_info['modified_time']
            
            # Create hierarchical structure: category/year
            if modified_time:
                year = modified_time.year
                target_subdir = target_dir / category / str(year)
            else:
                target_subdir = target_dir / category / 'unknown_date'
            
            target_file = target_subdir / source_path.name
            
            moves.append({
                'source': str(source_path),
                'target': str(target_file),
                'reason': f'Smart: {category} from {year if modified_time else "unknown date"}'
            })
        
        return moves
    
    def execute_plan(self, plan: Dict[str, Any], dry_run: bool = True) -> Dict[str, Any]:
        """Execute the organization plan"""
        
        results = {
            'moves_executed': 0,
            'duplicates_handled': 0,
            'errors': [],
            'created_directories': set()
        }
        
        if dry_run:
            logger.info("DRY RUN: No files will be moved")
            return results
        
        # Handle duplicates
        for duplicate in plan['duplicates']:
            try:
                duplicate_path = Path(duplicate['file'])
                if duplicate_path.exists():
                    duplicate_path.unlink()  # Delete duplicate
                    results['duplicates_handled'] += 1
                    logger.info(f"Deleted duplicate: {duplicate_path}")
            except Exception as e:
                error_msg = f"Error deleting duplicate {duplicate['file']}: {e}"
                results['errors'].append(error_msg)
                logger.error(error_msg)
        
        # Execute moves
        for move in plan['moves']:
            try:
                source_path = Path(move['source'])
                target_path = Path(move['target'])
                
                if not source_path.exists():
                    continue
                
                # Create target directory
                target_path.parent.mkdir(parents=True, exist_ok=True)
                results['created_directories'].add(str(target_path.parent))
                
                # Handle name conflicts
                if target_path.exists():
                    counter = 1
                    while target_path.exists():
                        name_parts = target_path.stem, counter, target_path.suffix
                        new_name = f"{name_parts[0]}_{name_parts[1]}{name_parts[2]}"
                        target_path = target_path.parent / new_name
                        counter += 1
                
                # Move file
                shutil.move(str(source_path), str(target_path))
                results['moves_executed'] += 1
                logger.info(f"Moved: {source_path} -> {target_path}")
                
            except Exception as e:
                error_msg = f"Error moving {move['source']}: {e}"
                results['errors'].append(error_msg)
                logger.error(error_msg)
        
        results['created_directories'] = list(results['created_directories'])
        return results
    
    def scan_directory(self, directory: str, recursive: bool = True, 
                      include_hidden: bool = False) -> List[Dict[str, Any]]:
        """Scan directory and analyze all files"""
        
        dir_path = Path(directory)
        if not dir_path.exists():
            raise ValueError(f"Directory does not exist: {directory}")
        
        files = []
        
        if recursive:
            pattern = "**/*" if include_hidden else "**/[!.]*"
            file_paths = dir_path.glob(pattern)
        else:
            pattern = "*" if include_hidden else "[!.]*"
            file_paths = dir_path.glob(pattern)
        
        for file_path in file_paths:
            if file_path.is_file():
                analysis = self.analyze_file(file_path)
                files.append(analysis)
                self.stats['files_processed'] += 1
                
                if self.stats['files_processed'] % 100 == 0:
                    logger.info(f"Processed {self.stats['files_processed']} files...")
        
        logger.info(f"Completed scanning. Total files: {len(files)}")
        return files
    
    def generate_report(self, file_analyses: List[Dict[str, Any]], 
                       plan: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate organization report"""
        
        # File statistics
        total_files = len(file_analyses)
        total_size = sum(f['size'] for f in file_analyses)
        
        # Category breakdown
        categories = {}
        for file_info in file_analyses:
            category = file_info['category']
            if category not in categories:
                categories[category] = {'count': 0, 'size': 0}
            categories[category]['count'] += 1
            categories[category]['size'] += file_info['size']
        
        # Size distribution
        size_rules = self.organization_rules['by_size']
        size_distribution = {'large': 0, 'medium': 0, 'small': 0, 'tiny': 0}
        
        for file_info in file_analyses:
            size = file_info['size']
            if size > size_rules['large']:
                size_distribution['large'] += 1
            elif size > size_rules['medium']:
                size_distribution['medium'] += 1
            elif size > size_rules['small']:
                size_distribution['small'] += 1
            else:
                size_distribution['tiny'] += 1
        
        report = {
            'summary': {
                'total_files': total_files,
                'total_size': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'categories': categories,
                'size_distribution': size_distribution,
                'duplicates_found': self.stats['duplicates_found']
            },
            'processing_stats': self.stats
        }
        
        if plan:
            report['organization_plan'] = {
                'moves_planned': len(plan['moves']),
                'duplicates_to_remove': len(plan['duplicates']),
                'space_to_free': sum(d['size'] for d in plan['duplicates'])
            }
        
        return report

def main():
    parser = argparse.ArgumentParser(description="Smart file organization tool")
    parser.add_argument('source_dir', help='Source directory to organize')
    parser.add_argument('--target-dir', help='Target directory for organized files')
    parser.add_argument('--strategy', choices=['by_type', 'by_date', 'by_size', 'smart'],
                       default='smart', help='Organization strategy')
    parser.add_argument('--rules', help='Custom rules file (JSON)')
    parser.add_argument('--recursive', action='store_true', help='Scan recursively')
    parser.add_argument('--include-hidden', action='store_true', help='Include hidden files')
    parser.add_argument('--dry-run', action='store_true', help='Show plan without executing')
    parser.add_argument('--report', help='Save report to file')
    parser.add_argument('--plan', help='Save plan to file')
    
    args = parser.parse_args()
    
    organizer = SmartFileOrganizer()
    
    # Load custom rules if provided
    if args.rules:
        organizer.load_custom_rules(args.rules)
        logger.info(f"Loaded custom rules from {args.rules}")
    
    # Set target directory
    if args.target_dir:
        target_dir = args.target_dir
    else:
        target_dir = Path(args.source_dir) / 'organized'
    
    # Scan source directory
    print(f"Scanning directory: {args.source_dir}")
    files = organizer.scan_directory(
        args.source_dir, 
        recursive=args.recursive,
        include_hidden=args.include_hidden
    )
    
    if not files:
        print("No files found to organize")
        return
    
    # Generate organization plan
    print(f"Generating organization plan using strategy: {args.strategy}")
    plan = organizer.generate_organization_plan(files, target_dir, args.strategy)
    
    # Generate report
    report = organizer.generate_report(files, plan)
    
    # Display summary
    print(f"\nOrganization Summary:")
    print(f"  Total files: {report['summary']['total_files']:,}")
    print(f"  Total size: {report['summary']['total_size_mb']:.2f} MB")
    print(f"  Duplicates found: {report['summary']['duplicates_found']}")
    print(f"  Files to move: {len(plan['moves'])}")
    print(f"  Duplicates to remove: {len(plan['duplicates'])}")
    print(f"  Space to free: {plan.get('space_to_free', 0) / (1024*1024):.2f} MB")
    
    # Show category breakdown
    print(f"\nFile Categories:")
    for category, stats in report['summary']['categories'].items():
        size_mb = stats['size'] / (1024 * 1024)
        print(f"  {category}: {stats['count']} files ({size_mb:.2f} MB)")
    
    # Show some example moves
    if plan['moves']:
        print(f"\nExample moves:")
        for move in plan['moves'][:5]:
            print(f"  {Path(move['source']).name} -> {move['target']}")
        if len(plan['moves']) > 5:
            print(f"  ... and {len(plan['moves']) - 5} more")
    
    # Save plan if requested
    if args.plan:
        with open(args.plan, 'w', encoding='utf-8') as f:
            json.dump(plan, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nPlan saved to {args.plan}")
    
    # Save report if requested
    if args.report:
        with open(args.report, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print(f"Report saved to {args.report}")
    
    # Execute plan
    if not args.dry_run:
        confirm = input(f"\nProceed with organization? (y/N): ")
        if confirm.lower() == 'y':
            print("Executing organization plan...")
            results = organizer.execute_plan(plan, dry_run=False)
            
            print(f"\nExecution Results:")
            print(f"  Files moved: {results['moves_executed']}")
            print(f"  Duplicates removed: {results['duplicates_handled']}")
            print(f"  Directories created: {len(results['created_directories'])}")
            print(f"  Errors: {len(results['errors'])}")
            
            if results['errors']:
                print(f"\nErrors:")
                for error in results['errors'][:5]:
                    print(f"  {error}")
        else:
            print("Organization cancelled")
    else:
        print(f"\nDry run completed. Use --dry-run=false to execute.")

if __name__ == "__main__":
    main()