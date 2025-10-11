#!/usr/bin/env python3
"""
compress_folder.py

Zip up folders for sharing or backup.
Simple and focused compression utility.
"""

import argparse
import zipfile
import tarfile
import shutil
from pathlib import Path
from typing import List, Optional


def create_zip(source_path: str, output_path: str, exclude_patterns: List[str] = None):
    """Create ZIP archive from folder"""
    source = Path(source_path)
    output = Path(output_path)
    
    if exclude_patterns is None:
        exclude_patterns = []
    
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zipf:
        if source.is_file():
            zipf.write(source, source.name)
        else:
            for file_path in source.rglob('*'):
                if file_path.is_file():
                    # Check exclusions
                    should_exclude = any(pattern in str(file_path) for pattern in exclude_patterns)
                    if not should_exclude:
                        arcname = file_path.relative_to(source.parent)
                        zipf.write(file_path, arcname)
    
    print(f"ZIP archive created: {output}")


def create_tar(source_path: str, output_path: str, compression: str = 'gz', 
               exclude_patterns: List[str] = None):
    """Create TAR archive from folder"""
    source = Path(source_path)
    output = Path(output_path)
    
    if exclude_patterns is None:
        exclude_patterns = []
    
    mode = f'w:{compression}' if compression else 'w'
    
    with tarfile.open(output, mode) as tarf:
        if source.is_file():
            tarf.add(source, source.name)
        else:
            for file_path in source.rglob('*'):
                if file_path.is_file():
                    should_exclude = any(pattern in str(file_path) for pattern in exclude_patterns)
                    if not should_exclude:
                        arcname = file_path.relative_to(source.parent)
                        tarf.add(file_path, arcname)
    
    print(f"TAR archive created: {output}")


def main():
    parser = argparse.ArgumentParser(description="Compress folders for sharing or backup")
    parser.add_argument('source', help='Source folder or file path')
    parser.add_argument('--output', help='Output archive path')
    parser.add_argument('--format', choices=['zip', 'tar', 'tar.gz', 'tar.bz2'], 
                       default='zip', help='Archive format')
    parser.add_argument('--exclude', nargs='+', 
                       default=['.git', '__pycache__', '.DS_Store', '*.pyc'],
                       help='Patterns to exclude')
    
    args = parser.parse_args()
    
    source_path = Path(args.source)
    
    if not source_path.exists():
        print(f"Error: Source path does not exist: {args.source}")
        return
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        if args.format == 'zip':
            output_path = f"{source_path.name}.zip"
        elif args.format == 'tar':
            output_path = f"{source_path.name}.tar"
        elif args.format == 'tar.gz':
            output_path = f"{source_path.name}.tar.gz"
        elif args.format == 'tar.bz2':
            output_path = f"{source_path.name}.tar.bz2"
    
    print(f"Compressing {args.source} to {output_path}...")
    
    # Create archive
    if args.format == 'zip':
        create_zip(args.source, output_path, args.exclude)
    elif args.format.startswith('tar'):
        compression = args.format.split('.')[-1] if '.' in args.format else None
        create_tar(args.source, output_path, compression, args.exclude)


if __name__ == "__main__":
    main()