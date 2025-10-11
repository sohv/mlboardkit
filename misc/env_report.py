#!/usr/bin/env python3
"""
env_report.py

Dump environment and system information useful for experiment reproducibility.
"""

import json
import platform
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any


def get_python_packages() -> Dict[str, str]:
    try:
        import pkg_resources
        return {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    except Exception:
        return {}


def get_gpu_info() -> Dict[str, Any]:
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader']).decode().strip()
        lines = [line for line in result.splitlines() if line.strip()]
        gpus = []
        for line in lines:
            parts = [p.strip() for p in line.split(',')]
            gpus.append({'name': parts[0], 'memory': parts[1], 'driver': parts[2]})
        return {'gpus': gpus}
    except Exception:
        return {'gpus': []}


def generate_report(output_path: str):
    data = {
        'platform': platform.platform(),
        'python_version': sys.version,
        'machine': platform.machine(),
        'processor': platform.processor(),
        'packages': get_python_packages(),
        'gpu': get_gpu_info()
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved environment report to {output_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='env_report.json')
    args = parser.parse_args()
    generate_report(args.output)
