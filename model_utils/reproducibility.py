#!/usr/bin/env python3
"""
reproducibility.py

Helpers to make experiments reproducible: seeding RNGs, capturing git commit, and
recording environment metadata.
"""

import random
import numpy as np
import os
import json
import subprocess
import sys
from datetime import datetime
from typing import Dict, Any


def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    np.random.seed(seed)


def get_git_commit() -> Dict[str, Any]:
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode().strip()
        diff = subprocess.check_output(['git', 'status', '--porcelain']).decode().strip()
        return {'commit': commit, 'branch': branch, 'dirty': bool(diff)}
    except Exception:
        return {'commit': None, 'branch': None, 'dirty': None}


def capture_experiment_metadata(output_path: str, extras: Dict[str, Any] = None):
    data = {
        'created_at': datetime.now().isoformat(),
        'git': get_git_commit(),
        'python_version': sys.version,
        'environment': dict(os.environ)
    }
    if extras:
        data.update(extras)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved experiment metadata to {output_path}")


if __name__ == '__main__':
    print('reproducibility helpers loaded')
