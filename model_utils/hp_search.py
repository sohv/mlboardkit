#!/usr/bin/env python3
"""
hp_search.py

Simple hyperparameter search runner supporting grid and random search.
It can either call a Python function (if imported) or run a shell command template per trial.

Usage examples:

# Run a grid search and print parameter combos
python3 hp_search.py --mode grid --params "learning_rate:0.001,0.01;batch_size:16,32" --dry-run

# Run random search executing a command template
python3 hp_search.py --mode random --params "lr:0.0001,0.001,0.01;wd:0,0.0001" --trials 10 \
    --cmd "python train.py --lr {lr} --wd {wd} --batch-size {batch_size}"
"""

import argparse
import itertools
import random
import subprocess
import shlex
from typing import Dict, Any, List, Tuple
import json


def parse_params(param_str: str) -> Dict[str, List[str]]:
    params = {}
    for part in param_str.split(';'):
        if not part.strip():
            continue
        k, vals = part.split(':', 1)
        params[k.strip()] = [v.strip() for v in vals.split(',') if v.strip()]
    return params


def grid_combinations(params: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    keys = list(params.keys())
    vals = [params[k] for k in keys]
    combos = [dict(zip(keys, prod)) for prod in itertools.product(*vals)]
    return combos


def random_combinations(params: Dict[str, List[str]], trials: int) -> List[Dict[str, Any]]:
    keys = list(params.keys())
    combos = []
    for _ in range(trials):
        combo = {k: random.choice(params[k]) for k in keys}
        combos.append(combo)
    return combos


def run_command_template(cmd_template: str, combo: Dict[str, Any], dry_run: bool = False):
    cmd = cmd_template.format(**combo)
    print(f"Running: {cmd}")
    if dry_run:
        return 0
    try:
        result = subprocess.run(shlex.split(cmd), check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        return e.returncode


def main():
    parser = argparse.ArgumentParser(description='Simple hyperparameter search runner')
    parser.add_argument('--mode', choices=['grid', 'random'], default='grid')
    parser.add_argument('--params', required=True, help='Parameter spec e.g. "lr:0.001,0.01;bs:16,32"')
    parser.add_argument('--trials', type=int, default=10, help='Number of trials for random search')
    parser.add_argument('--cmd', help='Command template to run per trial (use {param} placeholders)')
    parser.add_argument('--dry-run', action='store_true', help='Only print commands')
    parser.add_argument('--output', help='Save combinations to JSON file')

    args = parser.parse_args()
    params = parse_params(args.params)

    if args.mode == 'grid':
        combos = grid_combinations(params)
    else:
        combos = random_combinations(params, args.trials)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(combos, f, indent=2)
        print(f"Saved {len(combos)} combos to {args.output}")

    if args.cmd:
        for combo in combos:
            code = run_command_template(args.cmd, combo, args.dry_run)
            if code != 0:
                print(f"Trial failed with code {code}")
    else:
        for combo in combos:
            print(combo)


if __name__ == '__main__':
    main()
