#!/usr/bin/env python3
"""
seed_everything.py

Comprehensive reproducibility and random seed management for ML projects.
Goes beyond basic seeding to include environment, hardware, and framework-specific configurations.
"""

import argparse
import os
import json
import random
import sys
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

# Optional imports with graceful fallbacks
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    from sklearn.utils import check_random_state
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class ReproducibilityManager:
    """Comprehensive reproducibility manager for ML experiments"""
    
    def __init__(self, seed: int = 42, config_file: Optional[str] = None):
        self.seed = seed
        self.config = self._load_config(config_file) if config_file else {}
        self.environment_info = {}
        
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load reproducibility configuration from file"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load config file {config_file}: {e}")
            return {}
    
    def seed_everything(self, deterministic: bool = True, benchmark: bool = False):
        """Set seeds for all available random number generators"""
        print(f"Setting global seed to {self.seed}")
        
        # Python's built-in random
        random.seed(self.seed)
        
        # Environment variable for Python hash seed
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        
        # NumPy
        if HAS_NUMPY:
            np.random.seed(self.seed)
            print("✓ NumPy random seed set")
        
        # PyTorch
        if HAS_TORCH:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            
            if deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                # For newer PyTorch versions
                if hasattr(torch, 'use_deterministic_algorithms'):
                    torch.use_deterministic_algorithms(True)
            else:
                torch.backends.cudnn.benchmark = benchmark
            
            print("✓ PyTorch random seed set")
        
        # TensorFlow
        if HAS_TF:
            tf.random.set_seed(self.seed)
            print("✓ TensorFlow random seed set")
        
        # Additional environment variables for determinism
        if deterministic:
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
            os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    def create_seeded_generators(self) -> Dict[str, Any]:
        """Create seeded random number generators for different frameworks"""
        generators = {}
        
        # Python random generator
        python_rng = random.Random(self.seed)
        generators['python'] = python_rng
        
        # NumPy generator
        if HAS_NUMPY:
            numpy_rng = np.random.RandomState(self.seed)
            generators['numpy'] = numpy_rng
            
            # New-style NumPy generator (NumPy 1.17+)
            if hasattr(np.random, 'default_rng'):
                generators['numpy_new'] = np.random.default_rng(self.seed)
        
        # PyTorch generator
        if HAS_TORCH:
            torch_rng = torch.Generator()
            torch_rng.manual_seed(self.seed)
            generators['torch'] = torch_rng
        
        # Scikit-learn compatible generator
        if HAS_SKLEARN:
            sklearn_rng = check_random_state(self.seed)
            generators['sklearn'] = sklearn_rng
        
        return generators
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Collect comprehensive environment information"""
        info = {
            'python_version': sys.version,
            'platform': sys.platform,
            'seed': self.seed,
            'python_hash_seed': os.environ.get('PYTHONHASHSEED', 'not_set'),
            'environment_variables': {}
        }
        
        # Relevant environment variables
        env_vars = [
            'CUDA_VISIBLE_DEVICES', 'TF_DETERMINISTIC_OPS', 'TF_CUDNN_DETERMINISTIC',
            'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'
        ]
        
        for var in env_vars:
            info['environment_variables'][var] = os.environ.get(var, 'not_set')
        
        # Framework versions
        if HAS_NUMPY:
            info['numpy_version'] = np.__version__
        
        if HAS_TORCH:
            info['torch_version'] = torch.__version__
            info['cuda_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info['cuda_version'] = torch.version.cuda
                info['cudnn_version'] = torch.backends.cudnn.version()
                info['gpu_count'] = torch.cuda.device_count()
                info['gpu_names'] = [torch.cuda.get_device_name(i) 
                                   for i in range(torch.cuda.device_count())]
        
        if HAS_TF:
            info['tensorflow_version'] = tf.__version__
            info['tf_gpu_available'] = len(tf.config.list_physical_devices('GPU')) > 0
        
        self.environment_info = info
        return info
    
    def save_reproducibility_info(self, output_dir: str = "reproducibility"):
        """Save comprehensive reproducibility information"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Environment info
        env_info = self.get_environment_info()
        
        # Create reproducibility report
        report = {
            'timestamp': str(datetime.now()),
            'seed': self.seed,
            'environment': env_info,
            'configuration': self.config,
            'reproducibility_checklist': self._create_checklist()
        }
        
        # Save main report
        with open(output_path / 'reproducibility_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Create requirements snapshot
        self._save_requirements(output_path)
        
        # Create setup script
        self._create_setup_script(output_path)
        
        print(f"Reproducibility information saved to {output_path}")
    
    def _create_checklist(self) -> Dict[str, Any]:
        """Create reproducibility checklist"""
        checklist = {
            'random_seeds_set': True,
            'environment_variables_set': {
                'PYTHONHASHSEED': os.environ.get('PYTHONHASHSEED') is not None,
                'TF_DETERMINISTIC_OPS': os.environ.get('TF_DETERMINISTIC_OPS') == '1',
            },
            'framework_specific': {}
        }
        
        if HAS_TORCH:
            checklist['framework_specific']['pytorch'] = {
                'manual_seed_set': True,
                'cudnn_deterministic': getattr(torch.backends.cudnn, 'deterministic', False),
                'cudnn_benchmark': getattr(torch.backends.cudnn, 'benchmark', True),
            }
        
        if HAS_TF:
            checklist['framework_specific']['tensorflow'] = {
                'random_seed_set': True,
                'deterministic_ops_enabled': os.environ.get('TF_DETERMINISTIC_OPS') == '1'
            }
        
        return checklist
    
    def _save_requirements(self, output_path: Path):
        """Save current package requirements"""
        try:
            import pkg_resources
            
            installed_packages = [d for d in pkg_resources.working_set]
            requirements = []
            
            for package in installed_packages:
                requirements.append(f"{package.project_name}=={package.version}")
            
            with open(output_path / 'requirements.txt', 'w') as f:
                f.write('\n'.join(sorted(requirements)))
                
        except Exception as e:
            print(f"Could not save requirements: {e}")
    
    def _create_setup_script(self, output_path: Path):
        """Create a setup script to reproduce the environment"""
        script_content = f'''#!/usr/bin/env python3
"""
Auto-generated reproducibility setup script
Generated seed: {self.seed}
"""

import os
import random

def setup_reproducibility():
    """Setup reproducible environment"""
    seed = {self.seed}
    
    # Set environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    # Python random
    random.seed(seed)
    
    # NumPy
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    
    # PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    # TensorFlow
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    print(f"Reproducibility setup complete with seed {{seed}}")

if __name__ == "__main__":
    setup_reproducibility()
'''
        
        with open(output_path / 'setup_reproducibility.py', 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(output_path / 'setup_reproducibility.py', 0o755)
    
    def verify_reproducibility(self, test_function, iterations: int = 5) -> Dict[str, Any]:
        """Verify that a function produces consistent results"""
        results = []
        
        for i in range(iterations):
            # Re-seed before each iteration
            self.seed_everything()
            
            try:
                result = test_function()
                results.append(result)
            except Exception as e:
                return {'status': 'failed', 'error': str(e)}
        
        # Check if all results are identical
        first_result = results[0]
        all_identical = all(
            str(result) == str(first_result) for result in results[1:]
        )
        
        return {
            'status': 'reproducible' if all_identical else 'not_reproducible',
            'iterations': iterations,
            'results_sample': results[:3],  # Show first 3 results
            'all_identical': all_identical
        }
    
    def create_experiment_hash(self, experiment_config: Dict[str, Any]) -> str:
        """Create a unique hash for experiment configuration"""
        # Include seed and environment info
        hash_data = {
            'seed': self.seed,
            'config': experiment_config,
            'environment_subset': {
                'python_version': sys.version,
                'platform': sys.platform
            }
        }
        
        # Create deterministic hash
        hash_string = json.dumps(hash_data, sort_keys=True)
        return hashlib.md5(hash_string.encode()).hexdigest()[:12]


def generate_deterministic_split(data_size: int, 
                               train_ratio: float = 0.8, 
                               val_ratio: float = 0.1,
                               seed: int = 42) -> Dict[str, List[int]]:
    """Generate reproducible train/val/test splits"""
    if HAS_NUMPY:
        rng = np.random.RandomState(seed)
        indices = rng.permutation(data_size)
    else:
        rng = random.Random(seed)
        indices = list(range(data_size))
        rng.shuffle(indices)
    
    train_size = int(data_size * train_ratio)
    val_size = int(data_size * val_ratio)
    
    return {
        'train': indices[:train_size].tolist() if HAS_NUMPY else indices[:train_size],
        'val': indices[train_size:train_size + val_size].tolist() if HAS_NUMPY else indices[train_size:train_size + val_size],
        'test': indices[train_size + val_size:].tolist() if HAS_NUMPY else indices[train_size + val_size:]
    }


def main():
    parser = argparse.ArgumentParser(description="Comprehensive reproducibility management")
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--deterministic', action='store_true', 
                       help='Enable deterministic mode (slower but reproducible)')
    parser.add_argument('--save-info', action='store_true',
                       help='Save reproducibility information')
    parser.add_argument('--output-dir', default='reproducibility',
                       help='Output directory for reproducibility info')
    parser.add_argument('--verify', help='Python file with test function to verify reproducibility')
    parser.add_argument('--split-data', type=int, help='Generate data splits for given size')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='Validation set ratio')
    
    args = parser.parse_args()
    
    # Initialize reproducibility manager
    manager = ReproducibilityManager(seed=args.seed, config_file=args.config)
    
    # Set seeds
    manager.seed_everything(deterministic=args.deterministic)
    
    # Save reproducibility info
    if args.save_info:
        manager.save_reproducibility_info(args.output_dir)
    
    # Verify reproducibility
    if args.verify:
        # This would require dynamic import of test function
        print("Reproducibility verification would require custom test function")
    
    # Generate data splits
    if args.split_data:
        splits = generate_deterministic_split(
            args.split_data, 
            args.train_ratio, 
            args.val_ratio, 
            args.seed
        )
        
        print(f"Generated splits for {args.split_data} samples:")
        print(f"  Train: {len(splits['train'])} samples")
        print(f"  Val: {len(splits['val'])} samples") 
        print(f"  Test: {len(splits['test'])} samples")
        
        # Save splits
        output_path = Path(args.output_dir)
        output_path.mkdir(exist_ok=True)
        with open(output_path / 'data_splits.json', 'w') as f:
            json.dump(splits, f, indent=2)
    
    # Print environment info
    env_info = manager.get_environment_info()
    print("\nEnvironment Information:")
    print(f"  Python: {env_info['python_version']}")
    print(f"  Platform: {env_info['platform']}")
    print(f"  Seed: {env_info['seed']}")
    
    if HAS_TORCH and env_info.get('cuda_available'):
        print(f"  CUDA: {env_info.get('cuda_version', 'N/A')}")
        print(f"  GPUs: {env_info.get('gpu_count', 0)}")


if __name__ == "__main__":
    main()