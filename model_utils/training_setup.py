#!/usr/bin/env python3
"""
ML Training Environment Setup and Validation Script

Detects GPU availability, validates dependencies, configures logging,
and sets up optimal training environments for ML/AI workflows.

Usage:
    python3 training_setup.py check --verbose
    python3 training_setup.py setup --framework pytorch --gpu-check
    python3 training_setup.py benchmark --quick
    python3 training_setup.py install --requirements requirements.txt
"""

import argparse
import sys
import os
import subprocess
import platform
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import importlib.util


class SystemInfo:
    """Collects and analyzes system information."""
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get comprehensive system information."""
        import psutil
        
        info = {
            'platform': {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'python_version': platform.python_version(),
                'python_implementation': platform.python_implementation()
            },
            'hardware': {
                'cpu_count_logical': psutil.cpu_count(logical=True),
                'cpu_count_physical': psutil.cpu_count(logical=False),
                'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'memory_available_gb': psutil.virtual_memory().available / (1024**3),
                'disk_usage': {
                    path: psutil.disk_usage(path)._asdict() 
                    for path in ['/'] if os.path.exists(path)
                }
            }
        }
        
        return info
    
    @staticmethod
    def check_gpu_availability() -> Dict[str, Any]:
        """Check GPU availability and specifications."""
        gpu_info = {
            'cuda_available': False,
            'cuda_version': None,
            'gpu_count': 0,
            'gpus': [],
            'mps_available': False,  # Apple Metal Performance Shaders
            'rocm_available': False  # AMD ROCm
        }
        
        # Check CUDA
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info['cuda_available'] = True
                gpu_info['cuda_version'] = torch.version.cuda
                gpu_info['gpu_count'] = torch.cuda.device_count()
                
                for i in range(gpu_info['gpu_count']):
                    gpu_props = torch.cuda.get_device_properties(i)
                    gpu_info['gpus'].append({
                        'id': i,
                        'name': gpu_props.name,
                        'memory_total_gb': gpu_props.total_memory / (1024**3),
                        'compute_capability': f"{gpu_props.major}.{gpu_props.minor}"
                    })
        except ImportError:
            pass
        
        # Check Apple MPS
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                gpu_info['mps_available'] = True
        except (ImportError, AttributeError):
            pass
        
        # Check ROCm
        try:
            result = subprocess.run(['rocm-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                gpu_info['rocm_available'] = True
        except FileNotFoundError:
            pass
        
        return gpu_info


class DependencyChecker:
    """Checks and validates ML/AI dependencies."""
    
    COMMON_PACKAGES = {
        'pytorch': {
            'packages': ['torch', 'torchvision', 'torchaudio'],
            'import_names': ['torch', 'torchvision', 'torchaudio'],
            'gpu_check': lambda: __import__('torch').cuda.is_available()
        },
        'tensorflow': {
            'packages': ['tensorflow'],
            'import_names': ['tensorflow'],
            'gpu_check': lambda: len(__import__('tensorflow').config.list_physical_devices('GPU')) > 0
        },
        'transformers': {
            'packages': ['transformers', 'tokenizers'],
            'import_names': ['transformers', 'tokenizers'],
            'gpu_check': None
        },
        'sklearn': {
            'packages': ['scikit-learn'],
            'import_names': ['sklearn'],
            'gpu_check': None
        },
        'data_science': {
            'packages': ['pandas', 'numpy', 'matplotlib', 'seaborn', 'jupyter'],
            'import_names': ['pandas', 'numpy', 'matplotlib', 'seaborn', 'jupyter'],
            'gpu_check': None
        },
        'cv': {
            'packages': ['opencv-python', 'pillow'],
            'import_names': ['cv2', 'PIL'],
            'gpu_check': None
        },
        'nlp': {
            'packages': ['spacy', 'nltk', 'datasets'],
            'import_names': ['spacy', 'nltk', 'datasets'],
            'gpu_check': None
        }
    }
    
    @staticmethod
    def check_package_installation(packages: List[str]) -> Dict[str, Dict[str, Any]]:
        """Check if packages are installed and get their versions."""
        results = {}
        
        for package in packages:
            try:
                # Try to import the package
                if package in ['cv2']:
                    # Special case for opencv
                    import cv2
                    version = cv2.__version__
                    installed = True
                elif package in ['PIL']:
                    # Special case for Pillow
                    import PIL
                    version = PIL.__version__
                    installed = True
                elif package in ['sklearn']:
                    # Special case for scikit-learn
                    import sklearn
                    version = sklearn.__version__
                    installed = True
                else:
                    module = importlib.import_module(package)
                    version = getattr(module, '__version__', 'unknown')
                    installed = True
                
                results[package] = {
                    'installed': installed,
                    'version': version,
                    'error': None
                }
            
            except ImportError as e:
                results[package] = {
                    'installed': False,
                    'version': None,
                    'error': str(e)
                }
        
        return results
    
    @staticmethod
    def check_framework_installation(framework: str) -> Dict[str, Any]:
        """Check specific ML framework installation."""
        if framework not in DependencyChecker.COMMON_PACKAGES:
            return {'error': f'Unknown framework: {framework}'}
        
        framework_info = DependencyChecker.COMMON_PACKAGES[framework]
        package_results = DependencyChecker.check_package_installation(framework_info['import_names'])
        
        # Check GPU support if applicable
        gpu_support = None
        if framework_info['gpu_check']:
            try:
                gpu_support = framework_info['gpu_check']()
            except Exception as e:
                gpu_support = f'Error checking GPU support: {e}'
        
        return {
            'framework': framework,
            'packages': package_results,
            'gpu_support': gpu_support,
            'all_installed': all(p['installed'] for p in package_results.values())
        }
    
    @staticmethod
    def generate_requirements(frameworks: List[str], include_versions: bool = True) -> str:
        """Generate requirements.txt content for specified frameworks."""
        requirements = []
        
        for framework in frameworks:
            if framework in DependencyChecker.COMMON_PACKAGES:
                packages = DependencyChecker.COMMON_PACKAGES[framework]['packages']
                
                if include_versions:
                    # Get current versions
                    package_info = DependencyChecker.check_package_installation(
                        DependencyChecker.COMMON_PACKAGES[framework]['import_names']
                    )
                    
                    for pkg_name in packages:
                        import_name = DependencyChecker.COMMON_PACKAGES[framework]['import_names'][packages.index(pkg_name)]
                        if import_name in package_info and package_info[import_name]['installed']:
                            version = package_info[import_name]['version']
                            if version != 'unknown':
                                requirements.append(f"{pkg_name}=={version}")
                            else:
                                requirements.append(pkg_name)
                        else:
                            requirements.append(pkg_name)
                else:
                    requirements.extend(packages)
        
        return '\n'.join(sorted(set(requirements)))


class PerformanceBenchmark:
    """Benchmark system performance for ML workloads."""
    
    @staticmethod
    def cpu_benchmark(duration: int = 5) -> Dict[str, float]:
        """Benchmark CPU performance."""
        import numpy as np
        import time
        
        print(f"Running CPU benchmark for {duration} seconds...")
        
        # Matrix multiplication benchmark
        start_time = time.time()
        operations = 0
        
        while time.time() - start_time < duration:
            a = np.random.rand(1000, 1000)
            b = np.random.rand(1000, 1000)
            np.dot(a, b)
            operations += 1
        
        elapsed = time.time() - start_time
        ops_per_second = operations / elapsed
        
        return {
            'operations': operations,
            'duration': elapsed,
            'ops_per_second': ops_per_second,
            'gflops_estimate': ops_per_second * 2.0  # Rough estimate
        }
    
    @staticmethod
    def gpu_benchmark(device: str = 'cuda', duration: int = 5) -> Dict[str, Any]:
        """Benchmark GPU performance."""
        try:
            import torch
            
            if not torch.cuda.is_available() and device == 'cuda':
                return {'error': 'CUDA not available'}
            
            if device == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                return {'error': 'MPS not available'}
            
            print(f"Running GPU benchmark on {device} for {duration} seconds...")
            
            device_obj = torch.device(device)
            
            # Warm up
            a = torch.randn(1000, 1000, device=device_obj)
            b = torch.randn(1000, 1000, device=device_obj)
            torch.matmul(a, b)
            torch.cuda.synchronize() if device == 'cuda' else None
            
            # Benchmark
            start_time = time.time()
            operations = 0
            
            while time.time() - start_time < duration:
                a = torch.randn(1000, 1000, device=device_obj)
                b = torch.randn(1000, 1000, device=device_obj)
                torch.matmul(a, b)
                operations += 1
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            elapsed = time.time() - start_time
            ops_per_second = operations / elapsed
            
            return {
                'device': device,
                'operations': operations,
                'duration': elapsed,
                'ops_per_second': ops_per_second,
                'gflops_estimate': ops_per_second * 2.0
            }
        
        except ImportError:
            return {'error': 'PyTorch not available'}
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def memory_benchmark() -> Dict[str, Any]:
        """Benchmark memory performance."""
        import numpy as np
        import time
        
        print("Running memory benchmark...")
        
        # Memory allocation/deallocation benchmark
        start_time = time.time()
        
        # Large array operations
        for _ in range(10):
            large_array = np.random.rand(10000, 1000)
            result = np.sum(large_array, axis=1)
            del large_array, result
        
        elapsed = time.time() - start_time
        
        return {
            'memory_ops_duration': elapsed,
            'memory_bandwidth_estimate': (10 * 10000 * 1000 * 8) / elapsed / 1e9  # GB/s estimate
        }


class TrainingEnvironmentSetup:
    """Sets up optimal training environment."""
    
    @staticmethod
    def setup_logging(log_dir: str = './logs', level: str = 'INFO') -> Dict[str, str]:
        """Setup comprehensive logging configuration."""
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Setup file handlers
        file_handler = logging.FileHandler(log_path / 'training.log')
        file_handler.setFormatter(detailed_formatter)
        
        error_handler = logging.FileHandler(log_path / 'errors.log')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(simple_formatter)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, level.upper()))
        root_logger.addHandler(file_handler)
        root_logger.addHandler(error_handler)
        root_logger.addHandler(console_handler)
        
        return {
            'log_dir': str(log_path),
            'training_log': str(log_path / 'training.log'),
            'error_log': str(log_path / 'errors.log')
        }
    
    @staticmethod
    def optimize_environment_variables() -> Dict[str, str]:
        """Set optimal environment variables for training."""
        env_vars = {}
        
        # PyTorch optimizations
        env_vars['OMP_NUM_THREADS'] = str(os.cpu_count())
        env_vars['MKL_NUM_THREADS'] = str(os.cpu_count())
        env_vars['NUMEXPR_NUM_THREADS'] = str(os.cpu_count())
        
        # CUDA optimizations
        env_vars['CUDA_LAUNCH_BLOCKING'] = '0'  # Async CUDA kernel launches
        env_vars['TORCH_CUDNN_V8_API_ENABLED'] = '1'  # Enable cuDNN v8 API
        
        # Apply environment variables
        for key, value in env_vars.items():
            os.environ[key] = value
        
        return env_vars
    
    @staticmethod
    def create_training_script_template(framework: str = 'pytorch') -> str:
        """Create a basic training script template."""
        if framework == 'pytorch':
            return '''#!/usr/bin/env python3
"""
PyTorch Training Script Template
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from pathlib import Path


def setup_device():
    """Setup optimal device for training."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple MPS")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            logging.info(f'Batch {batch_idx}, Loss: {loss.item():.6f}')
    
    return total_loss / len(dataloader)


def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Setup device
    device = setup_device()
    
    # TODO: Initialize model, data, optimizer, etc.
    # model = YourModel().to(device)
    # optimizer = optim.Adam(model.parameters())
    # criterion = nn.CrossEntropyLoss()
    
    print("Training script template ready!")


if __name__ == '__main__':
    main()
'''
        
        return "# Training script template not available for this framework"


def main():
    parser = argparse.ArgumentParser(description="ML Training Environment Setup")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Check command
    check_parser = subparsers.add_parser('check', help='Check system capabilities')
    check_parser.add_argument('--verbose', action='store_true', help='Detailed output')
    check_parser.add_argument('--gpu-only', action='store_true', help='Check GPU only')
    check_parser.add_argument('--framework', choices=['pytorch', 'tensorflow', 'all'], 
                             default='all', help='Check specific framework')

    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup training environment')
    setup_parser.add_argument('--framework', choices=['pytorch', 'tensorflow'], 
                             default='pytorch', help='ML framework to setup for')
    setup_parser.add_argument('--log-dir', default='./logs', help='Logging directory')
    setup_parser.add_argument('--create-template', action='store_true', 
                             help='Create training script template')
    setup_parser.add_argument('--optimize-env', action='store_true', 
                             help='Set optimal environment variables')

    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark system performance')
    benchmark_parser.add_argument('--cpu', action='store_true', help='Benchmark CPU')
    benchmark_parser.add_argument('--gpu', action='store_true', help='Benchmark GPU')
    benchmark_parser.add_argument('--memory', action='store_true', help='Benchmark memory')
    benchmark_parser.add_argument('--duration', type=int, default=5, help='Benchmark duration (seconds)')
    benchmark_parser.add_argument('--quick', action='store_true', help='Quick benchmark (all components)')

    # Install command
    install_parser = subparsers.add_parser('install', help='Install dependencies')
    install_parser.add_argument('--requirements', help='Requirements file to install')
    install_parser.add_argument('--framework', choices=['pytorch', 'tensorflow', 'sklearn', 'all'],
                               help='Install framework dependencies')
    install_parser.add_argument('--gpu', action='store_true', help='Install GPU-enabled versions')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == 'check':
            print("🔍 Checking system capabilities...\n")
            
            if not args.gpu_only:
                # System info
                print("📊 System Information:")
                system_info = SystemInfo.get_system_info()
                print(f"  OS: {system_info['platform']['system']} {system_info['platform']['release']}")
                print(f"  Python: {system_info['platform']['python_version']}")
                print(f"  CPU: {system_info['hardware']['cpu_count_logical']} logical cores")
                print(f"  Memory: {system_info['hardware']['memory_total_gb']:.1f} GB total, "
                      f"{system_info['hardware']['memory_available_gb']:.1f} GB available")
            
            # GPU info
            print("\n🎮 GPU Information:")
            gpu_info = SystemInfo.check_gpu_availability()
            if gpu_info['cuda_available']:
                print(f"  ✅ CUDA available (v{gpu_info['cuda_version']})")
                print(f"  GPUs: {gpu_info['gpu_count']}")
                for gpu in gpu_info['gpus']:
                    print(f"    - {gpu['name']} ({gpu['memory_total_gb']:.1f} GB)")
            elif gpu_info['mps_available']:
                print("  ✅ Apple MPS available")
            elif gpu_info['rocm_available']:
                print("  ✅ AMD ROCm available")
            else:
                print("  ❌ No GPU acceleration available")
            
            # Framework check
            if not args.gpu_only:
                print("\n📦 Framework Status:")
                frameworks = ['pytorch', 'tensorflow'] if args.framework == 'all' else [args.framework]
                for framework in frameworks:
                    if framework != 'all':
                        result = DependencyChecker.check_framework_installation(framework)
                        status = "✅" if result['all_installed'] else "❌"
                        print(f"  {status} {framework.title()}: {'Installed' if result['all_installed'] else 'Missing packages'}")
                        
                        if args.verbose and 'packages' in result:
                            for pkg, info in result['packages'].items():
                                pkg_status = "✅" if info['installed'] else "❌"
                                version = f" (v{info['version']})" if info['version'] else ""
                                print(f"    {pkg_status} {pkg}{version}")

        elif args.command == 'setup':
            print("🚀 Setting up training environment...\n")
            
            if args.optimize_env:
                print("⚙️  Optimizing environment variables...")
                env_vars = TrainingEnvironmentSetup.optimize_environment_variables()
                for key, value in env_vars.items():
                    print(f"  {key}={value}")
            
            if args.log_dir:
                print(f"\n📝 Setting up logging in {args.log_dir}...")
                log_config = TrainingEnvironmentSetup.setup_logging(args.log_dir)
                print(f"  Training log: {log_config['training_log']}")
                print(f"  Error log: {log_config['error_log']}")
            
            if args.create_template:
                print(f"\n📄 Creating {args.framework} training script template...")
                template = TrainingEnvironmentSetup.create_training_script_template(args.framework)
                template_file = f"train_{args.framework}_template.py"
                with open(template_file, 'w') as f:
                    f.write(template)
                print(f"  Template saved: {template_file}")
            
            print("\n✅ Environment setup complete!")

        elif args.command == 'benchmark':
            print("🏃 Running performance benchmarks...\n")
            
            if args.quick or args.cpu:
                cpu_result = PerformanceBenchmark.cpu_benchmark(args.duration)
                print("💻 CPU Benchmark:")
                print(f"  Operations: {cpu_result['operations']}")
                print(f"  Ops/second: {cpu_result['ops_per_second']:.2f}")
                print(f"  Estimated GFLOPS: {cpu_result['gflops_estimate']:.2f}")
            
            if args.quick or args.gpu:
                print("\n🎮 GPU Benchmark:")
                # Try CUDA first, then MPS
                for device in ['cuda', 'mps']:
                    gpu_result = PerformanceBenchmark.gpu_benchmark(device, args.duration)
                    if 'error' not in gpu_result:
                        print(f"  Device: {device}")
                        print(f"  Operations: {gpu_result['operations']}")
                        print(f"  Ops/second: {gpu_result['ops_per_second']:.2f}")
                        print(f"  Estimated GFLOPS: {gpu_result['gflops_estimate']:.2f}")
                        break
                else:
                    print("  ❌ No GPU available for benchmarking")
            
            if args.quick or args.memory:
                print("\n💾 Memory Benchmark:")
                mem_result = PerformanceBenchmark.memory_benchmark()
                print(f"  Duration: {mem_result['memory_ops_duration']:.2f}s")
                print(f"  Estimated bandwidth: {mem_result['memory_bandwidth_estimate']:.2f} GB/s")

        elif args.command == 'install':
            print("📦 Installing dependencies...\n")
            
            if args.requirements:
                print(f"Installing from {args.requirements}...")
                subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', args.requirements])
            
            if args.framework:
                frameworks = ['pytorch', 'tensorflow', 'sklearn'] if args.framework == 'all' else [args.framework]
                
                for framework in frameworks:
                    if framework in DependencyChecker.COMMON_PACKAGES:
                        packages = DependencyChecker.COMMON_PACKAGES[framework]['packages']
                        print(f"Installing {framework} packages: {', '.join(packages)}")
                        
                        for package in packages:
                            subprocess.run([sys.executable, '-m', 'pip', 'install', package])

    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()