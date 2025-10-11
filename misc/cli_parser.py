"""
Supports nested configurations, type validation, automatic help generation,
config file integration, and environment variable overrides.

Usage:
    python3 cli_parser.py --help
    python3 cli_parser.py --config config.yaml --model.name bert-base-uncased
    python3 cli_parser.py train --learning-rate 0.001 --batch-size 32
"""

import argparse
import json
import yaml
import os
import sys
from pathlib import Path
from typing import Dict, Any, Union, List, Optional, Type
from dataclasses import dataclass, field, fields
from enum import Enum


class OptimizerType(Enum):
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"


class SchedulerType(Enum):
    LINEAR = "linear"
    COSINE = "cosine"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    name: str = "bert-base-uncased"
    checkpoint: Optional[str] = None
    max_length: int = 512
    num_labels: int = 2
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    cache_dir: Optional[str] = None
    revision: str = "main"
    use_auth_token: bool = False


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    learning_rate: float = 5e-5
    batch_size: int = 16
    num_epochs: int = 3
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    optimizer: OptimizerType = OptimizerType.ADAMW
    scheduler: SchedulerType = SchedulerType.LINEAR
    lr_scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True
    skip_memory_metrics: bool = True
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None


@dataclass
class DataConfig:
    """Data configuration parameters."""
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    test_file: Optional[str] = None
    text_column: str = "text"
    label_column: str = "label"
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    max_predict_samples: Optional[int] = None
    preprocessing_num_workers: int = 1
    overwrite_cache: bool = False
    pad_to_max_length: bool = False
    train_split_ratio: float = 0.8
    validation_split_ratio: float = 0.1
    test_split_ratio: float = 0.1


@dataclass
class LoggingConfig:
    """Logging and output configuration."""
    output_dir: str = "./outputs"
    logging_dir: Optional[str] = None
    logging_strategy: str = "steps"
    logging_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 500
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    ignore_data_skip: bool = False
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    run_name: Optional[str] = None
    disable_tqdm: bool = False


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    seed: int = 42
    local_rank: int = -1
    debug: bool = False
    resume_from_checkpoint: Optional[str] = None


class AdvancedArgumentParser:
    """Advanced argument parser with dataclass integration."""
    
    def __init__(self, config_class: Type = ExperimentConfig, 
                 description: str = "ML/AI Experiment Runner",
                 add_config_args: bool = True):
        self.config_class = config_class
        self.parser = argparse.ArgumentParser(
            description=description,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        self.config_group = None
        
        if add_config_args:
            self._add_config_arguments()
        
        self._add_dataclass_arguments(config_class)
    
    def _add_config_arguments(self):
        """Add configuration file related arguments."""
        self.config_group = self.parser.add_argument_group('Configuration')
        self.config_group.add_argument(
            '--config', '--config-file', type=str,
            help='Path to configuration file (YAML or JSON)'
        )
        self.config_group.add_argument(
            '--config-overrides', type=str, nargs='*',
            help='Override config values using key=value format'
        )
        self.config_group.add_argument(
            '--save-config', type=str,
            help='Save final configuration to file'
        )
        self.config_group.add_argument(
            '--print-config', action='store_true',
            help='Print final configuration and exit'
        )
    
    def _add_dataclass_arguments(self, config_class: Type, prefix: str = ""):
        """Recursively add arguments from dataclass fields."""
        for field_info in fields(config_class):
            field_name = field_info.name
            field_type = field_info.type
            field_default = field_info.default if field_info.default != field_info.default_factory else field_info.default_factory()
            
            # Create argument name with prefix
            arg_name = f"--{prefix}{field_name.replace('_', '-')}"
            
            # Handle nested dataclasses
            if hasattr(field_type, '__dataclass_fields__'):
                group = self.parser.add_argument_group(f'{prefix}{field_name}'.title())
                self._add_dataclass_arguments(field_type, f"{prefix}{field_name}.")
                continue
            
            # Handle Optional types
            origin = getattr(field_type, '__origin__', None)
            if origin is Union:
                args = getattr(field_type, '__args__', ())
                if len(args) == 2 and type(None) in args:
                    field_type = args[0] if args[1] is type(None) else args[1]
            
            # Handle Enum types
            if isinstance(field_default, Enum):
                choices = [e.value for e in field_type]
                self.parser.add_argument(
                    arg_name,
                    type=str,
                    default=field_default.value,
                    choices=choices,
                    help=f'Choose from: {", ".join(choices)}'
                )
                continue
            
            # Handle List types
            if origin is list or (hasattr(field_type, '__origin__') and field_type.__origin__ is list):
                list_type = getattr(field_type, '__args__', [str])[0]
                self.parser.add_argument(
                    arg_name,
                    type=list_type,
                    nargs='*',
                    default=field_default,
                    help=f'List of {list_type.__name__} values'
                )
                continue
            
            # Handle Dict types
            if origin is dict or (hasattr(field_type, '__origin__') and field_type.__origin__ is dict):
                self.parser.add_argument(
                    arg_name,
                    type=str,
                    nargs='*',
                    default=None,
                    help='Dictionary in key=value format'
                )
                continue
            
            # Handle basic types
            if field_type == bool:
                if field_default:
                    self.parser.add_argument(
                        arg_name,
                        action='store_false',
                        help=f'Disable {field_name.replace("_", " ")}'
                    )
                    # Add positive version too
                    self.parser.add_argument(
                        f'--no-{prefix}{field_name.replace("_", "-")}',
                        dest=f'{prefix}{field_name}',
                        action='store_false',
                        help=argparse.SUPPRESS
                    )
                else:
                    self.parser.add_argument(
                        arg_name,
                        action='store_true',
                        help=f'Enable {field_name.replace("_", " ")}'
                    )
            else:
                self.parser.add_argument(
                    arg_name,
                    type=field_type,
                    default=field_default,
                    help=f'{field_name.replace("_", " ").title()}'
                )
    
    def _load_config_file(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f) or {}
            elif path.suffix.lower() == '.json':
                return json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {path.suffix}")
    
    def _apply_config_overrides(self, config: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
        """Apply configuration overrides from command line."""
        for override in overrides:
            if '=' not in override:
                raise ValueError(f"Invalid override format: {override}. Use key=value")
            
            key, value = override.split('=', 1)
            keys = key.split('.')
            
            # Navigate to the correct nested location
            current = config
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            # Convert value to appropriate type
            final_key = keys[-1]
            if value.lower() in ['true', 'false']:
                current[final_key] = value.lower() == 'true'
            elif value.lower() == 'null':
                current[final_key] = None
            else:
                try:
                    current[final_key] = float(value) if '.' in value else int(value)
                except ValueError:
                    current[final_key] = value
        
        return config
    
    def _args_to_config_dict(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Convert parsed arguments to nested configuration dictionary."""
        config = {}
        args_dict = vars(args)
        
        for key, value in args_dict.items():
            if key in ['config', 'config_overrides', 'save_config', 'print_config']:
                continue
            
            if '.' in key:
                keys = key.split('.')
                current = config
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
            else:
                config[key] = value
        
        return config
    
    def _dict_to_dataclass(self, config_dict: Dict[str, Any], config_class: Type):
        """Convert dictionary to dataclass instance."""
        field_values = {}
        
        for field_info in fields(config_class):
            field_name = field_info.name
            field_type = field_info.type
            
            if field_name in config_dict:
                value = config_dict[field_name]
                
                # Handle nested dataclasses
                if hasattr(field_type, '__dataclass_fields__'):
                    if isinstance(value, dict):
                        value = self._dict_to_dataclass(value, field_type)
                
                # Handle Enum types
                elif hasattr(field_type, '__members__'):
                    if isinstance(value, str):
                        value = field_type(value)
                
                field_values[field_name] = value
        
        return config_class(**field_values)
    
    def parse_args(self, args: Optional[List[str]] = None) -> Any:
        """Parse arguments and return configuration object."""
        parsed_args = self.parser.parse_args(args)
        
        # Start with default configuration
        config_dict = {}
        
        # Load config file if provided
        if parsed_args.config:
            config_dict = self._load_config_file(parsed_args.config)
        
        # Apply command line arguments
        args_config = self._args_to_config_dict(parsed_args)
        self._merge_configs(config_dict, args_config)
        
        # Apply config overrides
        if parsed_args.config_overrides:
            config_dict = self._apply_config_overrides(config_dict, parsed_args.config_overrides)
        
        # Apply environment variable overrides
        config_dict = self._apply_env_overrides(config_dict)
        
        # Convert to dataclass
        config = self._dict_to_dataclass(config_dict, self.config_class)
        
        # Handle special flags
        if parsed_args.print_config:
            print(yaml.dump(config_dict, default_flow_style=False, indent=2))
            sys.exit(0)
        
        if parsed_args.save_config:
            with open(parsed_args.save_config, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            print(f"Configuration saved to: {parsed_args.save_config}")
        
        return config
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]):
        """Recursively merge configuration dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value
    
    def _apply_env_overrides(self, config: Dict[str, Any], prefix: str = "ML_") -> Dict[str, Any]:
        """Apply environment variable overrides."""
        for env_var, value in os.environ.items():
            if env_var.startswith(prefix):
                # Convert ML_TRAINING_LEARNING_RATE to training.learning_rate
                key_path = env_var[len(prefix):].lower().replace('_', '.')
                keys = key_path.split('.')
                
                current = config
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                
                # Type conversion
                final_key = keys[-1]
                if value.lower() in ['true', 'false']:
                    current[final_key] = value.lower() == 'true'
                elif value.lower() == 'null':
                    current[final_key] = None
                else:
                    try:
                        current[final_key] = float(value) if '.' in value else int(value)
                    except ValueError:
                        current[final_key] = value
        
        return config


def create_training_parser() -> AdvancedArgumentParser:
    """Create a parser specifically for training workflows."""
    parser = AdvancedArgumentParser(
        config_class=ExperimentConfig,
        description="Train ML/AI models with advanced configuration support"
    )
    
    # Add subcommands
    subparsers = parser.parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--dry-run', action='store_true', 
                             help='Print configuration without training')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a model')
    eval_parser.add_argument('--model-path', required=True, 
                            help='Path to trained model')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('--model-path', required=True, 
                               help='Path to trained model')
    predict_parser.add_argument('--input-file', required=True, 
                               help='Input file for predictions')
    predict_parser.add_argument('--output-file', required=True, 
                               help='Output file for predictions')
    
    return parser


def main():
    """Example usage of the advanced argument parser."""
    parser = create_training_parser()
    
    try:
        config = parser.parse_args()
        
        print("Parsed Configuration:")
        print("=" * 50)
        
        # Pretty print the configuration
        config_dict = {
            'model': {
                'name': config.model.name,
                'max_length': config.model.max_length,
                'num_labels': config.model.num_labels
            },
            'training': {
                'learning_rate': config.training.learning_rate,
                'batch_size': config.training.batch_size,
                'num_epochs': config.training.num_epochs,
                'optimizer': config.training.optimizer.value,
                'scheduler': config.training.scheduler.value
            },
            'data': {
                'train_file': config.data.train_file,
                'validation_file': config.data.validation_file,
                'text_column': config.data.text_column,
                'label_column': config.data.label_column
            },
            'logging': {
                'output_dir': config.logging.output_dir,
                'logging_steps': config.logging.logging_steps,
                'save_steps': config.logging.save_steps
            }
        }
        
        print(yaml.dump(config_dict, default_flow_style=False, indent=2))
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()