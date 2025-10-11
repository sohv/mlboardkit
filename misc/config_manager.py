#!/usr/bin/env python3
"""
Configuration Manager for ML/AI Training Workflows

Supports YAML/JSON configs with validation, environment overrides, and templates.
Useful for managing hyperparameters, model configs, and training settings.

Usage:
    python3 config_manager.py create --template finetuning --output config.yaml
    python3 config_manager.py validate config.yaml
    python3 config_manager.py merge base.yaml override.yaml --output final.yaml
    python3 config_manager.py set config.yaml training.learning_rate 0.001
"""

import argparse
import json
import yaml
import os
import sys
from pathlib import Path
from typing import Dict, Any, Union, List
import copy


class ConfigManager:
    """Manages configuration files for ML/AI workflows."""
    
    TEMPLATES = {
        "finetuning": {
            "model": {
                "name": "gpt-3.5-turbo",
                "checkpoint": None,
                "max_length": 2048,
                "architecture": {
                    "hidden_size": 768,
                    "num_layers": 12,
                    "num_heads": 12,
                    "dropout": 0.1
                }
            },
            "training": {
                "learning_rate": 5e-5,
                "batch_size": 16,
                "num_epochs": 3,
                "warmup_steps": 100,
                "weight_decay": 0.01,
                "gradient_accumulation_steps": 1,
                "max_grad_norm": 1.0,
                "scheduler": "cosine",
                "optimizer": "adamw"
            },
            "data": {
                "train_file": "train.jsonl",
                "val_file": "val.jsonl",
                "test_file": "test.jsonl",
                "max_train_samples": None,
                "preprocessing": {
                    "truncation": True,
                    "padding": "max_length",
                    "remove_unused_columns": True
                }
            },
            "logging": {
                "output_dir": "./outputs",
                "logging_dir": "./logs",
                "save_steps": 500,
                "eval_steps": 500,
                "logging_steps": 100,
                "save_total_limit": 2,
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False
            },
            "environment": {
                "seed": 42,
                "fp16": True,
                "dataloader_num_workers": 4,
                "remove_unused_columns": True,
                "push_to_hub": False,
                "hub_model_id": None
            }
        },
        
        "classification": {
            "model": {
                "name": "bert-base-uncased",
                "num_labels": 2,
                "hidden_dropout_prob": 0.1,
                "attention_probs_dropout_prob": 0.1
            },
            "training": {
                "learning_rate": 2e-5,
                "batch_size": 32,
                "num_epochs": 3,
                "weight_decay": 0.01,
                "warmup_ratio": 0.1
            },
            "data": {
                "text_column": "text",
                "label_column": "label",
                "max_length": 512,
                "train_split": 0.8,
                "val_split": 0.1,
                "test_split": 0.1
            },
            "evaluation": {
                "metrics": ["accuracy", "f1", "precision", "recall"],
                "eval_strategy": "epoch"
            }
        },
        
        "generation": {
            "model": {
                "name": "gpt2",
                "max_length": 1024,
                "pad_token_id": 50256
            },
            "generation": {
                "max_new_tokens": 100,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "do_sample": True,
                "num_return_sequences": 1,
                "repetition_penalty": 1.1
            },
            "training": {
                "learning_rate": 5e-5,
                "batch_size": 8,
                "gradient_accumulation_steps": 4,
                "num_epochs": 5
            }
        }
    }

    @staticmethod
    def load_config(file_path: str) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            elif path.suffix.lower() == '.json':
                return json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")

    @staticmethod
    def save_config(config: Dict[str, Any], file_path: str, format: str = None):
        """Save configuration to YAML or JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format is None:
            format = 'yaml' if path.suffix.lower() in ['.yaml', '.yml'] else 'json'
        
        with open(path, 'w', encoding='utf-8') as f:
            if format.lower() == 'yaml':
                yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)
            else:
                json.dump(config, f, indent=2, ensure_ascii=False)

    @staticmethod
    def create_from_template(template_name: str) -> Dict[str, Any]:
        """Create config from predefined template."""
        if template_name not in ConfigManager.TEMPLATES:
            available = ', '.join(ConfigManager.TEMPLATES.keys())
            raise ValueError(f"Unknown template: {template_name}. Available: {available}")
        
        return copy.deepcopy(ConfigManager.TEMPLATES[template_name])

    @staticmethod
    def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge two configurations."""
        result = copy.deepcopy(base_config)
        
        def _merge_dict(base: Dict, override: Dict):
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    _merge_dict(base[key], value)
                else:
                    base[key] = value
        
        _merge_dict(result, override_config)
        return result

    @staticmethod
    def get_nested_value(config: Dict[str, Any], key_path: str) -> Any:
        """Get value from nested config using dot notation (e.g., 'training.learning_rate')."""
        keys = key_path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value

    @staticmethod
    def set_nested_value(config: Dict[str, Any], key_path: str, value: Any):
        """Set value in nested config using dot notation."""
        keys = key_path.split('.')
        current = config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Convert string values to appropriate types
        if isinstance(value, str):
            # Try to convert to number or boolean
            if value.lower() in ['true', 'false']:
                value = value.lower() == 'true'
            elif value.lower() == 'null':
                value = None
            else:
                try:
                    # Try int first, then float
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    # Keep as string
                    pass
        
        current[keys[-1]] = value

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Basic validation rules
        if 'training' in config:
            training = config['training']
            
            # Learning rate validation
            if 'learning_rate' in training:
                lr = training['learning_rate']
                if not isinstance(lr, (int, float)) or lr <= 0 or lr > 1:
                    issues.append("Learning rate should be a positive number <= 1")
            
            # Batch size validation
            if 'batch_size' in training:
                bs = training['batch_size']
                if not isinstance(bs, int) or bs <= 0:
                    issues.append("Batch size should be a positive integer")
            
            # Epochs validation
            if 'num_epochs' in training:
                epochs = training['num_epochs']
                if not isinstance(epochs, int) or epochs <= 0:
                    issues.append("Number of epochs should be a positive integer")
        
        # File existence checks
        if 'data' in config:
            data = config['data']
            for file_key in ['train_file', 'val_file', 'test_file']:
                if file_key in data and data[file_key]:
                    file_path = data[file_key]
                    if isinstance(file_path, str) and not Path(file_path).exists():
                        issues.append(f"Data file not found: {file_path}")
        
        return issues

    @staticmethod
    def apply_env_overrides(config: Dict[str, Any], env_prefix: str = "CONFIG_") -> Dict[str, Any]:
        """Apply environment variable overrides to config."""
        result = copy.deepcopy(config)
        
        for env_var, value in os.environ.items():
            if env_var.startswith(env_prefix):
                # Convert CONFIG_TRAINING_LEARNING_RATE to training.learning_rate
                key_path = env_var[len(env_prefix):].lower().replace('_', '.')
                try:
                    ConfigManager.set_nested_value(result, key_path, value)
                except Exception as e:
                    print(f"Warning: Could not apply env override {env_var}: {e}")
        
        return result


def main():
    parser = argparse.ArgumentParser(description="Configuration Manager for ML/AI workflows")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Create command
    create_parser = subparsers.add_parser('create', help='Create config from template')
    create_parser.add_argument('--template', choices=list(ConfigManager.TEMPLATES.keys()),
                              required=True, help='Template to use')
    create_parser.add_argument('--output', required=True, help='Output file path')
    create_parser.add_argument('--format', choices=['yaml', 'json'], help='Output format')

    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate configuration file')
    validate_parser.add_argument('config', help='Configuration file to validate')

    # Merge command
    merge_parser = subparsers.add_parser('merge', help='Merge configuration files')
    merge_parser.add_argument('base', help='Base configuration file')
    merge_parser.add_argument('override', help='Override configuration file')
    merge_parser.add_argument('--output', required=True, help='Output file path')

    # Set command
    set_parser = subparsers.add_parser('set', help='Set configuration value')
    set_parser.add_argument('config', help='Configuration file to modify')
    set_parser.add_argument('key', help='Key path (e.g., training.learning_rate)')
    set_parser.add_argument('value', help='Value to set')

    # Get command
    get_parser = subparsers.add_parser('get', help='Get configuration value')
    get_parser.add_argument('config', help='Configuration file')
    get_parser.add_argument('key', help='Key path (e.g., training.learning_rate)')

    # List templates command
    list_parser = subparsers.add_parser('list-templates', help='List available templates')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == 'create':
            config = ConfigManager.create_from_template(args.template)
            ConfigManager.save_config(config, args.output, args.format)
            print(f"Created configuration file: {args.output}")

        elif args.command == 'validate':
            config = ConfigManager.load_config(args.config)
            issues = ConfigManager.validate_config(config)
            
            if issues:
                print("Configuration issues found:")
                for issue in issues:
                    print(f"  - {issue}")
                sys.exit(1)
            else:
                print("Configuration is valid!")

        elif args.command == 'merge':
            base_config = ConfigManager.load_config(args.base)
            override_config = ConfigManager.load_config(args.override)
            merged_config = ConfigManager.merge_configs(base_config, override_config)
            ConfigManager.save_config(merged_config, args.output)
            print(f"Merged configurations saved to: {args.output}")

        elif args.command == 'set':
            config = ConfigManager.load_config(args.config)
            ConfigManager.set_nested_value(config, args.key, args.value)
            ConfigManager.save_config(config, args.config)
            print(f"Set {args.key} = {args.value}")

        elif args.command == 'get':
            config = ConfigManager.load_config(args.config)
            value = ConfigManager.get_nested_value(config, args.key)
            if value is not None:
                print(value)
            else:
                print(f"Key not found: {args.key}")
                sys.exit(1)

        elif args.command == 'list-templates':
            print("Available templates:")
            for name, template in ConfigManager.TEMPLATES.items():
                print(f"  {name}:")
                for section in template.keys():
                    print(f"    - {section}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()