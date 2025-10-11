#!/usr/bin/env python3
"""
train_model.py

Generic PyTorch/Transformers training loop with configuration arguments.
Simple and focused training utility.
"""

import argparse
import json
import os
import math
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, PreTrainedTokenizer
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class SimpleDataset(Dataset):
    """Simple dataset for text classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx] if self.labels else 0
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_data(file_path: str, text_key: str = 'text', label_key: str = 'label'):
    """Load training data from file"""
    path = Path(file_path)
    
    if path.suffix.lower() == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif path.suffix.lower() == '.jsonl':
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    texts = []
    labels = []
    
    for item in data:
        if isinstance(item, dict):
            texts.append(item.get(text_key, str(item)))
            labels.append(item.get(label_key, 0))
        else:
            texts.append(str(item))
            labels.append(0)
    
    return texts, labels


def create_model(model_name: str, num_labels: int = 2, task_type: str = 'classification'):
    """Create model based on configuration"""
    if not HAS_TRANSFORMERS:
        raise RuntimeError("transformers library required for model training")
    
    if task_type == 'classification':
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        )
    else:
        model = AutoModel.from_pretrained(model_name)
    
    return model


def train_with_transformers(config: Dict[str, Any]):
    """Train model using Transformers Trainer"""
    
    # Load data
    print(f"Loading training data from {config['train_file']}...")
    train_texts, train_labels = load_data(
        config['train_file'], 
        config.get('text_key', 'text'),
        config.get('label_key', 'label')
    )
    
    val_texts, val_labels = None, None
    if config.get('val_file'):
        print(f"Loading validation data from {config['val_file']}...")
        val_texts, val_labels = load_data(
            config['val_file'],
            config.get('text_key', 'text'),
            config.get('label_key', 'label')
        )
    
    # Create tokenizer and model
    print(f"Loading model: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    
    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    num_labels = len(set(train_labels)) if train_labels else 2
    model = create_model(config['model_name'], num_labels, config.get('task_type', 'classification'))
    
    # Create datasets
    train_dataset = SimpleDataset(
        train_texts, train_labels, tokenizer, 
        config.get('max_length', 512)
    )
    
    val_dataset = None
    if val_texts:
        val_dataset = SimpleDataset(
            val_texts, val_labels, tokenizer,
            config.get('max_length', 512)
        )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.get('output_dir', './results'),
        num_train_epochs=config.get('epochs', 3),
        per_device_train_batch_size=config.get('batch_size', 8),
        per_device_eval_batch_size=config.get('eval_batch_size', 8),
        learning_rate=config.get('learning_rate', 2e-5),
        weight_decay=config.get('weight_decay', 0.01),
        logging_dir=config.get('logging_dir', './logs'),
        logging_steps=config.get('logging_steps', 500),
        evaluation_strategy="epoch" if val_dataset else "no",
        save_strategy="epoch",
        save_total_limit=config.get('save_total_limit', 2),
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="eval_loss" if val_dataset else None,
        greater_is_better=False,
        seed=config.get('seed', 42),
        fp16=config.get('fp16', False),
        dataloader_num_workers=config.get('num_workers', 0),
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save model
    print(f"Saving model to {config.get('output_dir', './results')}")
    trainer.save_model()
    tokenizer.save_pretrained(config.get('output_dir', './results'))
    
    # Evaluate on validation set
    if val_dataset:
        print("Evaluating on validation set...")
        eval_results = trainer.evaluate()
        print(f"Validation results: {eval_results}")
        
        # Save evaluation results
        with open(Path(config.get('output_dir', './results')) / 'eval_results.json', 'w') as f:
            json.dump(eval_results, f, indent=2)


def train_with_pytorch(config: Dict[str, Any]):
    """Basic PyTorch training loop"""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required for training")
    
    print("Basic PyTorch training not implemented in this simple version")
    print("Use --framework transformers for full functionality")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from file"""
    with open(config_path, 'r') as f:
        if config_path.endswith('.json'):
            return json.load(f)
        else:
            # Assume YAML (simplified)
            import yaml
            return yaml.safe_load(f)


def create_sample_config():
    """Create a sample training configuration"""
    config = {
        "model_name": "bert-base-uncased",
        "task_type": "classification",
        "train_file": "train.jsonl",
        "val_file": "val.jsonl",
        "text_key": "text",
        "label_key": "label",
        "output_dir": "./model_output",
        "epochs": 3,
        "batch_size": 8,
        "learning_rate": 2e-5,
        "max_length": 512,
        "seed": 42,
        "fp16": False,
        "logging_steps": 100,
        "save_total_limit": 2
    }
    
    with open('train_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Sample configuration saved to train_config.json")


def main():
    parser = argparse.ArgumentParser(description="Generic model training utility")
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--create-config', action='store_true',
                       help='Create sample configuration file')
    
    # Quick training options (override config)
    parser.add_argument('--model-name', default='bert-base-uncased',
                       help='Model name from HuggingFace')
    parser.add_argument('--train-file', help='Training data file')
    parser.add_argument('--val-file', help='Validation data file')
    parser.add_argument('--output-dir', default='./results', help='Output directory')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max-length', type=int, default=512, help='Max sequence length')
    parser.add_argument('--framework', choices=['transformers', 'pytorch'], 
                       default='transformers', help='Training framework')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    if args.create_config:
        create_sample_config()
        return
    
    # Load or create configuration
    if args.config:
        config = load_config(args.config)
    else:
        if not args.train_file:
            print("Error: --train-file required when not using config file")
            return
        
        config = {
            'model_name': args.model_name,
            'train_file': args.train_file,
            'val_file': args.val_file,
            'output_dir': args.output_dir,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'max_length': args.max_length,
            'seed': args.seed,
        }
    
    # Set random seed
    if HAS_TORCH:
        torch.manual_seed(config.get('seed', 42))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.get('seed', 42))
    
    # Create output directory
    os.makedirs(config.get('output_dir', './results'), exist_ok=True)
    
    # Save final config
    with open(Path(config.get('output_dir', './results')) / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Train model
    if args.framework == 'transformers':
        train_with_transformers(config)
    else:
        train_with_pytorch(config)
    
    print("Training completed!")


if __name__ == "__main__":
    main()