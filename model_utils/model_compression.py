#!/usr/bin/env python3
"""
model_compression.py

Model compression techniques including pruning, quantization, and distillation.
Simple and focused compression utility.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.utils.prune as prune
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from transformers import (
        AutoModel, AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


def magnitude_pruning(model, amount: float = 0.2):
    """Apply magnitude-based pruning to model"""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required for pruning")
    
    parameters_to_prune = []
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            parameters_to_prune.append((module, 'weight'))
    
    # Apply global magnitude pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    
    # Make pruning permanent
    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)
    
    return model


def structured_pruning(model, amount: float = 0.2):
    """Apply structured pruning (remove entire channels/neurons)"""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required for pruning")
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Prune entire neurons based on L2 norm
            prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
            prune.remove(module, 'weight')
        elif isinstance(module, nn.Conv2d):
            # Prune entire channels
            prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
            prune.remove(module, 'weight')
    
    return model


def dynamic_quantization(model):
    """Apply dynamic quantization to model"""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required for quantization")
    
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {nn.Linear, nn.LSTM, nn.GRU}, 
        dtype=torch.qint8
    )
    
    return quantized_model


def static_quantization(model, calibration_data=None):
    """Apply static quantization with calibration"""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required for quantization")
    
    model.eval()
    
    # Set quantization config
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # Prepare model for quantization
    model_prepared = torch.quantization.prepare(model)
    
    # Calibrate with sample data if provided
    if calibration_data is not None:
        with torch.no_grad():
            for data in calibration_data:
                model_prepared(data)
    
    # Convert to quantized model
    quantized_model = torch.quantization.convert(model_prepared)
    
    return quantized_model


class DistillationTrainer:
    """Knowledge distillation trainer"""
    
    def __init__(self, teacher_model, student_model, temperature: float = 4.0, alpha: float = 0.7):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        
        # Freeze teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()
    
    def distillation_loss(self, student_outputs, teacher_outputs, labels):
        """Calculate distillation loss"""
        # Soft targets from teacher
        teacher_probs = torch.nn.functional.softmax(teacher_outputs / self.temperature, dim=1)
        student_log_probs = torch.nn.functional.log_softmax(student_outputs / self.temperature, dim=1)
        
        # KL divergence loss
        distillation_loss = torch.nn.functional.kl_div(
            student_log_probs, teacher_probs, reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Hard target loss
        hard_target_loss = torch.nn.functional.cross_entropy(student_outputs, labels)
        
        # Combined loss
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * hard_target_loss
        
        return total_loss
    
    def train_step(self, batch, optimizer):
        """Single training step"""
        inputs, labels = batch
        
        # Get teacher outputs
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)
            if hasattr(teacher_outputs, 'logits'):
                teacher_outputs = teacher_outputs.logits
        
        # Get student outputs
        student_outputs = self.student_model(inputs)
        if hasattr(student_outputs, 'logits'):
            student_outputs = student_outputs.logits
        
        # Calculate loss
        loss = self.distillation_loss(student_outputs, teacher_outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()


def get_model_size(model):
    """Calculate model size in MB"""
    if not HAS_TORCH:
        return 0
    
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def count_parameters(model):
    """Count total and trainable parameters"""
    if not HAS_TORCH:
        return 0, 0
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def compress_transformers_model(model_name: str, compression_type: str, 
                               output_dir: str, **kwargs):
    """Compress a transformers model"""
    if not HAS_TRANSFORMERS:
        raise RuntimeError("transformers library required")
    
    print(f"Loading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Get original model stats
    original_size = get_model_size(model)
    original_params, _ = count_parameters(model)
    
    print(f"Original model size: {original_size:.2f} MB")
    print(f"Original parameters: {original_params:,}")
    
    # Apply compression
    if compression_type == 'pruning':
        amount = kwargs.get('amount', 0.2)
        method = kwargs.get('method', 'magnitude')
        
        print(f"Applying {method} pruning with amount {amount}")
        if method == 'magnitude':
            model = magnitude_pruning(model, amount)
        elif method == 'structured':
            model = structured_pruning(model, amount)
    
    elif compression_type == 'quantization':
        method = kwargs.get('method', 'dynamic')
        
        print(f"Applying {method} quantization")
        if method == 'dynamic':
            model = dynamic_quantization(model)
        elif method == 'static':
            model = static_quantization(model)
    
    # Get compressed model stats
    compressed_size = get_model_size(model)
    compressed_params, _ = count_parameters(model)
    
    print(f"Compressed model size: {compressed_size:.2f} MB")
    print(f"Compressed parameters: {compressed_params:,}")
    print(f"Size reduction: {(1 - compressed_size/original_size)*100:.1f}%")
    print(f"Parameter reduction: {(1 - compressed_params/original_params)*100:.1f}%")
    
    # Save compressed model
    os.makedirs(output_dir, exist_ok=True)
    
    if compression_type == 'quantization':
        # Save quantized model
        torch.jit.save(torch.jit.script(model), os.path.join(output_dir, 'model.pt'))
    else:
        # Save as transformers model
        model.save_pretrained(output_dir)
    
    tokenizer.save_pretrained(output_dir)
    
    # Save compression info
    compression_info = {
        'original_model': model_name,
        'compression_type': compression_type,
        'compression_kwargs': kwargs,
        'original_size_mb': original_size,
        'compressed_size_mb': compressed_size,
        'original_params': original_params,
        'compressed_params': compressed_params,
        'size_reduction_pct': (1 - compressed_size/original_size)*100,
        'param_reduction_pct': (1 - compressed_params/original_params)*100
    }
    
    with open(os.path.join(output_dir, 'compression_info.json'), 'w') as f:
        json.dump(compression_info, f, indent=2)
    
    print(f"Compressed model saved to {output_dir}")
    return model


def create_smaller_model(original_model_name: str, reduction_factor: float = 0.5):
    """Create a smaller model for distillation"""
    if not HAS_TRANSFORMERS:
        raise RuntimeError("transformers library required")
    
    # Load original model config
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(original_model_name)
    
    # Reduce model dimensions
    config.hidden_size = int(config.hidden_size * reduction_factor)
    config.intermediate_size = int(config.intermediate_size * reduction_factor)
    config.num_attention_heads = max(1, int(config.num_attention_heads * reduction_factor))
    
    # Create smaller model
    model = AutoModelForSequenceClassification.from_config(config)
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Model compression utility")
    parser.add_argument('model_name', help='Model name or path')
    parser.add_argument('--compression', choices=['pruning', 'quantization', 'distillation'],
                       required=True, help='Compression method')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    
    # Pruning options
    parser.add_argument('--pruning-amount', type=float, default=0.2,
                       help='Fraction of weights to prune')
    parser.add_argument('--pruning-method', choices=['magnitude', 'structured'],
                       default='magnitude', help='Pruning method')
    
    # Quantization options
    parser.add_argument('--quantization-method', choices=['dynamic', 'static'],
                       default='dynamic', help='Quantization method')
    
    # Distillation options
    parser.add_argument('--student-reduction', type=float, default=0.5,
                       help='Student model size reduction factor')
    parser.add_argument('--temperature', type=float, default=4.0,
                       help='Distillation temperature')
    parser.add_argument('--alpha', type=float, default=0.7,
                       help='Distillation loss weight')
    
    args = parser.parse_args()
    
    if not HAS_TORCH:
        print("Error: PyTorch required for model compression")
        return
    
    try:
        if args.compression == 'pruning':
            compress_transformers_model(
                args.model_name, 'pruning', args.output_dir,
                amount=args.pruning_amount,
                method=args.pruning_method
            )
        
        elif args.compression == 'quantization':
            compress_transformers_model(
                args.model_name, 'quantization', args.output_dir,
                method=args.quantization_method
            )
        
        elif args.compression == 'distillation':
            print("Distillation requires custom training loop - creating student model only")
            student_model = create_smaller_model(args.model_name, args.student_reduction)
            
            os.makedirs(args.output_dir, exist_ok=True)
            student_model.save_pretrained(args.output_dir)
            
            # Save distillation config
            distill_config = {
                'teacher_model': args.model_name,
                'student_reduction': args.student_reduction,
                'temperature': args.temperature,
                'alpha': args.alpha
            }
            
            with open(os.path.join(args.output_dir, 'distillation_config.json'), 'w') as f:
                json.dump(distill_config, f, indent=2)
            
            print(f"Student model saved to {args.output_dir}")
            print("Note: Implement training loop using DistillationTrainer class for full distillation")
    
    except Exception as e:
        print(f"Error during compression: {e}")


if __name__ == "__main__":
    main()