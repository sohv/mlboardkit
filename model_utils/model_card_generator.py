#!/usr/bin/env python3
"""
model_card_generator.py

Auto-generate model cards from experiment logs and configuration files.
"""

import argparse
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class ModelCardGenerator:
    def __init__(self):
        self.template = """# Model Card: {model_name}

## Model Details
- **Model Name**: {model_name}
- **Model Version**: {model_version}
- **Model Type**: {model_type}
- **Framework**: {framework}
- **Created**: {created_date}
- **Author**: {author}

## Model Description
{description}

## Training Data
{training_data_info}

## Training Procedure
{training_procedure}

## Performance Metrics
{performance_metrics}

## Limitations and Biases
{limitations}

## Intended Use
{intended_use}

## Ethical Considerations
{ethical_considerations}

## Technical Specifications
{technical_specs}

## Additional Information
{additional_info}
"""
    
    def generate_card(self, config: Dict[str, Any], metrics: Optional[Dict[str, Any]] = None) -> str:
        """Generate model card from configuration and metrics."""
        
        # Extract information with defaults
        model_info = {
            'model_name': config.get('model_name', 'Unknown Model'),
            'model_version': config.get('model_version', '1.0'),
            'model_type': config.get('model_type', 'Machine Learning Model'),
            'framework': config.get('framework', 'Unknown'),
            'created_date': datetime.now().strftime('%Y-%m-%d'),
            'author': config.get('author', 'Not specified'),
            'description': config.get('description', 'No description provided.'),
        }
        
        # Training data information
        data_info = config.get('training_data', {})
        training_data_info = f"""
- **Dataset**: {data_info.get('dataset_name', 'Not specified')}
- **Size**: {data_info.get('size', 'Not specified')}
- **Source**: {data_info.get('source', 'Not specified')}
- **Preprocessing**: {data_info.get('preprocessing', 'Not specified')}
"""
        
        # Training procedure
        training_config = config.get('training', {})
        training_procedure = f"""
- **Learning Rate**: {training_config.get('learning_rate', 'Not specified')}
- **Batch Size**: {training_config.get('batch_size', 'Not specified')}
- **Epochs**: {training_config.get('epochs', 'Not specified')}
- **Optimizer**: {training_config.get('optimizer', 'Not specified')}
- **Loss Function**: {training_config.get('loss_function', 'Not specified')}
"""
        
        # Performance metrics
        if metrics:
            performance_metrics = "\\n".join([f"- **{k}**: {v}" for k, v in metrics.items()])
        else:
            performance_metrics = "No metrics provided."
        
        # Technical specifications
        tech_specs = f"""
- **Model Size**: {config.get('model_size', 'Not specified')}
- **Parameters**: {config.get('parameters', 'Not specified')}
- **Input Format**: {config.get('input_format', 'Not specified')}
- **Output Format**: {config.get('output_format', 'Not specified')}
"""
        
        model_info.update({
            'training_data_info': training_data_info.strip(),
            'training_procedure': training_procedure.strip(),
            'performance_metrics': performance_metrics,
            'limitations': config.get('limitations', 'Not specified'),
            'intended_use': config.get('intended_use', 'Not specified'),
            'ethical_considerations': config.get('ethical_considerations', 'Not specified'),
            'technical_specs': tech_specs.strip(),
            'additional_info': config.get('additional_info', 'None')
        })
        
        return self.template.format(**model_info)


def main():
    parser = argparse.ArgumentParser(description='Generate model cards')
    parser.add_argument('--config', required=True, help='Configuration file (JSON/YAML)')
    parser.add_argument('--metrics', help='Metrics file (JSON)')
    parser.add_argument('--output', default='MODEL_CARD.md', help='Output markdown file')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        else:
            config = json.load(f)
    
    # Load metrics if provided
    metrics = None
    if args.metrics:
        with open(args.metrics, 'r') as f:
            metrics = json.load(f)
    
    # Generate model card
    generator = ModelCardGenerator()
    card_content = generator.generate_card(config, metrics)
    
    # Save to file
    with open(args.output, 'w') as f:
        f.write(card_content)
    
    print(f"Model card generated: {args.output}")


if __name__ == '__main__':
    main()