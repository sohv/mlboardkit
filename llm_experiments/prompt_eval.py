#!/usr/bin/env python3
"""
prompt_eval.py

Compare model responses across different prompts for consistency and quality evaluation.
"""

import argparse
import json
from typing import List, Dict, Any
from pathlib import Path


class PromptEvaluator:
    def __init__(self):
        self.evaluation_criteria = [
            'relevance', 'clarity', 'helpfulness', 'safety', 'accuracy'
        ]
    
    def load_prompts(self, prompts_file: str) -> List[Dict[str, Any]]:
        """Load prompts from JSON file."""
        with open(prompts_file, 'r') as f:
            return json.load(f)
    
    def load_responses(self, responses_file: str) -> List[Dict[str, Any]]:
        """Load model responses from JSON file."""
        with open(responses_file, 'r') as f:
            return json.load(f)
    
    def evaluate_response_length(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Basic response length analysis."""
        lengths = [len(resp.get('response', '')) for resp in responses]
        return {
            'avg_length': sum(lengths) / len(lengths) if lengths else 0,
            'min_length': min(lengths) if lengths else 0,
            'max_length': max(lengths) if lengths else 0,
            'total_responses': len(responses)
        }
    
    def compare_prompts(self, prompts: List[Dict[str, Any]], 
                       responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare responses across different prompts."""
        prompt_stats = {}
        
        for prompt in prompts:
            prompt_id = prompt.get('id', prompt.get('prompt_id'))
            matching_responses = [r for r in responses if r.get('prompt_id') == prompt_id]
            
            if matching_responses:
                lengths = [len(r.get('response', '')) for r in matching_responses]
                prompt_stats[prompt_id] = {
                    'prompt_text': prompt.get('prompt', '')[:100] + '...',
                    'response_count': len(matching_responses),
                    'avg_response_length': sum(lengths) / len(lengths),
                    'responses': matching_responses
                }
        
        return prompt_stats
    
    def generate_report(self, prompt_stats: Dict[str, Any]) -> str:
        """Generate evaluation report."""
        report = "# Prompt Evaluation Report\\n\\n"
        
        for prompt_id, stats in prompt_stats.items():
            report += f"## Prompt {prompt_id}\\n"
            report += f"**Text**: {stats['prompt_text']}\\n\\n"
            report += f"**Response Count**: {stats['response_count']}\\n"
            report += f"**Average Response Length**: {stats['avg_response_length']:.1f} characters\\n\\n"
            
            # Sample responses
            report += "**Sample Responses**:\\n"
            for i, resp in enumerate(stats['responses'][:3]):  # Show first 3
                response_text = resp.get('response', '')[:200] + '...'
                report += f"{i+1}. {response_text}\\n\\n"
            
            report += "---\\n\\n"
        
        return report


def main():
    parser = argparse.ArgumentParser(description='Evaluate model responses across prompts')
    parser.add_argument('--prompts', required=True, help='JSON file with prompts')
    parser.add_argument('--responses', required=True, help='JSON file with responses')
    parser.add_argument('--output', default='prompt_evaluation.md', help='Output report file')
    
    args = parser.parse_args()
    
    evaluator = PromptEvaluator()
    
    # Load data
    prompts = evaluator.load_prompts(args.prompts)
    responses = evaluator.load_responses(args.responses)
    
    # Compare prompts
    prompt_stats = evaluator.compare_prompts(prompts, responses)
    
    # Generate report
    report = evaluator.generate_report(prompt_stats)
    
    # Save report
    with open(args.output, 'w') as f:
        f.write(report)
    
    print(f"Evaluation report saved to: {args.output}")
    print(f"Analyzed {len(prompts)} prompts with {len(responses)} responses")


if __name__ == '__main__':
    main()