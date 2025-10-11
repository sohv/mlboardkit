#!/usr/bin/env python3
"""
response_consistency.py

Measure consistency under paraphrased prompts.
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any

# Simple paraphrasing templates
PARAPHRASE_TEMPLATES = [
    "Can you {verb} {object}?",
    "How do I {verb} {object}?",
    "What's the way to {verb} {object}?",
    "Please {verb} {object}",
    "I need to {verb} {object}",
    "Could you help me {verb} {object}?",
    "Show me how to {verb} {object}",
    "Explain how to {verb} {object}",
]

def extract_components(prompt: str) -> Dict[str, str]:
    """Extract verb and object from prompt"""
    # Simple extraction - could be improved with NLP
    words = prompt.lower().split()
    
    # Look for action verbs
    action_verbs = ['create', 'make', 'build', 'write', 'generate', 'develop', 'design']
    verb = next((word for word in words if word in action_verbs), 'create')
    
    # Extract object (rest of prompt)
    if verb in words:
        verb_idx = words.index(verb)
        obj_words = words[verb_idx+1:]
        obj = ' '.join(obj_words)
    else:
        obj = ' '.join(words[2:]) if len(words) > 2 else 'something'
    
    return {'verb': verb, 'object': obj}

def generate_paraphrases(prompt: str, num_paraphrases: int = 5) -> List[str]:
    """Generate paraphrased versions of a prompt"""
    components = extract_components(prompt)
    
    paraphrases = [prompt]  # Include original
    
    for _ in range(num_paraphrases - 1):
        template = random.choice(PARAPHRASE_TEMPLATES)
        paraphrase = template.format(**components)
        
        if paraphrase not in paraphrases:
            paraphrases.append(paraphrase)
    
    return paraphrases

def calculate_consistency(responses: List[str]) -> Dict[str, float]:
    """Calculate consistency metrics between responses"""
    if len(responses) < 2:
        return {'consistency_score': 1.0, 'similarity_pairs': 0}
    
    # Simple word-based similarity
    def jaccard_similarity(text1: str, text2: str) -> float:
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    similarities = []
    for i in range(len(responses)):
        for j in range(i+1, len(responses)):
            sim = jaccard_similarity(responses[i], responses[j])
            similarities.append(sim)
    
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
    
    return {
        'consistency_score': avg_similarity,
        'similarity_pairs': len(similarities),
        'min_similarity': min(similarities) if similarities else 0.0,
        'max_similarity': max(similarities) if similarities else 0.0
    }

def test_consistency(prompts: List[str], response_function=None) -> List[Dict[str, Any]]:
    """Test consistency for a list of prompts"""
    results = []
    
    for prompt in prompts:
        print(f"Testing prompt: {prompt[:50]}...")
        
        # Generate paraphrases
        paraphrases = generate_paraphrases(prompt)
        
        # Get responses (simulated if no function provided)
        responses = []
        for paraphrase in paraphrases:
            if response_function:
                try:
                    response = response_function(paraphrase)
                except Exception as e:
                    response = f"Error: {str(e)}"
            else:
                # Simulate varying responses
                base_response = f"Here's information about {paraphrase.split()[-1]}"
                if random.random() < 0.3:  # 30% chance of variation
                    response = base_response + " with additional details."
                else:
                    response = base_response
            
            responses.append(response)
        
        # Calculate consistency
        consistency_metrics = calculate_consistency(responses)
        
        result = {
            'original_prompt': prompt,
            'paraphrases': paraphrases,
            'responses': responses,
            **consistency_metrics
        }
        
        results.append(result)
    
    return results

def load_prompts(file_path: str) -> List[str]:
    """Load prompts from file"""
    path = Path(file_path)
    
    if path.suffix.lower() == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return [str(item) for item in data]
            else:
                return [str(data)]
    
    elif path.suffix.lower() == '.jsonl':
        prompts = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    if isinstance(item, dict):
                        prompts.append(item.get('prompt', str(item)))
                    else:
                        prompts.append(str(item))
                except json.JSONDecodeError:
                    continue
        return prompts
    
    else:
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

def main():
    parser = argparse.ArgumentParser(description="Measure response consistency under paraphrased prompts")
    parser.add_argument('--prompts', help='File with prompts to test')
    parser.add_argument('--prompt-text', help='Single prompt to test')
    parser.add_argument('--num-paraphrases', type=int, default=5, help='Number of paraphrases per prompt')
    parser.add_argument('--output', help='Output file for results')
    parser.add_argument('--simulate', action='store_true', help='Simulate responses for testing')
    parser.add_argument('--summary', action='store_true', help='Show summary statistics')
    
    args = parser.parse_args()
    
    # Get prompts
    if args.prompts:
        prompts = load_prompts(args.prompts)
        print(f"Loaded {len(prompts)} prompts")
    elif args.prompt_text:
        prompts = [args.prompt_text]
    else:
        # Default test prompts
        prompts = [
            "Write a Python function to sort a list",
            "Explain machine learning in simple terms",
            "Create a recipe for chocolate cake"
        ]
        print("Using default test prompts")
    
    # Test consistency
    print("Testing response consistency...")
    if args.simulate:
        print("(Simulating responses)")
    
    results = test_consistency(prompts, response_function=None if args.simulate else None)
    
    # Save results
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output}")
    
    # Show summary
    if args.summary or not args.output:
        consistency_scores = [r['consistency_score'] for r in results]
        
        print(f"\nConsistency Summary:")
        print(f"  Total prompts tested: {len(results)}")
        print(f"  Average consistency score: {sum(consistency_scores)/len(consistency_scores):.3f}")
        print(f"  High consistency (>0.8): {sum(1 for s in consistency_scores if s > 0.8)}")
        print(f"  Low consistency (<0.5): {sum(1 for s in consistency_scores if s < 0.5)}")
        
        # Show worst consistency case
        worst_idx = consistency_scores.index(min(consistency_scores))
        worst_result = results[worst_idx]
        print(f"\nLowest consistency ({worst_result['consistency_score']:.3f}):")
        print(f"  Prompt: {worst_result['original_prompt']}")

if __name__ == "__main__":
    main()