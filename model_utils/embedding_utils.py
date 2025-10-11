#!/usr/bin/env python3
"""
embedding_utils.py

Utilities to compute embeddings using sentence-transformers when available,
falling back to scikit-learn TF-IDF if not.
"""

from typing import List, Optional
import os


def compute_embeddings(texts: List[str], model_name: Optional[str] = None):
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name or 'all-MiniLM-L6-v2')
        embeddings = model.encode(texts, show_progress_bar=False)
        return embeddings
    except Exception:
        # Fallback to TF-IDF vectorizer
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(max_features=1024)
        X = vec.fit_transform(texts)
        return X.toarray()


def save_embeddings(embeddings, output_path: str):
    import numpy as np
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    np.save(output_path, embeddings)
    print(f"Saved embeddings to {output_path}")


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='JSON list of strings or a text file (one per line)')
    parser.add_argument('--output', default='embeddings.npy')
    parser.add_argument('--model', help='SentenceTransformer model name')
    args = parser.parse_args()

    texts = []
    if args.input.endswith('.json'):
        with open(args.input, 'r') as f:
            texts = json.load(f)
    else:
        with open(args.input, 'r') as f:
            texts = [l.strip() for l in f if l.strip()]

    emb = compute_embeddings(texts, args.model)
    save_embeddings(emb, args.output)
