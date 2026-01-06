# -*- coding: utf-8 -*-

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer


class ArxivEvaluator:
    def __init__(self, index_path, metadata_path):
        self.index = faiss.read_index(index_path)
        self.metadata = pd.read_parquet(metadata_path)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def search_papers(self, query, k=5):
        # 带类别过滤的检索
        query_embed = self.embedder.encode([query])
        distances, indices = self.index.search(query_embed, k)
        
        results = []
        for idx in indices[0]:
            record = self.metadata.iloc[idx]
            results.append({
                "score": float(distances[0][idx]),
                "paper_id": record['paper_id'],
                "text": record['text']
            })
        return results
    
    def evaluate(self, test_questions):
        for question, keyword in test_questions:
            print(f"Question: {question}")
            results = self.search_papers(question)
            print(f"Top Result Snippet: {results[0]['text'][:200]}...")
            print(f"Contains Keyword [{keyword}]? {keyword.lower() in results[0]['text'].lower()}")
            print("-" * 80)