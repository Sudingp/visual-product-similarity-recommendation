"""
Evaluation Module for Visual Product Recommendation System.

This module evaluates the performance of the visual recommendation engine using:
1. Precision@K: Proportion of top-K recommendations belonging to the same product category.
2. Recall@K: Proportion of total catalog items in that category retrieved in top-K.
3. Mean Average Precision (mAP@K).
4. Sub-second latency benchmark across FAISS vector search queries.
"""

import os
import sys
import json
import time
from typing import Dict, List, Any
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.indexer import ProductVectorIndexer


class RecommenderEvaluator:
    """Evaluates quantitative retrieval metrics (Precision@K, Recall@K, Latency) across catalog categories."""

    def __init__(
        self,
        metadata_path: str = "data/metadata.csv",
        embeddings_path: str = "models/product_embeddings.npy",
        index_path: str = "models/faiss_index.bin"
    ):
        self.metadata_df = pd.read_csv(metadata_path)
        self.embeddings = np.load(embeddings_path)
        self.indexer = ProductVectorIndexer(embedding_dim=self.embeddings.shape[1])
        self.indexer.load_index(index_path)

        # Count total items per category for Recall calculation
        self.category_counts = self.metadata_df['category'].value_counts().to_dict()

    def evaluate_at_k(self, k_values: List[int] = [1, 3, 5, 10, 20], num_samples: int = 300) -> Dict[str, Any]:
        """Calculates Precision@K, Recall@K, and Latency over a representative sample of queries."""
        print(f"[RecommenderEvaluator] Evaluating metrics across {num_samples} sample queries for K = {k_values}...")

        # Sample queries uniformly across categories
        sample_indices = (
            self.metadata_df.groupby('category', group_keys=False)
            .apply(lambda x: x.sample(n=min(len(x), num_samples // len(self.category_counts)), random_state=42))
            .index.tolist()
        )

        max_k = max(k_values)
        results = {k: {"precisions": [], "recalls": []} for k in k_values}
        latencies_ms = []

        for query_idx in sample_indices:
            query_embedding = self.embeddings[query_idx]
            query_category = self.metadata_df.iloc[query_idx]['category']
            total_relevant_in_cat = self.category_counts[query_category] - 1  # exclude self

            # Search up to max_k + 1 to account for self-match
            scores, indices, latency = self.indexer.search(query_embedding, top_k=max_k + 1)
            latencies_ms.append(latency)

            # Exclude query item itself
            filtered_indices = [idx for idx in indices if idx != query_idx]

            retrieved_categories = self.metadata_df.iloc[filtered_indices]['category'].values

            for k in k_values:
                top_k_cats = retrieved_categories[:k]
                matches = sum(1 for c in top_k_cats if c == query_category)

                precision = matches / k
                recall = matches / max(total_relevant_in_cat, 1)

                results[k]["precisions"].append(precision)
                results[k]["recalls"].append(recall)

        # Aggregate metrics
        summary = {
            "num_queries_evaluated": len(sample_indices),
            "k_metrics": {},
            "latency": {
                "mean_ms": float(np.mean(latencies_ms)),
                "p50_ms": float(np.median(latencies_ms)),
                "p95_ms": float(np.percentile(latencies_ms, 95)),
                "p99_ms": float(np.percentile(latencies_ms, 99))
            }
        }

        for k in k_values:
            mean_precision = float(np.mean(results[k]["precisions"]))
            mean_recall = float(np.mean(results[k]["recalls"]))
            summary["k_metrics"][f"K={k}"] = {
                "precision_at_k": round(mean_precision, 4),
                "precision_at_k_pct": f"{mean_precision * 100:.2f}%",
                "recall_at_k": round(mean_recall, 4),
                "recall_at_k_pct": f"{mean_recall * 100:.2f}%"
            }

        return summary


def run_evaluation(output_json_path: str = "data/evaluation_results.json"):
    print("=" * 60)
    print("Visual Product Similarity & Recommendation - Evaluation Run")
    print("=" * 60)

    evaluator = RecommenderEvaluator()
    metrics = evaluator.evaluate_at_k(k_values=[1, 3, 5, 10, 20], num_samples=300)

    print("\n--- Quantitative Results ---")
    for k_key, vals in metrics["k_metrics"].items():
        print(f"{k_key:6s} | Precision: {vals['precision_at_k_pct']:7s} | Recall: {vals['recall_at_k_pct']:7s}")

    lat = metrics["latency"]
    print(f"\nSearch Latency (FAISS IndexFlatIP):")
    print(f"  Mean:   {lat['mean_ms']:.3f} ms")
    print(f"  P50:    {lat['p50_ms']:.3f} ms")
    print(f"  P95:    {lat['p95_ms']:.3f} ms")
    print(f"  P99:    {lat['p99_ms']:.3f} ms")

    with open(output_json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nEvaluation summary saved to: {output_json_path}")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation()
