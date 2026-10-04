"""
Vector Database & Indexing Module using FAISS.

This module handles building, saving, loading, and querying vector similarity
indexes with Facebook AI Similarity Search (FAISS). Using Inner Product (IndexFlatIP)
with L2-normalized embeddings, it achieves exact, millisecond-level Cosine Similarity search.
"""

import os
import time
from typing import Tuple, Optional
import numpy as np
import faiss


class ProductVectorIndexer:
    """Manages FAISS vector indexing and similarity retrieval for product embeddings."""

    def __init__(self, embedding_dim: int = 2048):
        """Initialize the vector indexer.

        Args:
            embedding_dim: Dimensionality of feature embeddings (default 2048 for ResNet50).
        """
        self.embedding_dim = embedding_dim
        # IndexFlatIP calculates inner product: <u, v>. When ||u||=||v||=1, <u, v> == Cosine Similarity
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.is_trained = True
        self.total_vectors = 0

    def build_index(self, embeddings: np.ndarray) -> None:
        """Builds and populates the FAISS index from an embedding matrix.

        Args:
            embeddings: Float32 NumPy array of shape (N, embedding_dim).
        """
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected embeddings shape (N, {self.embedding_dim}), got {embeddings.shape}"
            )

        # Ensure embeddings are contiguous float32
        embeddings_contiguous = np.ascontiguousarray(embeddings, dtype=np.float32)

        start_time = time.time()
        print(f"[ProductVectorIndexer] Adding {len(embeddings_contiguous)} vectors to FAISS index...")
        self.index.reset()
        self.index.add(embeddings_contiguous)
        self.total_vectors = self.index.ntotal
        elapsed = (time.time() - start_time) * 1000.0

        print(f"[ProductVectorIndexer] Successfully indexed {self.total_vectors} items in {elapsed:.2f} ms.")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Searches the FAISS index for the Top-K nearest neighbors.

        Args:
            query_embedding: 1D array of shape (embedding_dim,) or 2D array of shape (1, embedding_dim).
            top_k: Number of nearest items to retrieve.

        Returns:
            Tuple of:
                - similarity_scores: Cosine similarities in range [-1.0, 1.0] (top_k,)
                - indices: Item indices in the catalog (top_k,)
                - latency_ms: Search query latency in milliseconds
        """
        if self.index.ntotal == 0:
            raise RuntimeError("FAISS index is empty. Please build or load an index first.")

        # Reshape to (1, embedding_dim) if 1D
        if query_embedding.ndim == 1:
            query_vector = query_embedding.reshape(1, -1)
        else:
            query_vector = query_embedding

        query_contiguous = np.ascontiguousarray(query_vector, dtype=np.float32)

        start_time = time.time()
        scores, indices = self.index.search(query_contiguous, top_k)
        latency_ms = (time.time() - start_time) * 1000.0

        return scores[0], indices[0], latency_ms

    def save_index(self, filepath: str) -> None:
        """Saves the binary FAISS index to disk.

        Args:
            filepath: Destination file path (e.g. models/faiss_index.bin).
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        faiss.write_index(self.index, filepath)
        print(f"[ProductVectorIndexer] FAISS index saved to: {filepath}")

    def load_index(self, filepath: str) -> None:
        """Loads a precomputed binary FAISS index from disk.

        Args:
            filepath: Path to the serialized FAISS index.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"FAISS index file not found at: {filepath}")

        self.index = faiss.read_index(filepath)
        self.embedding_dim = self.index.d
        self.total_vectors = self.index.ntotal
        print(f"[ProductVectorIndexer] Loaded FAISS index from {filepath} (Total vectors: {self.total_vectors}, Dim: {self.embedding_dim})")
