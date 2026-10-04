"""
Visual Product Recommendation Engine.

This module coordinates:
1. Feature extraction from query images.
2. FAISS vector search across catalog embeddings.
3. Similarity ranking & faceted filtering (category, price range, rating).
4. Formatting recommendation payloads for frontend presentation.
"""

import os
import sys
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.feature_extractor import ProductFeatureExtractor
from src.indexer import ProductVectorIndexer


class VisualProductRecommender:
    """End-to-end visual recommendation engine linking deep features, vector search, and faceted filtering."""

    def __init__(
        self,
        metadata_path: str = "data/metadata.csv",
        index_path: str = "models/faiss_index.bin",
        embeddings_path: str = "models/product_embeddings.npy",
        device: Optional[str] = None
    ):
        """Initializes the recommendation engine with catalog metadata and index.

        Args:
            metadata_path: Path to CSV metadata file.
            index_path: Path to FAISS binary index file.
            embeddings_path: Path to saved product embeddings npy file.
            device: 'cuda' or 'cpu'.
        """
        self.metadata_path = metadata_path
        self.index_path = index_path
        self.embeddings_path = embeddings_path

        # 1. Load catalog metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        self.metadata_df = pd.read_csv(metadata_path)
        print(f"[VisualProductRecommender] Loaded metadata for {len(self.metadata_df)} items.")

        # 2. Initialize Feature Extractor
        self.feature_extractor = ProductFeatureExtractor(device=device)

        # 3. Initialize & Load FAISS Index
        self.indexer = ProductVectorIndexer(embedding_dim=2048)
        if os.path.exists(index_path):
            self.indexer.load_index(index_path)
        else:
            print(f"[VisualProductRecommender] Warning: Index {index_path} not found. Must be built first.")

        # 4. Cached embeddings (optional, for direct cosine operations)
        self.embeddings = None
        if os.path.exists(embeddings_path):
            self.embeddings = np.load(embeddings_path)
            print(f"[VisualProductRecommender] Loaded embeddings matrix: {self.embeddings.shape}")

    def recommend(
        self,
        query_input: Union[str, Image.Image, int],
        top_k: int = 6,
        category_filter: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        min_rating: Optional[float] = None,
        exclude_self: bool = True
    ) -> Tuple[pd.DataFrame, float, np.ndarray]:
        """Generates visually similar product recommendations for a query image or catalog ID.

        Args:
            query_input: Image path, PIL Image, or catalog index integer.
            top_k: Number of recommendations to return.
            category_filter: Optional category string to restrict recommendations.
            min_price: Optional lower bound on product price.
            max_price: Optional upper bound on product price.
            min_rating: Optional minimum customer rating.
            exclude_self: Whether to filter out the query item itself if query is from catalog.

        Returns:
            Tuple of:
                - recommendations_df: DataFrame with top recommended items and similarity metrics.
                - latency_ms: FAISS search time in milliseconds.
                - query_embedding: The extracted query vector.
        """
        query_catalog_idx = None

        # 1. Extract query embedding
        if isinstance(query_input, (int, np.integer)):
            query_catalog_idx = int(query_input)
            if self.embeddings is not None and query_catalog_idx < len(self.embeddings):
                query_embedding = self.embeddings[query_catalog_idx]
            else:
                img_path = self.metadata_df.iloc[query_catalog_idx]['local_image_path']
                query_embedding = self.feature_extractor.extract_embedding(img_path)
        else:
            query_embedding = self.feature_extractor.extract_embedding(query_input)

        # 2. Over-fetch candidates if filters are applied to guarantee top_k after filtering
        has_filters = any([category_filter, min_price is not None, max_price is not None, min_rating is not None])
        fetch_k = min(self.indexer.total_vectors, max(top_k * 5 if has_filters else top_k + 2, 20))

        # 3. Vector Similarity Search via FAISS
        raw_scores, raw_indices, latency_ms = self.indexer.search(query_embedding, top_k=fetch_k)

        # 4. Build candidate DataFrame
        candidates = self.metadata_df.iloc[raw_indices].copy()
        candidates['similarity_score'] = raw_scores
        candidates['catalog_index'] = raw_indices
        # Convert cosine similarity (-1 to 1) into percentage match (0% to 100%)
        candidates['similarity_pct'] = (np.clip(raw_scores, 0.0, 1.0) * 100).round(1)

        # 5. Filter out query item itself if querying from catalog
        if exclude_self and query_catalog_idx is not None:
            candidates = candidates[candidates['catalog_index'] != query_catalog_idx]

        # 6. Apply faceted filters
        if category_filter and category_filter != "All Categories":
            candidates = candidates[candidates['category'] == category_filter]

        if min_price is not None:
            candidates = candidates[candidates['price_usd'] >= min_price]

        if max_price is not None:
            candidates = candidates[candidates['price_usd'] <= max_price]

        if min_rating is not None:
            candidates = candidates[candidates['rating'] >= min_rating]

        # 7. Select Top-K and reset rank
        recommendations = candidates.head(top_k).reset_index(drop=True)
        recommendations['rank'] = range(1, len(recommendations) + 1)

        return recommendations, latency_ms, query_embedding
