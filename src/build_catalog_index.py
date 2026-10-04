"""
Catalog Embedding & FAISS Index Generation Script.

This offline pipeline extracts ResNet50 visual embeddings for all catalog images,
persists the embedding matrix to disk, builds a FAISS similarity index, and performs
a verification sanity query.
"""

import os
import sys
import time
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.feature_extractor import ProductFeatureExtractor
from src.indexer import ProductVectorIndexer


def build_pipeline(
    metadata_path: str = "data/metadata.csv",
    output_embeddings_path: str = "models/product_embeddings.npy",
    output_index_path: str = "models/faiss_index.bin",
    batch_size: int = 32
):
    print("=" * 65)
    print("Starting Offline Feature Extraction & FAISS Index Building Pipeline")
    print("=" * 65)

    os.makedirs(os.path.dirname(output_embeddings_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_index_path), exist_ok=True)

    # 1. Load Metadata
    print(f"[1/4] Loading catalog metadata from {metadata_path}...")
    df = pd.read_csv(metadata_path)
    image_paths = df['local_image_path'].tolist()
    total_images = len(image_paths)
    print(f"Total catalog images to process: {total_images}")

    # 2. Extract Embeddings with ResNet50
    print("\n[2/4] Initializing ResNet50 Feature Extractor...")
    extractor = ProductFeatureExtractor()

    start_extraction = time.time()
    embeddings = extractor.extract_batch_embeddings(image_paths, batch_size=batch_size)
    extraction_time = time.time() - start_extraction
    print(f"Feature extraction finished in {extraction_time:.2f} seconds ({total_images / extraction_time:.1f} images/sec).")

    # 3. Save Embeddings
    print(f"\n[3/4] Persisting embedding matrix to {output_embeddings_path}...")
    np.save(output_embeddings_path, embeddings)
    print(f"Saved {embeddings.shape[0]} embeddings of dimensionality {embeddings.shape[1]}. Size on disk: {os.path.getsize(output_embeddings_path) / (1024*1024):.2f} MB")

    # 4. Build and Save FAISS Index
    print(f"\n[4/4] Building FAISS IndexFlatIP (Cosine Similarity)...")
    indexer = ProductVectorIndexer(embedding_dim=embeddings.shape[1])
    indexer.build_index(embeddings)
    indexer.save_index(output_index_path)

    # Verification Query
    print("\n--- Running Sanity Verification Query ---")
    sample_query = embeddings[0]
    scores, indices, latency_ms = indexer.search(sample_query, top_k=5)
    print(f"Verification query completed in {latency_ms:.3f} ms!")
    print(f"Top-5 neighbor indices: {indices}")
    print(f"Top-5 similarity scores (inner products): {scores}")
    print(f"Query item category: {df.iloc[0]['category']}")
    print(f"Top match category:  {df.iloc[indices[0]]['category']}")
    print("=" * 65)
    print("Pipeline completed successfully! All assets ready for serving.")
    print("=" * 65)


if __name__ == "__main__":
    build_pipeline()
