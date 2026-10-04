"""
Feature Extraction Module for Visual Product Recommendation System.

This module utilizes a pretrained ResNet50 Convolutional Neural Network (ImageNet weights)
with the final classification head replaced by an identity mapping, extracting 2048-dimensional
dense visual representations (embeddings) from product images. Embeddings are L2-normalized
so that dot products compute exact Cosine Similarity.
"""

import os
from typing import List, Union
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm


class ProductFeatureExtractor:
    """Extracts dense feature representations from product images using a pretrained ResNet-50 backbone."""

    def __init__(self, device: str = None):
        """Initialize the feature extractor with pretrained ResNet50 weights."""
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[ProductFeatureExtractor] Initializing ResNet50 backbone on device: {self.device}")

        # Load pretrained ResNet50 with standard ImageNet weights
        weights = models.ResNet50_Weights.DEFAULT
        self.model = models.resnet50(weights=weights)

        # Replace classification layer with Identity to output 2048-dim GAP features
        self.model.fc = nn.Identity()

        # Set model to evaluation mode and freeze parameters
        self.model.eval()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False

        # Define ImageNet preprocessing transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def preprocess_image(self, image_input: Union[str, Image.Image]) -> torch.Tensor:
        """Preprocesses an image path or PIL Image into a normalized tensor.

        Args:
            image_input: File path to image or PIL.Image instance.

        Returns:
            Preprocessed image tensor of shape (1, 3, 224, 224).
        """
        try:
            if isinstance(image_input, str):
                image = Image.open(image_input).convert("RGB")
            elif isinstance(image_input, Image.Image):
                image = image_input.convert("RGB")
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")

            tensor = self.transform(image)
            return tensor.unsqueeze(0)  # Add batch dimension: (1, 3, H, W)
        except Exception as error:
            print(f"[Error] Failed to preprocess image {image_input}: {error}")
            raise error

    def extract_embedding(self, image_input: Union[str, Image.Image]) -> np.ndarray:
        """Extracts and L2-normalizes a 2048-dimensional feature embedding for a single image.

        Args:
            image_input: File path or PIL.Image instance.

        Returns:
            1D NumPy array of shape (2048,), L2-normalized.
        """
        tensor = self.preprocess_image(image_input).to(self.device)
        with torch.no_grad():
            embedding = self.model(tensor)
            embedding_np = embedding.cpu().numpy().flatten()

        # Compute L2 norm for Cosine Similarity equivalence
        norm = np.linalg.norm(embedding_np)
        if norm > 0:
            embedding_np = embedding_np / norm

        return embedding_np.astype(np.float32)

    def extract_batch_embeddings(
        self,
        image_paths: List[str],
        batch_size: int = 32
    ) -> np.ndarray:
        """Extracts dense feature representations for a list of image paths in batches.

        Args:
            image_paths: List of file paths to images.
            batch_size: Batch size for batch inference.

        Returns:
            2D NumPy array of shape (N, 2048), L2-normalized.
        """
        total_images = len(image_paths)
        all_embeddings = []

        print(f"[ProductFeatureExtractor] Extracting embeddings for {total_images} catalog items (batch size: {batch_size})...")

        for start_idx in tqdm(range(0, total_images, batch_size), desc="Extracting embeddings"):
            end_idx = min(start_idx + batch_size, total_images)
            batch_paths = image_paths[start_idx:end_idx]

            batch_tensors = []
            valid_indices = []

            for i, img_path in enumerate(batch_paths):
                try:
                    tensor = self.preprocess_image(img_path)
                    batch_tensors.append(tensor)
                    valid_indices.append(i)
                except Exception as err:
                    print(f"[Warning] Skipping unreadable image {img_path}: {err}")

            if not batch_tensors:
                continue

            batch_tensor = torch.cat(batch_tensors, dim=0).to(self.device)

            with torch.no_grad():
                batch_embeddings = self.model(batch_tensor).cpu().numpy()

            # L2 normalize each embedding row
            norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            batch_embeddings = batch_embeddings / norms

            all_embeddings.append(batch_embeddings.astype(np.float32))

        if all_embeddings:
            final_embeddings = np.vstack(all_embeddings)
        else:
            final_embeddings = np.empty((0, 2048), dtype=np.float32)

        print(f"[ProductFeatureExtractor] Extraction complete. Embedding matrix shape: {final_embeddings.shape}")
        return final_embeddings
