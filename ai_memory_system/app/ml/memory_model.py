"""
Adaptive memory model: MLP that learns importance scoring.
Replaces rule-based scoring with learned predictions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MODEL_PATH = MODEL_DIR / "importance_model.pt"
CONFIG_PATH = MODEL_DIR / "model_config.json"
logger = logging.getLogger(__name__)


class ImportanceNet(nn.Module):
    """Simple MLP for predicting chunk importance (0-1)."""

    def __init__(self, input_dim: int = 3, hidden_dim1: int = 8, hidden_dim2: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def ensure_model_dir() -> None:
    """Create model directory if it doesn't exist."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_model(
    features: list[list[float]],
    labels: list[int],
    epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 0.01,
    verbose: bool = True,
) -> Optional[ImportanceNet]:
    """
    Train the importance scoring MLP.

    Args:
        features: List of [frequency, recency, similarity]
        labels: List of importance labels (0 or 1)
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        verbose: Print training progress

    Returns:
        Trained model or None if insufficient data
    """
    if len(features) < 4:
        if verbose:
            logger.warning("Insufficient training data: %s samples (need at least 4)", len(features))
        return None

    ensure_model_dir()

    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ImportanceNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
            avg_loss = total_loss / len(dataloader)
            logger.info("Epoch %s/%s, Loss: %.4f", epoch + 1, epochs, avg_loss)

    model.eval()
    save_model(model)

    if verbose:
        logger.info("Model trained and saved to %s", MODEL_PATH)

    return model


def save_model(model: ImportanceNet) -> None:
    """Save trained model to disk."""
    ensure_model_dir()
    torch.save(model.state_dict(), MODEL_PATH)


def load_model() -> Optional[ImportanceNet]:
    """Load trained model from disk if it exists."""
    if not MODEL_PATH.exists():
        return None

    model = ImportanceNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def predict_importance(
    frequency: float,
    recency: float,
    similarity: float,
    model: Optional[ImportanceNet] = None,
) -> float:
    """
    Predict importance score for a chunk using the trained model.

    Args:
        frequency: Normalized frequency (0-1)
        recency: Normalized recency (0-1)
        similarity: Similarity score (0-1)
        model: Optional pre-loaded model (loads if None)

    Returns:
        Predicted importance score (0-1), or None if model not available
    """
    if model is None:
        model = load_model()

    if model is None:
        return None

    with torch.no_grad():
        features = torch.tensor([[frequency, recency, similarity]], dtype=torch.float32)
        output = model(features).item()
    return float(output)


def model_exists() -> bool:
    """Check if a trained model exists."""
    return MODEL_PATH.exists()


def delete_model() -> None:
    """Delete the trained model."""
    if MODEL_PATH.exists():
        MODEL_PATH.unlink()


def train_from_buffer(
    buffer: list[dict],
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 0.01,
    verbose: bool = False,
) -> bool:
    """
    Train model directly from memory buffer items.
    Each buffered item is treated as important (label=1).
    This is System C v2: items are pre-filtered by importance threshold.
    
    Args:
        buffer: List of dicts with keys: frequency, recency, similarity, importance, text
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        verbose: Print training progress
    
    Returns:
        True if training succeeded, False if buffer empty or too small
    """
    if not buffer:
        return False

    features = []
    labels = []

    for item in buffer:
        try:
            features.append([
                float(item["frequency"]),
                float(item["recency"]),
                float(item["similarity"]),
            ])
            label = item.get("label", item.get("importance", 1))
            labels.append(1 if int(label) > 0 else 0)
        except (KeyError, TypeError, ValueError):
            continue

    if len(features) < 4:
        if verbose:
            logger.warning("Not enough valid buffer samples to train: %s (need >= 4)", len(features))
        return False

    if verbose:
        logger.info("Training from buffer: %s samples", len(features))

    model = train_model(
        features=features,
        labels=labels,
        epochs=epochs,
        batch_size=min(batch_size, len(features)),
        learning_rate=learning_rate,
        verbose=verbose,
    )
    return model is not None
