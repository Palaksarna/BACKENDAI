"""
Training pipeline for adaptive memory model.
Loads retrieval logs, trains MLP, validates improvements.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

from .data_logger import load_training_data
from .memory_model import train_model, predict_importance, load_model, model_exists

logger = logging.getLogger(__name__)


def train_importance_model(
    min_samples: int = 10,
    epochs: int = 15,
    batch_size: int = 4,
    learning_rate: float = 0.01,
    verbose: bool = True,
) -> bool:
    """
    Train the importance scoring model from retrieval logs.

    Args:
        min_samples: Minimum samples required to train
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        verbose: Print training progress

    Returns:
        True if training succeeded, False otherwise
    """
    try:
        # Load training data from logs
        features, labels = load_training_data()

        if len(features) < min_samples:
            if verbose:
                logger.warning(
                    "Insufficient log data: %s samples (need at least %s). Continue using System B (rule-based scoring).",
                    len(features),
                    min_samples,
                )
            return False

        if verbose:
            positive_labels = sum(labels)
            logger.info("Training data loaded")
            logger.info("Samples: %s", len(features))
            logger.info("Positive (important): %s (%.1f%%)", positive_labels, 100 * positive_labels / len(labels))
            logger.info("Negative (less important): %s", len(labels) - positive_labels)

        # Train the model
        model = train_model(
            features=features,
            labels=labels,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            verbose=verbose,
        )

        if model is None:
            if verbose:
                logger.error("Model training failed. Using System B.")
            return False

        if verbose:
            logger.info("Model trained successfully. System C is now active.")

        return True

    except Exception as e:
        if verbose:
            logger.exception("Training error: %s", e)
        return False


def get_system_info() -> dict:
    """Get current system state info."""
    return {
        "model_trained": model_exists(),
        "system": "System C (ML-based)" if model_exists() else "System B (Rule-based)",
    }


def validate_model_predictions(
    test_samples: Optional[list[list[float]]] = None,
    verbose: bool = True,
) -> Optional[dict]:
    """
    Validate model predictions on test data.

    Args:
        test_samples: Optional list of [frequency, recency, similarity] to predict on
        verbose: Print validation results

    Returns:
        Validation stats dict or None if model not available
    """
    model = load_model()
    if model is None:
        if verbose:
            logger.warning("No trained model available for validation.")
        return None

    if test_samples is None:
        # Use default test cases
        test_samples = [
            [0.0, 0.0, 0.0],  # Never used, not recent, not similar
            [1.0, 1.0, 1.0],  # High frequency, recent, similar
            [0.5, 0.5, 0.5],  # Medium across all
            [0.8, 0.2, 0.7],  # High freq/sim, low recency
        ]

    predictions = []
    for freq, recency, sim in test_samples:
        pred = predict_importance(frequency=freq, recency=recency, similarity=sim, model=model)
        predictions.append(pred)

    if verbose:
        logger.info("Model validation predictions")
        for (freq, recency, sim), pred in zip(test_samples, predictions):
            logger.info("[F:%.2f, R:%.2f, S:%.2f] -> %.3f", freq, recency, sim, pred)

    return {
        "test_samples": test_samples,
        "predictions": predictions,
        "mean_prediction": sum(predictions) / len(predictions),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    logger.info("AI Memory System - Model Training Tool")

    info = get_system_info()
    logger.info("Current system: %s", info["system"])

    if len(sys.argv) > 1 and sys.argv[1] == "train":
        logger.info("Starting training pipeline")
        success = train_importance_model(verbose=True)
        if success:
            validate_model_predictions(verbose=True)
        sys.exit(0 if success else 1)
    else:
        logger.info("Usage: python train.py train")
        logger.info("This script:")
        logger.info("  1. Loads retrieval logs collected by the system")
        logger.info("  2. Trains MLP model on importance patterns")
        logger.info("  3. Validates predictions on test cases")
        logger.info("  4. Activates System C (ML-based scoring) on success")
        logger.info("Logging happens automatically during /chat API calls.")
        logger.info("Run 'python train.py train' once you have 10+ interaction logs.")
