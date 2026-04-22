from __future__ import annotations

from pathlib import Path
import json
import logging
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..services.embedding import EMBEDDING_DIMENSION, get_embedding


logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"
NEURAL_FACT_MODEL_PATH = MODEL_DIR / "neural_fact_encoder.pt"
NEURAL_FACT_BANK_PATH = MODEL_DIR / "neural_fact_bank.json"


class FactEncoder(nn.Module):
    """Autoencoder used to learn dense fact representations from embeddings."""

    def __init__(self, input_dim: int = EMBEDDING_DIMENSION, hidden_dim: int = 64, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def _ensure_model_dir() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _fact_text(record: Dict[str, Any]) -> str:
    value = str(record.get("value", "")).strip()
    if not value:
        return ""
    return value


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0

    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def train_neural_fact_memory(
    promoted_facts: Sequence[Dict[str, Any]],
    epochs: int = 60,
    batch_size: int = 8,
    learning_rate: float = 0.005,
    min_samples: int = 3,
    verbose: bool = False,
) -> bool:
    records = [record for record in promoted_facts if _fact_text(record)]
    if len(records) < min_samples:
        if verbose:
            logger.info("Neural fact training skipped: %s samples (need at least %s)", len(records), min_samples)
        return False

    _ensure_model_dir()

    fact_texts = [_fact_text(record) for record in records]
    embeddings = [get_embedding(text) for text in fact_texts]
    X = torch.tensor(embeddings, dtype=torch.float32)

    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)

    model = FactEncoder(input_dim=EMBEDDING_DIMENSION)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for (batch_x,) in loader:
            optimizer.zero_grad()
            reconstructed = model(batch_x)
            loss = loss_fn(reconstructed, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
            avg_loss = total_loss / len(loader)
            logger.info("Neural fact epoch %s/%s loss=%.5f", epoch + 1, epochs, avg_loss)

    model.eval()
    with torch.no_grad():
        latent_vectors = model.encode(X).tolist()

    torch.save(model.state_dict(), NEURAL_FACT_MODEL_PATH)

    bank = []
    for record, fact_text, latent in zip(records, fact_texts, latent_vectors):
        bank.append(
            {
                "key": str(record.get("key", "")).strip().lower(),
                "value": record.get("value"),
                "fact_text": fact_text,
                "frequency": int(record.get("frequency", 0)),
                "last_seen": record.get("last_seen"),
                "latent": latent,
            }
        )

    with NEURAL_FACT_BANK_PATH.open("w", encoding="utf-8") as handle:
        json.dump(bank, handle, indent=2, ensure_ascii=True)

    if verbose:
        logger.info("Neural fact memory trained and saved with %s facts", len(bank))

    return True


def _load_neural_fact_bank() -> List[Dict[str, Any]]:
    if not NEURAL_FACT_BANK_PATH.exists():
        return []

    try:
        with NEURAL_FACT_BANK_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def _load_encoder_model() -> FactEncoder | None:
    if not NEURAL_FACT_MODEL_PATH.exists():
        return None

    model = FactEncoder(input_dim=EMBEDDING_DIMENSION)
    model.load_state_dict(torch.load(NEURAL_FACT_MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def retrieve_neural_facts(query: str, top_k: int = 3, min_similarity: float = 0.15) -> List[Dict[str, Any]]:
    model = _load_encoder_model()
    bank = _load_neural_fact_bank()
    if model is None or not bank:
        return []

    query_embedding = torch.tensor([get_embedding(query)], dtype=torch.float32)
    with torch.no_grad():
        query_latent = model.encode(query_embedding).squeeze(0).tolist()

    scored: List[Dict[str, Any]] = []
    for item in bank:
        latent = item.get("latent")
        if not isinstance(latent, list):
            continue
        similarity = _cosine_similarity(query_latent, [float(v) for v in latent])
        if similarity < min_similarity:
            continue
        scored.append(
            {
                "key": item.get("key", ""),
                "value": item.get("value"),
                "fact_text": str(item.get("fact_text", "")).strip(),
                "similarity": float(similarity),
            }
        )

    scored.sort(key=lambda item: item["similarity"], reverse=True)
    return scored[: max(0, int(top_k))]


def neural_fact_model_ready() -> bool:
    return NEURAL_FACT_MODEL_PATH.exists() and NEURAL_FACT_BANK_PATH.exists()