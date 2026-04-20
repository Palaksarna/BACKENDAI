"""
Data logger for collecting retrieval training data.
Logs each retrieval event with features and a label for model training.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "training_logs"
TRAINING_DATA_FILE = LOGS_DIR / "retrieval_data.jsonl"
EVALUATION_DATA_FILE = LOGS_DIR / "evaluation_results.jsonl"


@dataclass
class RetrievalLog:
    """A single retrieval event with features and importance label."""
    frequency: float
    recency: float
    similarity: float
    importance: int = 0  # 1 if chunk was useful, 0 otherwise


@dataclass
class EvaluationLog:
    """A single evaluation result for a system and task type."""
    system: str
    task: str
    correct: int
    total: int


def ensure_logs_dir() -> None:
    """Create training logs directory if it doesn't exist."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def log_retrieval(frequency: float, recency: float, similarity: float, importance: int = 0) -> None:
    """
    Log a retrieval event for later training.

    Args:
        frequency: Normalized frequency (0-1)
        recency: Normalized recency (0-1)
        similarity: Similarity score (0-1)
        importance: Label: 1 if chunk was useful, 0 otherwise
    """
    ensure_logs_dir()

    record = RetrievalLog(
        frequency=float(frequency),
        recency=float(recency),
        similarity=float(similarity),
        importance=int(importance),
    )

    with open(TRAINING_DATA_FILE, "a") as f:
        f.write(json.dumps(asdict(record)) + "\n")


def load_training_data() -> tuple[List[List[float]], List[int]]:
    """
    Load training data from logs.

    Returns:
        (features, labels) where features is [[freq, rec, sim], ...] and labels is [0 or 1, ...]
    """
    ensure_logs_dir()

    if not TRAINING_DATA_FILE.exists():
        return [], []

    features = []
    labels = []

    with open(TRAINING_DATA_FILE, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                features.append([
                    record["frequency"],
                    record["recency"],
                    record["similarity"],
                ])
                labels.append(record["importance"])
            except (json.JSONDecodeError, KeyError):
                pass

    return features, labels


def clear_logs() -> None:
    """Clear all training logs."""
    ensure_logs_dir()
    if TRAINING_DATA_FILE.exists():
        TRAINING_DATA_FILE.unlink()


def get_log_count() -> int:
    """Get number of logged retrieval events."""
    ensure_logs_dir()
    if not TRAINING_DATA_FILE.exists():
        return 0

    count = 0
    with open(TRAINING_DATA_FILE, "r") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def log_evaluation_result(system: str, task: str, correct: int, total: int) -> None:
    """Log accuracy-style evaluation results for a given system and task."""
    ensure_logs_dir()
    record = EvaluationLog(
        system=system,
        task=task,
        correct=int(correct),
        total=int(total),
    )
    with open(EVALUATION_DATA_FILE, "a") as f:
        f.write(json.dumps(asdict(record)) + "\n")


def load_evaluation_summary() -> dict:
    """Load a compact evaluation summary grouped by system and task."""
    ensure_logs_dir()
    if not EVALUATION_DATA_FILE.exists():
        return {}

    summary: dict = {}
    with open(EVALUATION_DATA_FILE, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                system = str(record.get("system", "unknown"))
                task = str(record.get("task", "unknown"))
                correct = int(record.get("correct", 0))
                total = int(record.get("total", 0))
                summary.setdefault(system, {}).setdefault(task, {"correct": 0, "total": 0})
                summary[system][task]["correct"] += correct
                summary[system][task]["total"] += total
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

    for system, tasks in summary.items():
        for task, stats in tasks.items():
            total = stats["total"]
            stats["accuracy"] = (stats["correct"] / total) if total else 0.0

    return summary
