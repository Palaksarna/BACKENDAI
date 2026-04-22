from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import math
import re
from typing import Any, Dict, List, Optional, Sequence


ACTIVE = "ACTIVE"
PASSIVE = "PASSIVE"
ARCHIVED = "ARCHIVED"


class MemoryForgettingAgent:
    """Evaluate memory value and assign lifecycle states."""

    def __init__(
        self,
        w1: float = 0.4,
        w2: float = 0.2,
        w3: float = 0.2,
        w4: float = 0.2,
        recency_half_life_seconds: int = 24 * 60 * 60,
    ) -> None:
        self.w1 = float(w1)
        self.w2 = float(w2)
        self.w3 = float(w3)
        self.w4 = float(w4)
        self.recency_half_life_seconds = max(1, int(recency_half_life_seconds))
        self._decay_lambda = math.log(2) / self.recency_half_life_seconds
        self.memories: List[Dict[str, Any]] = []
        self._state_counts: Counter[str] = Counter()

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _to_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _now_timestamp() -> float:
        return datetime.now(timezone.utc).timestamp()

    @staticmethod
    def _extract_fact_key(memory: Dict[str, Any]) -> str:
        metadata = memory.get("metadata") if isinstance(memory.get("metadata"), dict) else {}
        explicit_key = str(metadata.get("key", "")).strip().lower()
        if explicit_key:
            return explicit_key

        text = str(memory.get("text", "")).strip().lower()
        if not text:
            return ""

        if re.search(r"\b(my name is|name:)\b", text):
            return "name"
        if re.search(r"\b(my age is|i am\s+\d{1,3}|age:)\b", text):
            return "age"
        if re.search(r"\b(my goal is|goal:)\b", text):
            return "goal"

        return ""

    def _recency_score(self, last_accessed: float, now_ts: float) -> float:
        if last_accessed <= 0:
            return 0.0

        age_seconds = max(0.0, now_ts - last_accessed)
        return max(0.0, min(1.0, math.exp(-self._decay_lambda * age_seconds)))

    @staticmethod
    def _compress_text(text: str, max_words: int = 16) -> str:
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words]) + " ..."

    def evaluate_memory(self, memory_list: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        now_ts = self._now_timestamp()
        frequencies = [max(0, self._to_int(memory.get("frequency", 0))) for memory in memory_list]
        max_frequency = max(frequencies) if frequencies else 1

        evaluated: List[Dict[str, Any]] = []
        for memory in memory_list:
            item = dict(memory)
            memory_type = str(item.get("type", "general")).strip().lower() or "general"
            key = str(item.get("key", "")).strip().lower() or self._extract_fact_key(item)

            importance_score = max(0.0, min(1.0, self._to_float(item.get("importance_score", 0.0))))
            frequency = max(0, self._to_int(item.get("frequency", 0)))
            normalized_frequency = min(1.0, frequency / max_frequency) if max_frequency > 0 else 0.0

            last_accessed = self._to_float(item.get("last_accessed", 0.0))
            recency_score = self._recency_score(last_accessed=last_accessed, now_ts=now_ts)
            semantic_relevance = max(0.0, min(1.0, self._to_float(item.get("semantic_relevance", 0.0))))

            final_score = (
                (self.w1 * importance_score)
                + (self.w2 * normalized_frequency)
                + (self.w3 * recency_score)
                + (self.w4 * semantic_relevance)
            )
            final_score = max(0.0, min(1.0, final_score))

            previous_state = str(item.get("memory_state", PASSIVE)).strip().upper() or PASSIVE
            if memory_type == "user_fact" and key in {"name", "age", "goal", "profession", "location", "education"}:
                memory_state = ACTIVE
            elif memory_type == "preference" and key in {"interest", "skill"}:
                memory_state = ACTIVE
            elif final_score >= 0.65:
                memory_state = ACTIVE
            elif final_score >= 0.30:
                memory_state = PASSIVE
            else:
                memory_state = ARCHIVED

            item["type"] = memory_type
            item["key"] = key
            item["importance_score"] = importance_score
            item["frequency"] = frequency
            item["normalized_frequency"] = normalized_frequency
            item["recency_score"] = recency_score
            item["semantic_relevance"] = semantic_relevance
            item["final_score"] = final_score
            item["previous_state"] = previous_state
            item["memory_state"] = memory_state

            evaluated.append(item)

        self.memories = evaluated
        self._state_counts = Counter(memory.get("memory_state", PASSIVE) for memory in evaluated)
        return self.memories

    def update_memory_states(self) -> List[Dict[str, Any]]:
        changed: List[Dict[str, Any]] = []
        for memory in self.memories:
            previous_state = str(memory.get("previous_state", PASSIVE)).strip().upper() or PASSIVE
            current_state = str(memory.get("memory_state", PASSIVE)).strip().upper() or PASSIVE
            if previous_state == current_state:
                continue

            changed.append(memory)
            text = str(memory.get("text", "")).strip()
            preview = text if len(text) <= 80 else text[:77] + "..."
            if previous_state == ACTIVE and current_state == PASSIVE:
                print(f"memory_downgraded ACTIVE->PASSIVE: {preview}")
            elif current_state == ARCHIVED:
                print(f"memory_archived {previous_state}->{current_state}: {preview}")
            else:
                print(f"memory_state_changed {previous_state}->{current_state}: {preview}")

        return changed

    def archive_low_value_memories(self, compress: bool = True) -> List[Dict[str, Any]]:
        archived: List[Dict[str, Any]] = []
        for memory in self.memories:
            if str(memory.get("memory_state", PASSIVE)).upper() != ARCHIVED:
                continue
            if compress:
                text = str(memory.get("text", "")).strip()
                memory["archived_summary"] = self._compress_text(text)
            archived.append(memory)
        return archived

    def get_active_memories(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        active = [memory for memory in self.memories if str(memory.get("memory_state", PASSIVE)).upper() == ACTIVE]
        active = sorted(active, key=lambda memory: self._to_float(memory.get("final_score", 0.0)), reverse=True)
        if limit is None:
            return active
        return active[: max(0, int(limit))]

    def get_state_counts(self) -> Dict[str, int]:
        return {
            ACTIVE: int(self._state_counts.get(ACTIVE, 0)),
            PASSIVE: int(self._state_counts.get(PASSIVE, 0)),
            ARCHIVED: int(self._state_counts.get(ARCHIVED, 0)),
        }