from __future__ import annotations

import json
import re
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence
import time
import uuid

from ..ml.neural_fact_memory import (
    neural_fact_model_ready,
    retrieve_neural_facts,
    train_neural_fact_memory,
)
from ..db.chroma_client import collection, ensure_knowledge_base
from .embedding import get_embedding


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
FACT_STORE_PATH = DATA_DIR / "fact_store.json"
PROMOTED_FACT_STORE_PATH = DATA_DIR / "promoted_fact_store.json"
PROMOTION_THRESHOLD = 2
FACT_USAGE_PROMOTION_THRESHOLD = 2


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def _normalize_value(key: str, value: Any) -> str:
    if key == "age":
        text = _normalize_text(str(value))
        match = re.search(r"\b(\d{1,3})\b", text)
        if match:
            return match.group(1)
        return text.casefold()
    return _normalize_text(str(value)).casefold()


def _display_value(key: str, value: Any) -> Any:
    if isinstance(value, str):
        return _normalize_text(value)
    return value


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default

    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, payload: Any) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=DATA_DIR, encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        temp_path = Path(handle.name)

    temp_path.replace(path)


def _load_fact_records() -> List[Dict[str, Any]]:
    records = _read_json(FACT_STORE_PATH, [])
    return records if isinstance(records, list) else []


def _save_fact_records(records: Sequence[Dict[str, Any]]) -> None:
    _write_json(FACT_STORE_PATH, list(records))


def _load_promoted_records() -> List[Dict[str, Any]]:
    records = _read_json(PROMOTED_FACT_STORE_PATH, [])
    return records if isinstance(records, list) else []


def _save_promoted_records(records: Sequence[Dict[str, Any]]) -> None:
    _write_json(PROMOTED_FACT_STORE_PATH, list(records))


def _fact_label(key: str) -> str:
    labels = {
        "name": "Name",
        "age": "Age",
        "interest": "Interests",
    }
    return labels.get(key, key.replace("_", " ").title())


def _split_interest_values(raw_value: str) -> List[str]:
    parts = [part.strip(" ,.;!?\"'\n\t") for part in re.split(r"\s*(?:,|\band\b|&|\bplus\b)\s*", raw_value, flags=re.IGNORECASE) if part.strip()]
    return [part for part in parts if part]


def _strip_fact_prefix(text: Any) -> str:
    normalized = _normalize_text(str(text))
    if not normalized:
        return ""

    prefix_match = re.match(r"^[A-Za-z][A-Za-z _-]{0,40}:\s*(.+)$", normalized)
    if prefix_match:
        return _normalize_text(prefix_match.group(1))

    return normalized


def _fact_document_text(key: str, value: Any) -> str:
    return _strip_fact_prefix(_display_value(key, value))


def _clean_fact_value(key: str, value: str) -> str:
    text = _normalize_text(value).rstrip(".,!?")
    if not text:
        return ""

    if key == "name":
        text = re.split(r"\s+(?:and|but|because|so)\s+(?=i\b|my\b|we\b|they\b|he\b|she\b|you\b)", text, maxsplit=1, flags=re.IGNORECASE)[0]
    elif key in {"location", "profession", "education", "goal", "skill", "interest"}:
        text = re.split(r"\s+(?:and|but|because|so)\s+(?=i\b|my\b|we\b|they\b|he\b|she\b|you\b)", text, maxsplit=1, flags=re.IGNORECASE)[0]

    return _normalize_text(text).rstrip(".,!?")


def _fact_memory_type(key: str) -> str:
    if key in {"interest", "skill", "preference"}:
        return "preference"
    return "user_fact"


def _upsert_vector_fact_candidate(key: str, value: Any) -> None:
    ensure_knowledge_base()

    normalized_value = _normalize_value(key, value)
    if not normalized_value:
        return

    fact_value = _display_value(key, value)
    now_ts = time.time()
    existing = collection.get(
        where={
            "$and": [
                {"tag": "fact_candidate"},
                {"fact_key": key},
                {"fact_value_normalized": normalized_value},
            ]
        },
        include=["metadatas"],
    )

    existing_ids = existing.get("ids") or []
    existing_metadatas = existing.get("metadatas") or []
    if existing_ids:
        doc_id = existing_ids[0]
        metadata = existing_metadatas[0] if existing_metadatas else {}
        safe_metadata = dict(metadata) if isinstance(metadata, dict) else {}
        mention_count = int(safe_metadata.get("mention_count", 0)) + 1
        updated_metadata = {
            **safe_metadata,
            "mention_count": mention_count,
            "last_seen": now_ts,
            "fact_key": key,
            "fact_value": fact_value,
            "fact_value_normalized": normalized_value,
            "tag": "fact_candidate",
            "memory_type": _fact_memory_type(key),
            "type": "user_fact",
        }
        collection.update(ids=[doc_id], metadatas=[updated_metadata])
        return

    document = _fact_document_text(key, fact_value)
    collection.add(
        ids=[f"fact-candidate-{uuid.uuid4()}"],
        documents=[document],
        embeddings=[get_embedding(document)],
        metadatas=[
            {
                "source": "fact_input",
                "type": "user_fact",
                "memory_type": _fact_memory_type(key),
                "tag": "fact_candidate",
                "fact_key": key,
                "fact_value": fact_value,
                "fact_value_normalized": normalized_value,
                "mention_count": 1,
                "usage_count": 0,
                "frequency": 0,
                "last_used": 0.0,
                "last_seen": now_ts,
                "created_at": now_ts,
                "score": 0.0,
                "memory_state": "PASSIVE",
            }
        ],
    )


def extract_facts(message: str) -> List[Dict[str, Any]]:
    normalized = _normalize_text(message)
    if not normalized:
        return []

    facts: List[Dict[str, Any]] = []

    name_patterns = [
        r"(?:my name is|call me|i am called)\s+([A-Za-z][A-Za-z .'-]{0,80}?)(?=\s+(?:and|but|because|so)\s+(?=i\b|my\b|we\b|they\b|he\b|she\b|you\b)|[.,!?]|$)",
    ]
    age_patterns = [
        r"(?:my age is|i am|i'm|im)\s+(\d{1,3})(?:\s+years? old)?",
    ]
    interest_patterns = [
        r"(?:i like|i love|i enjoy|i'm into|im into|i am into)\s+([^.!?]+)",
        r"(?:my favorite(?: [a-z]+)? is)\s+([^.!?]+)",
    ]
    profession_patterns = [
        r"(?:i'm|im|i am)\s+a\s+([^,.\n]+?)(?:\s+(?:at|in|for))?(?:[.,\n]|$)",
        r"(?:i work as|i'm employed as|my job is|my profession is)\s+([^.!?]+)",
        r"(?:my role is|i work in)\s+([^.!?]+)",
    ]
    location_patterns = [
        r"(?:i live in|i'm from|i'm based in|i'm located in|i reside in)\s+([^.!?]+)",
        r"(?:i'm in|i am in)\s+([A-Z][a-zA-Z\s]+?)(?:[.,\n]|$)",
    ]
    education_patterns = [
        r"(?:i studied at|i graduated from|i went to|i attended|i study at)\s+([^.!?]+)",
        r"(?:i have a degree in|my degree is in|i majored in)\s+([^.!?]+)",
    ]
    goal_patterns = [
        r"(?:my goal is|my aim is|i want to|i aspire to|i aim to|i'm trying to)\s+([^.!?]+)",
        r"(?:my goal:)\s+([^.!?]+)",
    ]
    skill_patterns = [
        r"(?:i'm skilled in|i'm good at|i can|i'm proficient in|my skills are)\s+([^.!?]+)",
        r"(?:my skills include|i know)\s+([^.!?]+)",
    ]

    for pattern in name_patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            sentence = _normalize_text(match.group(0)).rstrip(".,!?")
            if sentence:
                facts.append({"key": "name", "value": sentence})
            break

    for pattern in age_patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            sentence = _normalize_text(match.group(0)).rstrip(".,!?")
            if sentence:
                facts.append({"key": "age", "value": sentence})
            break

    for pattern in interest_patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            raw_interest = _clean_fact_value("interest", match.group(1))
            prefix = _normalize_text(match.group(0)[: max(0, match.start(1) - match.start(0))]).rstrip(".,!?")
            for interest in _split_interest_values(raw_interest):
                if interest:
                    sentence = _normalize_text(f"{prefix} {interest}".strip()).rstrip(".,!?")
                    facts.append({"key": "interest", "value": sentence})
            break

    for pattern in profession_patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            sentence = _normalize_text(match.group(0)).rstrip(".,!?")
            if sentence:
                facts.append({"key": "profession", "value": sentence})
            break

    for pattern in location_patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            sentence = _normalize_text(match.group(0)).rstrip(".,!?")
            if sentence:
                facts.append({"key": "location", "value": sentence})
            break

    for pattern in education_patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            sentence = _normalize_text(match.group(0)).rstrip(".,!?")
            if sentence:
                facts.append({"key": "education", "value": sentence})
            break

    for pattern in goal_patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            sentence = _normalize_text(match.group(0)).rstrip(".,!?")
            if sentence:
                facts.append({"key": "goal", "value": sentence})
            break

    for pattern in skill_patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            raw_skills = _clean_fact_value("skill", match.group(1))
            prefix = _normalize_text(match.group(0)[: max(0, match.start(1) - match.start(0))]).rstrip(".,!?")
            for skill in _split_interest_values(raw_skills):
                if skill and len(skill) > 1:
                    sentence = _normalize_text(f"{prefix} {skill}".strip()).rstrip(".,!?")
                    facts.append({"key": "skill", "value": sentence})
            break

    unique: List[Dict[str, Any]] = []
    seen = set()
    for fact in facts:
        key = str(fact.get("key", "")).strip().lower()
        value = fact.get("value")
        normalized_value = _normalize_value(key, value)
        signature = (key, normalized_value)
        if not key or not normalized_value or signature in seen:
            continue
        seen.add(signature)
        unique.append({"key": key, "value": value})

    return unique


def update_fact_store(facts: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not facts:
        return get_promoted_facts()

    records = _load_fact_records()
    now = _now_iso()

    for fact in facts:
        key = str(fact.get("key", "")).strip().lower()
        value = fact.get("value")
        if not key or value is None:
            continue

        normalized_value = _normalize_value(key, value)
        if not normalized_value:
            continue

        _upsert_vector_fact_candidate(key, value)

        record = next(
            (
                item
                for item in records
                if str(item.get("key", "")).strip().lower() == key
                and str(item.get("normalized_value", "")).strip().casefold() == normalized_value
            ),
            None,
        )

        if record is None:
            record = {
                "key": key,
                "value": _display_value(key, value),
                "normalized_value": normalized_value,
                "frequency": 0,
                "first_seen": now,
                "last_seen": now,
                "promoted": False,
            }
            records.append(record)

        record["value"] = _display_value(key, value)
        record["normalized_value"] = normalized_value
        record["frequency"] = int(record.get("frequency", 0)) + 1
        record["last_seen"] = now

    _save_fact_records(records)

    return get_promoted_facts()


def promote_used_facts_for_neural_training(used_fact_memories: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Promote consistently used facts to neural memory training set."""
    promoted_records = _load_promoted_records()
    now = _now_iso()

    for item in used_fact_memories:
        key = str(item.get("key", "")).strip().lower()
        value = item.get("value")
        usage_count = int(item.get("usage_count", 0))
        if not key or value is None or usage_count < FACT_USAGE_PROMOTION_THRESHOLD:
            continue

        normalized_value = _normalize_value(key, value)
        if not normalized_value:
            continue

        record = next(
            (
                entry
                for entry in promoted_records
                if str(entry.get("key", "")).strip().lower() == key
                and str(entry.get("normalized_value", "")).strip().casefold() == normalized_value
            ),
            None,
        )

        if record is None:
            record = {
                "key": key,
                "value": _display_value(key, value),
                "normalized_value": normalized_value,
                "frequency": usage_count,
                "first_seen": now,
                "last_seen": item.get("last_seen", now),
                "promoted": True,
            }
            promoted_records.append(record)
        else:
            record["value"] = _display_value(key, value)
            record["frequency"] = max(int(record.get("frequency", 0)), usage_count)
            record["last_seen"] = item.get("last_seen", now)
            record["promoted"] = True

    _save_promoted_records(promoted_records)
    try:
        train_neural_fact_memory(promoted_records, verbose=False)
    except Exception:
        pass

    return get_promoted_facts()


def remove_archived_facts_from_promoted(archived_memories: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Forget archived facts by removing them from promoted + neural fact bank source."""
    promoted_records = _load_promoted_records()
    if not promoted_records:
        return []

    archived_signatures = set()
    for memory in archived_memories:
        metadata = memory.get("metadata") if isinstance(memory.get("metadata"), dict) else {}
        key = str(metadata.get("fact_key", memory.get("key", ""))).strip().lower()
        value = metadata.get("fact_value", memory.get("value"))
        if not key or value is None:
            continue
        archived_signatures.add((key, _normalize_value(key, value)))

    if not archived_signatures:
        return get_promoted_facts()

    filtered = [
        record
        for record in promoted_records
        if (str(record.get("key", "")).strip().lower(), str(record.get("normalized_value", "")).strip().casefold())
        not in archived_signatures
    ]

    _save_promoted_records(filtered)
    try:
        train_neural_fact_memory(filtered, verbose=False)
    except Exception:
        pass

    return get_promoted_facts()


def get_fact_debug_snapshot(query: str = "", limit: int = 20) -> Dict[str, Any]:
    """Expose fact lifecycle status for debugging and verification."""
    ensure_knowledge_base()

    raw = collection.get(where={"tag": "fact_candidate"}, include=["documents", "metadatas"]) or {}
    ids = raw.get("ids") or []
    documents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []

    max_items = max(1, int(limit))
    candidates: List[Dict[str, Any]] = []
    for index, doc_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) and isinstance(metadatas[index], dict) else {}
        document = documents[index] if index < len(documents) else ""
        candidates.append(
            {
                "id": doc_id,
                "text": _strip_fact_prefix(document),
                "key": str(metadata.get("fact_key", "")).strip().lower(),
                "value": metadata.get("fact_value"),
                "mention_count": int(metadata.get("mention_count", 0)),
                "usage_count": int(metadata.get("usage_count", 0)),
                "memory_state": str(metadata.get("memory_state", "PASSIVE")).upper(),
                "last_seen": metadata.get("last_seen"),
                "last_used": metadata.get("last_used"),
                "is_neural_promoted": False,
            }
        )

    promoted_records = get_promoted_facts()
    promoted_signatures = {
        (
            str(record.get("key", "")).strip().lower(),
            str(record.get("normalized_value", "")).strip().casefold(),
        )
        for record in promoted_records
    }

    for candidate in candidates:
        key = str(candidate.get("key", "")).strip().lower()
        value = candidate.get("value")
        signature = (key, _normalize_value(key, value)) if key and value is not None else None
        candidate["is_neural_promoted"] = bool(signature and signature in promoted_signatures)

    candidates.sort(key=lambda item: (int(item.get("usage_count", 0)), int(item.get("mention_count", 0))), reverse=True)
    trimmed_candidates = candidates[:max_items]

    state_counts = {"ACTIVE": 0, "PASSIVE": 0, "ARCHIVED": 0}
    for item in candidates:
        state = str(item.get("memory_state", "PASSIVE")).upper()
        if state not in state_counts:
            state_counts[state] = 0
        state_counts[state] += 1

    neural_preview = retrieve_neural_facts(query, top_k=min(5, max_items)) if query else []

    return {
        "counts": {
            "fact_candidates": len(candidates),
            "promoted_facts": len(promoted_records),
            "states": state_counts,
        },
        "neural_model_ready": neural_fact_model_ready(),
        "top_candidates": trimmed_candidates,
        "promoted_facts": promoted_records[:max_items],
        "neural_preview": neural_preview,
    }


def get_promoted_facts() -> List[Dict[str, Any]]:
    promoted_records = _load_promoted_records()
    if promoted_records:
        return sorted(promoted_records, key=lambda item: str(item.get("last_seen", "")), reverse=True)

    records = [record for record in _load_fact_records() if bool(record.get("promoted", False))]
    if records:
        _save_promoted_records(records)
    return sorted(records, key=lambda item: str(item.get("last_seen", "")), reverse=True)


def get_active_facts() -> List[Dict[str, Any]]:
    """Retrieve ALL ACTIVE facts from Chroma (not just promoted ones).
    Includes new facts mentioned but not yet promoted, ensuring they appear in LLM context."""
    ensure_knowledge_base()
    
    try:
        results = collection.get(
            where={"$and": [{"tag": "fact_candidate"}, {"memory_state": ACTIVE}]},
            include=["metadatas", "documents"]
        )
        
        if not results or not results.get("ids"):
            return []
        
        records: List[Dict[str, Any]] = []
        for doc_id, metadata, doc_text in zip(results["ids"], results["metadatas"], results["documents"]):
            record: Dict[str, Any] = dict(metadata or {})
            record["id"] = doc_id
            record["text"] = doc_text
            records.append(record)
        
        return sorted(records, key=lambda item: str(item.get("last_seen", "")), reverse=True)
    except Exception:
        return []


def _merge_promoted_facts(facts: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for fact in facts:
        key = str(fact.get("key", "")).strip().lower()
        if not key:
            continue
        grouped[key].append(dict(fact))
    return grouped


def _format_profile_lines(promoted_facts: Sequence[Dict[str, Any]]) -> List[str]:
    grouped = _merge_promoted_facts(promoted_facts)
    lines: List[str] = []

    if "name" in grouped:
        latest = max(grouped["name"], key=lambda item: str(item.get("last_seen", "")))
        lines.append(f"- Name: {latest.get('value', '')}")

    if "age" in grouped:
        latest = max(grouped["age"], key=lambda item: str(item.get("last_seen", "")))
        lines.append(f"- Age: {latest.get('value', '')}")

    if "interest" in grouped:
        interests: List[str] = []
        seen = set()
        for fact in sorted(grouped["interest"], key=lambda item: str(item.get("last_seen", ""))):
            value = str(fact.get("value", "")).strip()
            if not value:
                continue
            normalized = value.casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            interests.append(value)
        if interests:
            lines.append(f"- Interests: {', '.join(interests)}")

    if "profession" in grouped:
        latest = max(grouped["profession"], key=lambda item: str(item.get("last_seen", "")))
        lines.append(f"- Profession: {latest.get('value', '')}")

    if "location" in grouped:
        latest = max(grouped["location"], key=lambda item: str(item.get("last_seen", "")))
        lines.append(f"- Location: {latest.get('value', '')}")

    if "education" in grouped:
        latest = max(grouped["education"], key=lambda item: str(item.get("last_seen", "")))
        lines.append(f"- Education: {latest.get('value', '')}")

    if "goal" in grouped:
        goals: List[str] = []
        seen = set()
        for fact in sorted(grouped["goal"], key=lambda item: str(item.get("last_seen", ""))):
            value = str(fact.get("value", "")).strip()
            if not value:
                continue
            normalized = value.casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            goals.append(value)
        if goals:
            lines.append(f"- Goals: {', '.join(goals)}")

    if "skill" in grouped:
        skills: List[str] = []
        seen = set()
        for fact in sorted(grouped["skill"], key=lambda item: str(item.get("last_seen", ""))):
            value = str(fact.get("value", "")).strip()
            if not value:
                continue
            normalized = value.casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            skills.append(value)
        if skills:
            lines.append(f"- Skills: {', '.join(skills)}")

    handled_keys = {"name", "age", "interest", "profession", "location", "education", "goal", "skill"}
    for key in sorted(k for k in grouped.keys() if k not in handled_keys):
        latest = max(grouped[key], key=lambda item: str(item.get("last_seen", "")))
        lines.append(f"- {_fact_label(key)}: {latest.get('value', '')}")

    return lines


def build_prompt(
    message: str,
    retrieved_context: Sequence[str],
    promoted_facts: Sequence[Dict[str, Any]],
    neural_facts: Sequence[Dict[str, Any]] | None = None,
) -> str:
    context_lines = [f"- {chunk}" for chunk in retrieved_context if str(chunk).strip()]
    neural_fact_lines = []
    for item in (neural_facts or []):
        fact_text = str(item.get("fact_text", "")).strip()
        if fact_text:
            neural_fact_lines.append(f"- {fact_text}")

    context_section = chr(10).join(context_lines) if context_lines else "(No relevant context)"
    neural_section = chr(10).join(neural_fact_lines) if neural_fact_lines else "(No learned facts)"

    return f"""You are a helpful assistant.

Learned Facts (from previous interactions):
{neural_section}

Retrieved Context:
{context_section}

User Message:
{message}

Answer:"""