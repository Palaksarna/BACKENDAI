from __future__ import annotations

import json
import re
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
FACT_STORE_PATH = DATA_DIR / "fact_store.json"
PROMOTED_FACT_STORE_PATH = DATA_DIR / "promoted_fact_store.json"
PROMOTION_THRESHOLD = 2


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def _normalize_value(key: str, value: Any) -> str:
    if key == "age":
        try:
            return str(int(value))
        except (TypeError, ValueError):
            return _normalize_text(str(value)).casefold()
    return _normalize_text(str(value)).casefold()


def _display_value(key: str, value: Any) -> Any:
    if key == "age":
        return int(value)
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


def extract_facts(message: str) -> List[Dict[str, Any]]:
    normalized = _normalize_text(message)
    if not normalized:
        return []

    facts: List[Dict[str, Any]] = []

    name_patterns = [
        r"(?:my name is|call me|i am called)\s+([A-Za-z][A-Za-z .'-]{0,80})",
    ]
    age_patterns = [
        r"(?:my age is|i am|i'm|im)\s+(\d{1,3})(?:\s+years? old)?",
    ]
    interest_patterns = [
        r"(?:i like|i love|i enjoy|i'm into|im into|i am into)\s+([^.!?]+)",
        r"(?:my favorite(?: [a-z]+)? is)\s+([^.!?]+)",
    ]

    for pattern in name_patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            value = _normalize_text(match.group(1)).rstrip(".,!?")
            if value:
                facts.append({"key": "name", "value": _display_value("name", value)})
            break

    for pattern in age_patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            facts.append({"key": "age", "value": int(match.group(1))})
            break

    for pattern in interest_patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            raw_interest = _normalize_text(match.group(1)).rstrip(".,!?")
            for interest in _split_interest_values(raw_interest):
                if interest:
                    facts.append({"key": "interest", "value": _display_value("interest", interest)})
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

        if int(record["frequency"]) >= PROMOTION_THRESHOLD:
            record["promoted"] = True

    _save_fact_records(records)
    _save_promoted_records([record for record in records if bool(record.get("promoted", False))])

    return get_promoted_facts()


def get_promoted_facts() -> List[Dict[str, Any]]:
    promoted_records = _load_promoted_records()
    if promoted_records:
        return sorted(promoted_records, key=lambda item: str(item.get("last_seen", "")), reverse=True)

    records = [record for record in _load_fact_records() if bool(record.get("promoted", False))]
    if records:
        _save_promoted_records(records)
    return sorted(records, key=lambda item: str(item.get("last_seen", "")), reverse=True)


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

    handled_keys = {"name", "age", "interest"}
    for key in sorted(k for k in grouped.keys() if k not in handled_keys):
        latest = max(grouped[key], key=lambda item: str(item.get("last_seen", "")))
        lines.append(f"- {_fact_label(key)}: {latest.get('value', '')}")

    return lines


def build_prompt(
    message: str,
    retrieved_context: Sequence[str],
    promoted_facts: Sequence[Dict[str, Any]],
) -> str:
    profile_lines = _format_profile_lines(promoted_facts)
    context_lines = [f"- {chunk}" for chunk in retrieved_context if str(chunk).strip()]

    return f"""You are a helpful assistant. Use the user profile as persistent memory and the retrieved context as grounding.
Do not invent facts, and do not overwrite the profile unless the user explicitly provides a new fact.
If the profile or context is irrelevant, answer from general knowledge.
Be concise and natural.

User Profile:
{chr(10).join(profile_lines) if profile_lines else '- No promoted facts yet.'}

Retrieved Context:
{chr(10).join(context_lines) if context_lines else '- No relevant context available.'}

User Message:
{message}

Answer:"""