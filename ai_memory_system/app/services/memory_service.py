from typing import Any, Dict, List, Optional
import math
import re
import time

from ..db.chroma_client import collection, ensure_knowledge_base
from .embedding import get_embedding
from .fact_memory import (
    extract_facts,
    update_fact_store,
    promote_used_facts_for_neural_training,
    remove_archived_facts_from_promoted,
)
from .forgetting_agent import ACTIVE, ARCHIVED, PASSIVE, MemoryForgettingAgent
from ..ml.data_logger import log_retrieval
from ..ml.memory_model import predict_importance, load_model, train_from_buffer

ALPHA = 0.2
BETA = 0.1
GAMMA = 0.7
USER_MEMORY_BOOST = 0.05
SIMILARITY_UPDATE_THRESHOLD = 0.6
MAX_MEMORY_UPDATES_PER_QUERY = 4
RECENCY_HALF_LIFE_SECONDS = 24 * 60 * 60
SIMILARITY_WEIGHT = 0.5
IMPORTANCE_WEIGHT = 0.3
RECENCY_WEIGHT = 0.2
TOPK_SIMILARITY_SAFETY = 3
DYNAMIC_THRESHOLD_OFFSET = 0.05
PREFERENCE_BOOST = 0.08
SHORT_TERM_FACT_TTL_SECONDS = 30 * 60
SHORT_TERM_FACT_CACHE_LIMIT = 20

# System C v2: Memory buffer and lifecycle
IMPORTANCE_THRESHOLD = 0.55
BUFFER_SIZE = 10
TRAIN_EPOCHS = 5
THRESHOLD = 0.45
MAX_CONTEXT = 5
RECENCY_DECAY_LAMBDA = math.log(2) / RECENCY_HALF_LIFE_SECONDS
memory_buffer: List[Dict[str, Any]] = []  # In-memory lifecycle buffer
short_term_fact_cache: Dict[str, Dict[str, Any]] = {}


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    safe_metadata: Dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            safe_metadata[key] = value
        else:
            safe_metadata[key] = str(value)
    return safe_metadata


def _base_memory_metadata(source: str, item_type: str, topic: str = "") -> Dict[str, Any]:
    return {
        "source": source,
        "type": item_type,
        "topic": topic,
        "frequency": 0,
        "last_used": 0.0,
        "score": 0.0,
        "created_at": time.time(),
    }


def _recency_component(last_used: float, now: float) -> float:
    if last_used <= 0:
        return 0.0

    age_seconds = max(0.0, now - last_used)
    return math.exp(-RECENCY_DECAY_LAMBDA * age_seconds)


def _memory_score(frequency: int, last_used: float, now: float) -> float:
    normalized_frequency = _normalize_frequency(frequency)
    normalized_recency = _recency_component(last_used, now)
    return (ALPHA * normalized_frequency) + (BETA * normalized_recency)


def _normalize_frequency(frequency: int) -> float:
    safe_frequency = max(0, frequency)
    # Saturating normalization in [0, 1): 0 -> 0, grows with repeated use.
    return safe_frequency / (safe_frequency + 5.0)


def _similarity_from_distance(distance: Any) -> float:
    if distance is None:
        return 0.0

    d = _to_float(distance, default=1.0)
    # Chroma cosine distance is typically in [0, 2]. Convert to a bounded similarity.
    similarity = 1.0 - (d / 2.0)
    return max(0.0, min(1.0, similarity))


def _detect_query_intents(query: str) -> Dict[str, bool]:
    lowered = _normalize_text(query).lower()
    factual_question = any(
        phrase in lowered
        for phrase in (
            "what is ",
            "what are ",
            "who is ",
            "what does ",
            "define ",
            "explain ",
            "tell me about ",
        )
    ) or lowered.endswith("?")

    personal_memory_query = any(
        phrase in lowered
        for phrase in (
            "my name",
            "my age",
            "my preference",
            "my preferences",
            "remember",
            "recall",
            "personal",
            "about me",
            "my favorite",
            "i like",
            "i prefer",
            "i live in",
        )
    )

    return {
        "factual_question": factual_question,
        "personal_memory_query": personal_memory_query,
        "mixed_query": factual_question and personal_memory_query,
    }


def _is_user_memory(metadata: Dict[str, Any]) -> bool:
    return str(metadata.get("type", "")).strip() == "user_fact" or str(metadata.get("tag", "")).strip() == "user_memory"


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    normalized = str(value).strip().lower()
    return normalized in {"1", "true", "yes", "y"}


def _has_user_identity_metadata(metadata: Dict[str, Any]) -> bool:
    return "user_identity" in metadata


def _is_preference_text(text: str) -> bool:
    lowered = _normalize_text(text).lower()
    return any(
        phrase in lowered
        for phrase in (
            "i prefer",
            "my preference",
            "my preferences",
            "my favorite",
            "i like",
            "user prefers",
            "user likes",
        )
    )


def _is_user_identity_text(text: str) -> bool:
    lowered = _normalize_text(text).lower()
    identity_patterns = (
        r"\bmy name is\b",
        r"\bi am\s+\d{1,3}\b",
        r"\bmy age is\b",
        r"\bi'?m\s+\d{1,3}\b",
        r"\buser name is\b",
        r"\buser age is\b",
    )
    return any(re.search(pattern, lowered) for pattern in identity_patterns)


def _classify_memory_type(metadata: Dict[str, Any], text: str) -> str:
    explicit_type = str(metadata.get("memory_type", "")).strip().lower()
    if explicit_type in {"user_fact", "general_knowledge", "preference", "context"}:
        return explicit_type

    doc_type = str(metadata.get("type", "")).strip().lower()
    source = str(metadata.get("source", "")).strip().lower()

    if _has_user_identity_metadata(metadata):
        return "user_fact"
    if _is_user_memory(metadata) or doc_type == "user_fact":
        return "preference" if _is_preference_text(text) else "user_fact"
    if doc_type in {"qa_pair", "question"}:
        return "context"
    if source == "default" or doc_type in {"seed", "fact"}:
        return "general_knowledge"
    if _is_preference_text(text):
        return "preference"

    return "context"


def _is_personal_memory(metadata: Dict[str, Any], memory_type: str) -> bool:
    return (
        memory_type in {"user_fact", "preference"}
        or _is_user_memory(metadata)
        or _has_user_identity_metadata(metadata)
    )


def _normalize_fact_output(document: Any, metadata: Dict[str, Any]) -> str:
    raw_document = _normalize_text(str(document))
    if not raw_document:
        return ""

    fact_value = metadata.get("fact_value")
    if fact_value not in {None, ""} and str(metadata.get("tag", "")).strip() == "fact_candidate":
        return _normalize_text(str(fact_value))

    if str(metadata.get("tag", "")).strip() == "fact_candidate" or _is_personal_memory(metadata, _classify_memory_type(metadata, raw_document)):
        prefix_match = re.match(r"^[A-Za-z][A-Za-z _-]{0,40}:\s*(.+)$", raw_document)
        if prefix_match:
            return _normalize_text(prefix_match.group(1))

    return raw_document


def _tokenize_for_diversity(text: str) -> set[str]:
    normalized = _normalize_text(text).lower()
    return set(re.findall(r"[a-z0-9']+", normalized))


def _is_diverse_buffer_candidate(text: str) -> bool:
    candidate_tokens = _tokenize_for_diversity(text)
    if not candidate_tokens:
        return False

    for item in memory_buffer:
        existing_text = str(item.get("text", ""))
        existing_tokens = _tokenize_for_diversity(existing_text)
        if not existing_tokens:
            continue

        intersection = len(candidate_tokens.intersection(existing_tokens))
        union = len(candidate_tokens.union(existing_tokens))
        if union == 0:
            continue

        # Skip near-duplicates so the buffer keeps varied training signals.
        jaccard = intersection / union
        if jaccard >= 0.9:
            return False

    return True


def _cache_short_term_fact(document: str, metadata: Dict[str, Any], memory_type: str, timestamp: float) -> None:
    normalized_doc = _normalize_text(document)
    if not normalized_doc:
        return

    if memory_type not in {"user_fact", "preference"} and not _has_user_identity_metadata(metadata):
        return

    cache_key = normalized_doc.lower()
    short_term_fact_cache[cache_key] = {
        "document": normalized_doc,
        "metadata": metadata,
        "memory_type": memory_type,
        "cached_at": timestamp,
    }
    _prune_short_term_fact_cache(timestamp)


def _prune_short_term_fact_cache(now: float) -> None:
    stale_keys = [
        key
        for key, value in short_term_fact_cache.items()
        if now - _to_float(value.get("cached_at"), default=0.0) > SHORT_TERM_FACT_TTL_SECONDS
    ]
    for key in stale_keys:
        short_term_fact_cache.pop(key, None)

    if len(short_term_fact_cache) <= SHORT_TERM_FACT_CACHE_LIMIT:
        return

    ordered_items = sorted(
        short_term_fact_cache.items(),
        key=lambda item: _to_float(item[1].get("cached_at"), default=0.0),
        reverse=True,
    )
    keep_keys = {key for key, _ in ordered_items[:SHORT_TERM_FACT_CACHE_LIMIT]}
    for key in list(short_term_fact_cache.keys()):
        if key not in keep_keys:
            short_term_fact_cache.pop(key, None)


def _cached_fact_records_for_query(query_intents: Dict[str, bool], now: float) -> List[Dict[str, Any]]:
    _prune_short_term_fact_cache(now)
    if not (query_intents.get("personal_memory_query") or query_intents.get("mixed_query")):
        return []

    ordered = sorted(
        short_term_fact_cache.values(),
        key=lambda entry: _to_float(entry.get("cached_at"), default=0.0),
        reverse=True,
    )

    records: List[Dict[str, Any]] = []
    for index, entry in enumerate(ordered):
        document = str(entry.get("document", "")).strip()
        if not document:
            continue
        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        memory_type = str(entry.get("memory_type", "context"))
        records.append(
            {
                "doc_id": f"short-term-cache-{index}",
                "document": _normalize_fact_output(document, metadata),
                "metadata": metadata,
                "frequency": 0,
                "similarity": 1.0,
                "norm_freq": 0.0,
                "norm_recency": 1.0,
                "importance": 1.0,
                "final_score": 1.0,
                "memory_type": memory_type,
                "is_personal_memory": True,
                "force_user_identity": _is_truthy(metadata.get("user_identity")),
            }
        )

    return records


def add_to_buffer(
    frequency: float,
    recency: float,
    similarity: float,
    importance: float,
    text: str,
) -> None:
    """
    Store only high-importance chunks into memory buffer.
    
    Args:
        frequency: Normalized frequency (0-1)
        recency: Normalized recency (0-1)
        similarity: Similarity score (0-1)
        importance: Predicted importance score (0-1)
        text: Document text
    """
    normalized_text = _normalize_text(text)
    if not normalized_text:
        return

    if importance <= IMPORTANCE_THRESHOLD:
        return

    existing_texts = {_normalize_text(item.get("text", "")) for item in memory_buffer}
    if normalized_text in existing_texts:
        return

    if not _is_diverse_buffer_candidate(normalized_text):
        return

    memory_buffer.append(
        {
            "frequency": float(frequency),
            "recency": float(recency),
            "similarity": float(similarity),
            "importance": float(importance),
            "text": normalized_text,
            "label": 1,
        }
    )
    print(f"Buffer size: {len(memory_buffer)} / {BUFFER_SIZE}")


def check_and_train() -> bool:
    """
    Auto-train when buffer reaches threshold.
    Clears buffer only on successful training.
    
    Returns:
        True if training was triggered and succeeded, False otherwise
    """
    if len(memory_buffer) < BUFFER_SIZE:
        return False

    snapshot = list(memory_buffer)
    try:
        print(f"Training started: {len(snapshot)} buffered samples")
        trained = train_from_buffer(
            buffer=snapshot,
            epochs=TRAIN_EPOCHS,
            batch_size=min(8, len(snapshot)),
            verbose=True,
        )
        if trained:
            memory_buffer.clear()
            print("Training finished: model saved and buffer cleared")
            return True
        print("Training finished: not enough valid data or training returned False")
        return False
    except Exception as exc:
        # Safety fallback: keep running without crashing
        print(f"Training failed: {exc}")
        return False


def retrieve_relevant_documents(query: str, limit: int = 5, include_archived: bool = False) -> List[str]:
    if not query or not query.strip():
        return []

    ensure_knowledge_base()

    normalized_query = _normalize_text(query)
    query_intents = _detect_query_intents(normalized_query)
    embedding = get_embedding(normalized_query)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=max(limit * 4, 20),
        include=["documents", "metadatas", "distances"],
    )

    ids = results.get("ids") or []
    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []
    distances = results.get("distances") or []
    if not documents or not ids:
        cached_only = _cached_fact_records_for_query(query_intents, time.time())
        if cached_only:
            return [record["document"] for record in cached_only][: max(1, min(MAX_CONTEXT, limit))]
        return []

    now = time.time()
    loaded_model = load_model()  # Load trained model if available
    max_context = max(1, min(MAX_CONTEXT, limit))
    candidate_records = []

    for index, document in enumerate(documents[0]):
        if not document:
            continue

        doc_id = ids[0][index] if ids and ids[0] and index < len(ids[0]) else None
        if not doc_id:
            continue

        raw_metadata = metadatas[0][index] if metadatas and metadatas[0] and index < len(metadatas[0]) else {}
        metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
        frequency = _to_int(metadata.get("frequency"), default=0)
        last_used = _to_float(metadata.get("last_used"), default=0.0)
        distance = distances[0][index] if distances and distances[0] and index < len(distances[0]) else None
        similarity = _similarity_from_distance(distance)
        memory_type = _classify_memory_type(metadata, document)
        is_personal_memory = _is_personal_memory(metadata, memory_type)
        force_user_identity = _has_user_identity_metadata(metadata)

        normalized_frequency = _normalize_frequency(frequency)
        normalized_recency = _recency_component(last_used, now)

        # Compute importance: model-based (if available) or fallback to System B
        importance = None
        if loaded_model is not None:
            try:
                importance = predict_importance(
                    frequency=normalized_frequency,
                    recency=normalized_recency,
                    similarity=similarity,
                    model=loaded_model,
                )
            except Exception:
                importance = None

        # Fallback to System B if model not available or prediction failed
        if importance is None:
            importance = (GAMMA * similarity) + (ALPHA * normalized_frequency) + (BETA * normalized_recency)

        # Keep user-specific memory prominent and preferences slightly boosted.
        if is_personal_memory:
            importance += USER_MEMORY_BOOST
        if memory_type == "preference":
            importance += PREFERENCE_BOOST

        bounded_importance = max(0.0, min(1.0, float(importance)))
        final_score = (
            (SIMILARITY_WEIGHT * similarity)
            + (IMPORTANCE_WEIGHT * bounded_importance)
            + (RECENCY_WEIGHT * normalized_recency)
        )

        candidate_records.append(
            {
                "doc_id": doc_id,
                "document": document,
                "text": document,
                "metadata": metadata,
                "frequency": frequency,
                "similarity": similarity,
                "norm_freq": normalized_frequency,
                "norm_recency": normalized_recency,
                "importance": bounded_importance,
                "importance_score": bounded_importance,
                "final_score": max(0.0, min(1.0, float(final_score))),
                "memory_type": memory_type,
                "is_personal_memory": is_personal_memory,
                "force_user_identity": force_user_identity,
                "used_model": loaded_model is not None,
            }
        )

    # Increased retention: higher weight on recency for new facts, lower importance weight
    # w1=importance (0.30), w2=frequency (0.10), w3=recency (0.40), w4=semantic (0.20)
    forgetting_agent = MemoryForgettingAgent(w1=0.30, w2=0.10, w3=0.40, w4=0.20)
    memory_items: List[Dict[str, Any]] = []
    for record in candidate_records:
        metadata = record["metadata"]
        # Extract key from fact_key (Chroma fact candidates) or key field (general memories)
        extracted_key = str(metadata.get("fact_key", metadata.get("key", ""))).strip().lower()
        memory_items.append(
            {
                "id": record["doc_id"],
                "text": record["document"],
                "importance_score": record["importance"],
                "frequency": record["frequency"],
                "last_accessed": _to_float(metadata.get("last_used"), default=0.0),
                "created_at": _to_float(metadata.get("created_at"), default=now),
                "type": str(metadata.get("type", record["memory_type"] or "general")),
                "key": extracted_key,
                "semantic_relevance": record["similarity"],
                "memory_state": str(metadata.get("memory_state", PASSIVE)).upper(),
                "metadata": metadata,
            }
        )

    evaluated_memories = forgetting_agent.evaluate_memory(memory_items)
    forgetting_agent.update_memory_states()
    archived_memories = forgetting_agent.archive_low_value_memories(compress=True)
    state_counts = forgetting_agent.get_state_counts()
    print(
        "memory_state_counts "
        f"ACTIVE={state_counts.get(ACTIVE, 0)} "
        f"PASSIVE={state_counts.get(PASSIVE, 0)} "
        f"ARCHIVED={state_counts.get(ARCHIVED, 0)}"
    )
    if archived_memories:
        print(f"archived_memories_count={len(archived_memories)}")
        remove_archived_facts_from_promoted(archived_memories)

    evaluated_by_id = {
        str(memory.get("id", "")): memory
        for memory in evaluated_memories
        if memory.get("id")
    }

    for record in candidate_records:
        evaluated = evaluated_by_id.get(str(record["doc_id"]))
        if not evaluated:
            record["memory_state"] = PASSIVE
            continue
        record["memory_state"] = str(evaluated.get("memory_state", PASSIVE)).upper()
        record["final_score"] = max(0.0, min(1.0, _to_float(evaluated.get("final_score"), default=record["final_score"])))
        record["importance"] = max(0.0, min(1.0, _to_float(evaluated.get("importance_score"), default=record["importance"])))
        if evaluated.get("archived_summary"):
            record["archived_summary"] = str(evaluated.get("archived_summary"))

    active_candidates = [record for record in candidate_records if record.get("memory_state") == ACTIVE]
    passive_candidates = [record for record in candidate_records if record.get("memory_state") == PASSIVE]
    archived_candidates = [record for record in candidate_records if record.get("memory_state") == ARCHIVED]

    selected_records = sorted(
        active_candidates,
        key=lambda r: (r["is_personal_memory"], r["final_score"], r["similarity"]),
        reverse=True,
    )[:max_context]

    if len(selected_records) < max_context:
        already_selected = {record["doc_id"] for record in selected_records}
        passive_fill = [
            record
            for record in sorted(passive_candidates, key=lambda r: (r["final_score"], r["similarity"]), reverse=True)
            if record["doc_id"] not in already_selected
        ]
        selected_records.extend(passive_fill[: max_context - len(selected_records)])

    if include_archived and len(selected_records) < max_context:
        already_selected = {record["doc_id"] for record in selected_records}
        archived_fill = [
            record
            for record in sorted(archived_candidates, key=lambda r: (r["final_score"], r["similarity"]), reverse=True)
            if record["doc_id"] not in already_selected
        ]
        selected_records.extend(archived_fill[: max_context - len(selected_records)])

    # Mixed queries should include at least one personal memory when possible.
    if query_intents["mixed_query"] and not any(record["is_personal_memory"] for record in selected_records):
        personal_candidates = [record for record in active_candidates + passive_candidates if record["is_personal_memory"]]
        if personal_candidates:
            selected_records = [
                sorted(personal_candidates, key=lambda r: (r["final_score"], r["similarity"]), reverse=True)[0]
            ] + selected_records
            selected_records = selected_records[:max_context]

    # Consistency fix: include cached personal facts when retrieval misses them.
    cached_records = _cached_fact_records_for_query(query_intents, now)
    if cached_records and not any(record["is_personal_memory"] for record in selected_records):
        selected_records.extend(cached_records)
        selected_records = selected_records[:max_context]

    if not selected_records:
        print("used_similarity_fallback")
        fallback_pool = active_candidates + passive_candidates
        if include_archived:
            fallback_pool += archived_candidates
        fallback_candidates = sorted(fallback_pool, key=lambda r: r["similarity"], reverse=True)[:max_context]
        selected_records = fallback_candidates

    if not selected_records and cached_records:
        selected_records = cached_records[:max_context]

    unique_documents = [
        _normalize_fact_output(record["document"], record["metadata"]).strip()
        for record in selected_records
        if _normalize_fact_output(record["document"], record["metadata"]).strip()
    ]
    print(f"Selected chunks: {len(selected_records)}")

    for record in selected_records:
        _cache_short_term_fact(
            document=record["document"],
            metadata=record["metadata"],
            memory_type=record["memory_type"],
            timestamp=now,
        )

    selected_doc_ids = {record["doc_id"] for record in selected_records if record.get("doc_id")}

    metadata_updates: Dict[str, Dict[str, Any]] = {}
    for record in candidate_records:
        doc_id = record["doc_id"]
        if str(doc_id).startswith("short-term-cache-"):
            continue

        metadata = record["metadata"]
        merged_metadata = {
            **metadata,
            "memory_state": record.get("memory_state", PASSIVE),
            "importance_score": max(0.0, min(1.0, _to_float(record.get("importance"), default=0.0))),
            "final_score": max(0.0, min(1.0, _to_float(record.get("final_score"), default=0.0))),
            "last_evaluated": now,
            "created_at": _to_float(metadata.get("created_at"), default=now),
            "type": str(metadata.get("type", record.get("memory_type", "general"))),
        }
        if record.get("archived_summary"):
            merged_metadata["archived_summary"] = str(record.get("archived_summary"))
        metadata_updates[str(doc_id)] = _sanitize_metadata(merged_metadata)

    if selected_records:
        update_time = time.time()
        updates_done = 0
        used_fact_candidates: List[Dict[str, Any]] = []
        for record in selected_records:
            doc_id = record["doc_id"]
            metadata = record["metadata"]
            frequency = record["frequency"]
            similarity = record["similarity"]
            norm_freq = record["norm_freq"]
            norm_recency = record["norm_recency"]
            importance = record["importance"]
            document = record["document"]

            if record.get("memory_state") == ACTIVE:
                add_to_buffer(
                    frequency=norm_freq,
                    recency=norm_recency,
                    similarity=similarity,
                    importance=importance,
                    text=document,
                )

            if str(doc_id).startswith("short-term-cache-"):
                continue

            existing_metadata = dict(metadata_updates.get(str(doc_id), metadata))
            usage_count = _to_int(existing_metadata.get("usage_count"), default=0) + 1
            existing_metadata["usage_count"] = usage_count
            existing_metadata["last_seen"] = update_time
            metadata_updates[str(doc_id)] = _sanitize_metadata(existing_metadata)

            if str(existing_metadata.get("tag", "")).strip() == "fact_candidate":
                used_fact_candidates.append(
                    {
                        "key": existing_metadata.get("fact_key"),
                        "value": existing_metadata.get("fact_value"),
                        "usage_count": usage_count,
                        "last_seen": update_time,
                    }
                )

            if updates_done >= MAX_MEMORY_UPDATES_PER_QUERY:
                break
            if similarity <= SIMILARITY_UPDATE_THRESHOLD:
                continue

            new_frequency = max(0, frequency) + 1
            new_last_used = update_time
            new_score = _memory_score(new_frequency, new_last_used, update_time)

            existing_metadata = dict(metadata_updates.get(str(doc_id), metadata))
            merged_metadata = {
                **existing_metadata,
                "frequency": new_frequency,
                "last_used": new_last_used,
                "score": new_score,
            }

            metadata_updates[str(doc_id)] = _sanitize_metadata(merged_metadata)

            # Log retrieval with normalized features and importance (1 = high confidence retrieval)
            normalized_frequency = _normalize_frequency(new_frequency)
            normalized_recency = _recency_component(new_last_used, update_time)
            importance_label = 1

            log_retrieval(
                frequency=normalized_frequency,
                recency=normalized_recency,
                similarity=similarity,
                importance=importance_label,
            )

            updates_done += 1

        if used_fact_candidates:
            promote_used_facts_for_neural_training(used_fact_candidates)

    if metadata_updates:
        update_ids = list(metadata_updates.keys())
        update_metadatas = [metadata_updates[doc_id] for doc_id in update_ids]
        collection.update(ids=update_ids, metadatas=update_metadatas)

    # Training labels: 1 for chunks used in final context, 0 for retrieved but unused chunks.
    for record in candidate_records:
        label = 1 if record["doc_id"] in selected_doc_ids else 0
        log_retrieval(
            frequency=record["norm_freq"],
            recency=record["norm_recency"],
            similarity=record["similarity"],
            importance=label,
        )

    # System C v2: Auto-train if buffer threshold reached
    check_and_train()

    return unique_documents


def store_learned_interaction(question: str, answer: str) -> None:
    """Legacy compatibility hook that only updates structured facts."""
    if question:
        update_fact_store(extract_facts(question))


def store_learned_fact(statement: str) -> None:
    """Legacy compatibility hook that only updates structured facts."""
    if statement:
        update_fact_store(extract_facts(statement))