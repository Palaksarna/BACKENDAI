import uuid
import time
import torch

from ..ml.buffer import add_to_buffer, clear_buffer, get_buffer, ready_to_train
from ..db.chroma_client import collection
from .embedding import get_embedding
from ..ml.neural_model import model
from ..ml.train import train_mlp

MEMORY_THRESHOLD = 0.5
DECAY_RATE = 0.0001  # slower decay
PROMOTION_MIN_USAGE = 3
PROMOTION_MIN_AGE_SECONDS = 60


def should_promote(meta):
    timestamp = meta.get("timestamp")
    if timestamp is None:
        return False

    age = time.time() - float(timestamp)
    return meta.get("usage_count", 0) >= PROMOTION_MIN_USAGE and age > PROMOTION_MIN_AGE_SECONDS


def _find_existing_fact(user_id, text):
    """Return (id, metadata) for an exact user fact match if present."""
    results = collection.get(where={"user_id": user_id})

    ids = results.get("ids") or []
    docs = results.get("documents") or []
    metadatas = results.get("metadatas") or []

    for i, doc in enumerate(docs):
        if doc == text:
            meta = metadatas[i] or {}
            return ids[i], meta

    return None, None


def store_memory(user_id, text):
    if not text or not text.strip():
        return None

    normalized_text = text.strip()
    existing_id, existing_meta = _find_existing_fact(user_id, normalized_text)

    if existing_id:
        updated_metadata = {
            **existing_meta,
            "user_id": user_id,
            # Keep the original timestamp so age keeps increasing naturally.
            "timestamp": existing_meta.get("timestamp", time.time()),
            "usage_count": existing_meta.get("usage_count", 1) + 1,
            "last_seen": time.time(),
        }

        collection.update(ids=[existing_id], metadatas=[updated_metadata])

        if should_promote(updated_metadata):
            add_to_buffer(get_embedding(normalized_text), existing_id)

        return {
            "id": existing_id,
            "text": normalized_text,
            "timestamp": updated_metadata["timestamp"],
            "usage_count": updated_metadata["usage_count"],
            "updated": True,
        }

    embedding = get_embedding(normalized_text)
    memory_id = str(uuid.uuid4())
    timestamp = time.time()

    metadata = {
        "user_id": user_id,
        "timestamp": timestamp,
        "usage_count": 1,
        "last_seen": timestamp,
    }

    collection.add(
        ids=[memory_id],
        embeddings=[embedding],
        documents=[normalized_text],
        metadatas=[metadata]
    )

    return {
        "id": memory_id,
        "text": normalized_text,
        "timestamp": timestamp,
        "usage_count": 1,
        "updated": False,
    }


def retrieve_memory(user_id, query):
    embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=5
    )

    memories = []

    if not results["documents"] or len(results["documents"][0]) == 0:
        return memories

    for i, doc in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i] or {}
        memory_id = results["ids"][0][i]

        if metadata.get("user_id") == user_id:
            memories.append((doc, metadata))

            updated_metadata = {
                **metadata,
                "usage_count": metadata.get("usage_count", 1) + 1
            }

            collection.update(
                ids=[memory_id],
                metadatas=[updated_metadata]
            )

            if should_promote(updated_metadata):
                add_to_buffer(get_embedding(doc), memory_id)

    return memories


def process_memory_buffer():
    if not ready_to_train():
        return []

    buffered_items = get_buffer()
    if not buffered_items:
        return []

    embeddings = [item[0] for item in buffered_items]
    ids = [item[1] for item in buffered_items]

    train_mlp(embeddings)
    collection.delete(ids=ids)
    clear_buffer()

    return ids


def get_neural_context(query):
    query_embedding = get_embedding(query)
    x = torch.tensor(query_embedding, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        mlp_output = model(x).squeeze(0).cpu().tolist()

    return mlp_output


def forget_memory():
    results = collection.get()

    if not results["ids"]:
        return

    ids_to_delete = []

    for i, meta in enumerate(results["metadatas"]):
        if not meta:
            continue

        age = time.time() - meta.get("timestamp", time.time())

        score = meta.get("usage_count", 1) - (DECAY_RATE * age)

        if score < MEMORY_THRESHOLD:
            ids_to_delete.append(results["ids"][i])

    if ids_to_delete:
        collection.delete(ids=ids_to_delete)