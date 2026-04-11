import uuid
import time
from ..db.chroma_client import collection
from .embedding import get_embedding

MEMORY_THRESHOLD = 0.5
DECAY_RATE = 0.0001  # slower decay


def store_memory(user_id, text):
    if not text or not text.strip():
        return None

    embedding = get_embedding(text)
    memory_id = str(uuid.uuid4())
    timestamp = time.time()

    metadata = {
        "user_id": user_id,
        "timestamp": timestamp,
        "usage_count": 1
    }

    collection.add(
        ids=[memory_id],
        embeddings=[embedding],
        documents=[text],
        metadatas=[metadata]
    )

    return {
        "id": memory_id,
        "text": text,
        "timestamp": timestamp
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

    return memories


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