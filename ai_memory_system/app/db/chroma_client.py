from __future__ import annotations

import chromadb
import time

from ..services.embedding import get_embedding

DEFAULT_DOCUMENTS = [
	"Embeddings turn text into vectors so similar content can be retrieved efficiently.",
	"A vector database stores embedded documents and returns the closest matches for a query.",
	"RAG combines retrieved context with an LLM so answers are grounded in the stored knowledge.",
	"If no strong match is found, the LLM should answer with the retrieved context it has and be explicit about uncertainty.",
]

client = chromadb.EphemeralClient()
collection = client.get_or_create_collection(name="knowledge_base")

_SEEDED = False


def ensure_knowledge_base() -> None:
	global _SEEDED

	if _SEEDED:
		return

	if collection.count() == 0:
		embeddings = [get_embedding(document) for document in DEFAULT_DOCUMENTS]
		collection.add(
			ids=[f"default-doc-{index}" for index in range(len(DEFAULT_DOCUMENTS))],
			documents=DEFAULT_DOCUMENTS,
			embeddings=embeddings,
			metadatas=[
				{
					"source": "default",
					"type": "seed",
					"frequency": 0,
					"last_used": 0.0,
					"score": 0.0,
					"created_at": time.time(),
				}
				for _ in DEFAULT_DOCUMENTS
			],
		)

	_SEEDED = True