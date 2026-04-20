from __future__ import annotations

import hashlib
import math
import re


EMBEDDING_DIMENSION = 128


def get_embedding(text: str) -> list[float]:
    vector = [0.0] * EMBEDDING_DIMENSION
    tokens = re.findall(r"\w+", text.lower())

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        index = int(digest, 16) % EMBEDDING_DIMENSION
        vector[index] += 1.0

    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]