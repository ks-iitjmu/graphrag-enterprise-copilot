"""Minimal client for the local Ollama server (embeddings now, chat in later steps)."""
import sys
from pathlib import Path

import numpy as np
import requests

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import EMBED_MODEL, OLLAMA_URL  # noqa: E402

# nomic-embed-text expects a task prefix: documents and queries are embedded differently
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


def embed(texts: list[str], prefix: str = DOC_PREFIX, batch: int = 64) -> np.ndarray:
    """Embed texts with Ollama; returns L2-normalised float32 vectors (cosine = dot product)."""
    out = []
    for i in range(0, len(texts), batch):
        r = requests.post(f"{OLLAMA_URL}/api/embed", timeout=300,
                          json={"model": EMBED_MODEL,
                                "input": [prefix + t for t in texts[i:i + batch]]})
        r.raise_for_status()
        out.extend(r.json()["embeddings"])
    v = np.asarray(out, dtype="float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True).clip(min=1e-12)
