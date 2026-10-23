"""Minimal client for the local Ollama server: embeddings and structured (JSON) chat."""
import json
import sys
from pathlib import Path

import numpy as np
import requests

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import EMBED_MODEL, LLM_MODEL, OLLAMA_URL  # noqa: E402

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


def chat_json(system: str, user: str, schema: dict, think: bool = False,
              num_ctx: int = 8192, max_tokens: int = 600, timeout: int = 300) -> dict:
    """Ask the local LLM for JSON that matches `schema` (Ollama structured outputs).

    Temperature 0 for repeatable extraction; "thinking" off by default for speed. Local
    models occasionally emit whitespace until the length limit, cutting the JSON off; such
    answers are retried with a larger limit and a little randomness."""
    last = None
    for attempt in range(3):
        r = requests.post(f"{OLLAMA_URL}/api/chat", timeout=timeout, json={
            "model": LLM_MODEL, "stream": False, "think": think, "format": schema,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": 0 if attempt == 0 else 0.3, "num_ctx": num_ctx,
                        "num_predict": max_tokens * (attempt + 1)},
        })
        r.raise_for_status()
        try:
            return json.loads(r.json()["message"]["content"])
        except json.JSONDecodeError as e:
            last = e
    raise last
