"""Semantic search over 10-K chunks.

    from retrieval.vector_search import VectorStore
    hits = VectorStore().search("export controls on AI chips", tickers=["NVDA"], k=5)

Command line:
    python retrieval/vector_search.py "export controls on AI chips" --ticker NVDA --year 2025
"""
import argparse
import re
import sys
from pathlib import Path

import faiss
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import INDEX  # noqa: E402
from core.ollama import QUERY_PREFIX, embed  # noqa: E402


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def near_duplicate(a: set[str], b: set[str], threshold: float = 0.8) -> bool:
    """Jaccard similarity of word sets: 10-Ks repeat paragraphs across sections and years."""
    return len(a & b) / max(len(a | b), 1) >= threshold


class VectorStore:
    def __init__(self, index_dir: Path = INDEX):
        self.index = faiss.read_index(str(index_dir / "faiss.index"))
        self.chunks = pd.read_parquet(index_dir / "chunks.parquet")

    def search(self, query: str, k: int = 5, tickers: list[str] | None = None,
               years: list[int] | None = None, items: list[str] | None = None,
               dedupe: bool = True) -> list[dict]:
        """Top-k chunks by cosine similarity, optionally filtered by company, year, section.

        Filters are applied after the search, so we over-fetch candidates first. With
        dedupe, passages that are near-copies of an already selected one are skipped."""
        q = embed([query], prefix=QUERY_PREFIX)
        filtered = bool(tickers or years or items)
        n = min(self.index.ntotal, k * 50 if filtered else k * 5)
        scores, ids = self.index.search(q, n)
        hits, seen = [], []
        for score, i in zip(scores[0], ids[0]):
            if i < 0:
                continue
            row = self.chunks.iloc[i]
            if tickers and row["ticker"] not in tickers:
                continue
            if years and int(row["fiscal_year"]) not in years:
                continue
            if items and row["item"] not in items:
                continue
            words = _words(row["text"])
            if dedupe and any(near_duplicate(words, w) for w in seen):
                continue
            seen.append(words)
            hits.append({**row.to_dict(), "score": float(score)})
            if len(hits) == k:
                break
        return hits


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--ticker", action="append")
    ap.add_argument("--year", type=int, action="append")
    ap.add_argument("-k", type=int, default=5)
    a = ap.parse_args()
    for h in VectorStore().search(a.query, k=a.k, tickers=a.ticker, years=a.year):
        print(f"\n[{h['score']:.3f}] {h['header']}  ({h['chunk_id']})")
        print(h["text"][:400].replace("\n", " ") + " ...")
