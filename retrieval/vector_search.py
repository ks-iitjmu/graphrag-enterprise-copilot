"""Semantic search over 10-K chunks.

    from retrieval.vector_search import VectorStore
    hits = VectorStore().search("export controls on AI chips", tickers=["NVDA"], k=5)

Command line:
    python retrieval/vector_search.py "export controls on AI chips" --ticker NVDA --year 2025
"""
import argparse
import sys
from pathlib import Path

import faiss
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import INDEX  # noqa: E402
from core.ollama import QUERY_PREFIX, embed  # noqa: E402


class VectorStore:
    def __init__(self, index_dir: Path = INDEX):
        self.index = faiss.read_index(str(index_dir / "faiss.index"))
        self.chunks = pd.read_parquet(index_dir / "chunks.parquet")

    def search(self, query: str, k: int = 5, tickers: list[str] | None = None,
               years: list[int] | None = None, items: list[str] | None = None) -> list[dict]:
        """Top-k chunks by cosine similarity, optionally filtered by company, year, section.

        Filters are applied after the search, so we over-fetch candidates first."""
        q = embed([query], prefix=QUERY_PREFIX)
        filtered = bool(tickers or years or items)
        n = min(self.index.ntotal, k * 50 if filtered else k)
        scores, ids = self.index.search(q, n)
        hits = []
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
