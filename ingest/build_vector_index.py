"""Chunk the parsed 10-K sections, embed them and build a FAISS index.

Each chunk is about CHUNK_CHARS characters, cut at line boundaries, with CHUNK_OVERLAP
characters carried over from the previous chunk. Every chunk is embedded together with
a short header (company, fiscal year, section) so that "NVIDIA export risks" retrieves
NVIDIA's risk factors rather than a similar passage from another company.

Output (data/index/):
  faiss.index      inner-product index over normalised vectors (= cosine similarity)
  chunks.parquet   chunk text and metadata, row i matches vector i
Usage:  python ingest/build_vector_index.py
"""
import json
import sys
import time
from pathlib import Path

import faiss
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import CHUNK_CHARS, CHUNK_OVERLAP, INDEX, SECTIONS  # noqa: E402
from core.ollama import embed  # noqa: E402


def split_long(line: str, size: int) -> list[str]:
    """Split a single over-long line at sentence ends, or hard-cut as a last resort."""
    if len(line) <= size:
        return [line]
    parts, cur = [], ""
    for sentence in line.replace(". ", ".\n").split("\n"):
        if len(cur) + len(sentence) > size and cur:
            parts.append(cur.strip())
            cur = ""
        cur += sentence + " "
    if cur.strip():
        parts.append(cur.strip())
    return [p[i:i + size] for p in parts for i in range(0, len(p), size)]


def chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    lines = [piece for line in text.split("\n") for piece in split_long(line, size)]
    chunks, cur = [], []
    for line in lines:
        if cur and sum(len(x) + 1 for x in cur) + len(line) > size:
            chunks.append("\n".join(cur))
            carry, total = [], 0
            for prev in reversed(cur):                 # overlap: last lines up to `overlap`
                if total + len(prev) > overlap:
                    break
                carry.insert(0, prev)
                total += len(prev) + 1
            cur = carry
        cur.append(line)
    if cur:
        chunks.append("\n".join(cur))
    return [c for c in chunks if len(c) > 80]          # drop fragments


def header(doc: dict, section: dict) -> str:
    return (f"{doc['company']} ({doc['ticker']}) 10-K, fiscal year {doc['fiscal_year']}, "
            f"Item {section['item']}: {section['title']}")


def main():
    rows = []
    for path in sorted(SECTIONS.glob("*.json")):
        doc = json.loads(path.read_text())
        for s in doc["sections"]:
            for k, text in enumerate(chunk_text(s["text"])):
                rows.append({"chunk_id": f"{doc['ticker']}_{doc['fiscal_year']}_{s['item']}_{k:04d}",
                             "ticker": doc["ticker"], "company": doc["company"],
                             "fiscal_year": doc["fiscal_year"], "item": s["item"],
                             "section": s["title"], "url": doc["url"],
                             "header": header(doc, s), "text": text})
    chunks = pd.DataFrame(rows)
    print(f"{len(chunks):,} chunks from {chunks[['ticker', 'fiscal_year']].drop_duplicates().shape[0]} "
          f"reports; median {chunks['text'].str.len().median():.0f} characters")

    t0 = time.time()
    vectors = embed((chunks["header"] + "\n" + chunks["text"]).tolist())
    print(f"Embedded in {time.time() - t0:.0f}s, dimension {vectors.shape[1]}")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    INDEX.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX / "faiss.index"))
    chunks.to_parquet(INDEX / "chunks.parquet", index=False)
    print(f"Wrote {INDEX}/faiss.index ({index.ntotal:,} vectors) and chunks.parquet")
    print(chunks.groupby("ticker").size().rename("chunks").to_string())


if __name__ == "__main__":
    main()
