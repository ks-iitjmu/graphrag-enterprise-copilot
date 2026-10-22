"""Question -> semantic search over 10-K passages, filtered by the companies and fiscal
years named in the question.

    python tools/vector_tool.py "What does NVIDIA say about export controls?"
"""
import sys
from functools import lru_cache
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from core.entities import companies_in, years_in  # noqa: E402
from retrieval.vector_search import VectorStore  # noqa: E402
from tools.base import ToolResult  # noqa: E402

TOP_K = 6


@lru_cache(maxsize=1)
def store() -> VectorStore:
    return VectorStore()


def answer(question: str, k: int = TOP_K, tickers: list | None = None,
           years: list | None = None) -> ToolResult:
    tickers = tickers if tickers is not None else companies_in(question)
    years = years if years is not None else years_in(question)
    res = ToolResult(tool="vector", question=question, attempts=1)
    res.query = f"tickers={tickers or 'all'} years={years or 'all'}"
    per_company = max(2, k // len(tickers)) if len(tickers) > 1 else k
    hits = []
    for t in (tickers or [None]):             # balance passages across named companies
        hits += store().search(question, k=per_company, tickers=[t] if t else None,
                               years=years or None)
    if not hits and years:                    # a year filter can be too strict
        hits = store().search(question, k=k, tickers=tickers or None)
    res.passages = sorted(hits, key=lambda h: -h["score"])[: max(k, len(tickers) * 2)]
    res.evidence = [h["chunk_id"] for h in res.passages]
    if not res.passages:
        res.error = "no matching passages"
    return res


if __name__ == "__main__":
    r = answer(" ".join(sys.argv[1:]))
    print(r.query)
    for p in r.passages:
        print(f"\n[{p['score']:.3f}] {p['header']} ({p['chunk_id']})\n{p['text'][:300]} ...")
