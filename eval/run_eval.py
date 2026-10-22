"""Run every test question through two systems and record what they did.

  agent      the full copilot: routing, SQL / Cypher / vector tools, citation checks
  baseline   vector-only RAG: the same LLM, answer prompt, citation check and passages,
             but no routing, no graph and no SQL

Results are appended to eval/results/runs.jsonl as each question finishes, so an
interrupted run resumes where it stopped.

Usage:  python eval/run_eval.py [--systems agent baseline] [--limit N]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from agent import graph as agent_graph  # noqa: E402
from tools import vector_tool  # noqa: E402

EVAL = Path(__file__).resolve().parent
RESULTS = EVAL / "results"
RUNS = RESULTS / "runs.jsonl"


def contexts_of(results) -> list[str]:
    """Retrieved contexts as plain strings (what RAGAS judges answers against)."""
    out = []
    for r in results:
        if not r.ok:
            continue
        if r.tool == "vector":
            out += [p["text"] for p in r.passages]
        else:
            out.append(r.as_context())
    return out


def run_agent(question: str) -> dict:
    s = agent_graph.ask(question)
    return {"tools": s["tools"], "answer": s["answer"], "citations": s.get("citations", []),
            "flagged": s.get("removed_citations", []), "contexts": contexts_of(s["results"]),
            "retrieved_chunks": [c for r in s["results"] if r.tool != "sql" for c in r.evidence],
            "queries": {r.tool: r.query for r in s["results"]},
            "tool_errors": {r.tool: r.error for r in s["results"] if r.error},
            "timings": s.get("timings", {})}


def run_baseline(question: str) -> dict:
    t0 = time.time()
    res = vector_tool.answer(question)
    state = {"question": question, "results": [res], "timings": {}}
    state.update(agent_graph.answer(state))
    state.update(agent_graph.check_citations(state))
    return {"tools": ["vector"], "answer": state["answer"], "citations": state["citations"],
            "flagged": state["removed_citations"], "contexts": contexts_of([res]),
            "retrieved_chunks": res.evidence, "queries": {"vector": res.query},
            "tool_errors": {"vector": res.error} if res.error else {},
            "timings": {"total": round(time.time() - t0, 2)}}


SYSTEMS = {"agent": run_agent, "baseline": run_baseline}


def main(systems: list[str], limit: int | None):
    tests = [json.loads(l) for l in (EVAL / "testset.jsonl").read_text().splitlines()]
    if limit:
        tests = tests[:limit]
    RESULTS.mkdir(parents=True, exist_ok=True)
    done = set()
    if RUNS.exists():
        done = {(r["system"], r["id"]) for r in map(json.loads, RUNS.read_text().splitlines())}
    todo = [(s, t) for s in systems for t in tests if (s, t["id"]) not in done]
    print(f"{len(tests)} questions x {len(systems)} systems; {len(todo)} runs to do")
    with RUNS.open("a") as f:
        for k, (system, t) in enumerate(todo, 1):
            t0 = time.time()
            try:
                out = SYSTEMS[system](t["question"])
            except Exception as e:                    # record the failure and keep going
                out = {"tools": [], "answer": "", "citations": [], "flagged": [], "contexts": [],
                       "retrieved_chunks": [], "queries": {}, "tool_errors": {"run": str(e)}, "timings": {}}
            out.update({"system": system, "id": t["id"], "seconds": round(time.time() - t0, 2)})
            f.write(json.dumps(out) + "\n")
            f.flush()
            print(f"  [{k}/{len(todo)}] {system:8s} {t['id']} {t['category']:12s} "
                  f"{out['seconds']:5.1f}s route={'+'.join(out['tools'])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=["agent", "baseline"], choices=list(SYSTEMS))
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    main(a.systems, a.limit)
