"""List the questions a system got wrong, wrongly refused, or answered when it should have
refused, with its route and answer.

Usage:  python eval/show_failures.py [agent|baseline]
"""
import json
import sys
from pathlib import Path

import pandas as pd

EVAL = Path(__file__).resolve().parent
system = sys.argv[1] if len(sys.argv) > 1 else "agent"
tests = {t["id"]: t for t in map(json.loads, (EVAL / "testset.jsonl").read_text().splitlines())}
runs = {r["id"]: r for r in map(json.loads, (EVAL / "results/runs.jsonl").read_text().splitlines())
        if r["system"] == system}
s = pd.read_csv(EVAL / "results/exact_scores.csv")
bad = s[(s.system == system) & ((s.correct == 0) | (s.false_refusal == 1) | (s.refusal_ok == 0))]
for i in bad.id:
    t, r = tests[i], runs[i]
    print(f"--- {i} [{t['category']}] {t['question']}")
    print(f"route: {'+'.join(r['tools'])}   errors: {r['tool_errors'] or 'none'}")
    for tool, q in r["queries"].items():
        if tool != "vector":
            print(f"{tool} query: {q}")
    print(f"expected: {t['reference']}")
    print(f"answer:   {r['answer'][:500]}\n")
print(f"{len(bad)} of {(s.system == system).sum()} questions")
