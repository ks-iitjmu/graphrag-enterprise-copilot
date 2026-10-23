"""Score eval/results/runs.jsonl and write eval/results/summary.md.

Exact checks (no judge needed)
  route        the system used every tool the question needs (agent only)
  values       expected numbers appear in the answer (tolerance for rounding)
  entities     expected companies / names appear in the answer
  correct      all expected values AND entities present
  refusal      unanswerable questions: the answer says the data does not contain it;
               answerable questions: the share wrongly refused
  citations    share of answers with at least one valid citation; share of citations
               that pointed to sources never retrieved
  source_hit   text questions: the passage the question was written from was retrieved

RAGAS 0.4 metrics (judged by the local LLM; requests are routed to Ollama's native API
with the output schema enforced, see eval/ollama_judge.py)
  faithfulness, answer_relevancy, context_precision (with reference), context_recall

Usage:  python eval/score.py                 exact checks only (fast)
        python eval/score.py --ragas         + RAGAS (slow; cached in results/ragas.jsonl)
        python eval/score.py --ragas --smoke RAGAS on 2 answers, to test the setup
"""
import argparse
import asyncio
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from agent.graph import NOT_FOUND  # noqa: E402
from config import EMBED_MODEL, LLM_MODEL, OLLAMA_URL  # noqa: E402

EVAL = Path(__file__).resolve().parent
RESULTS = EVAL / "results"
RAGAS_CACHE = RESULTS / "ragas.jsonl"
RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
ENTITY_ALIASES = {"meta platforms": ["meta"], "alphabet": ["alphabet", "google"],
                  "tsmc": ["tsmc", "taiwan semiconductor"], "united states": ["united states", "u.s.", "us "]}
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
BIG_NUMBER = re.compile(r"(?<![\d.])-?\d{1,3}(?:,\d{3}){3,}(?![\d,])")   # 62,360,000,000


def readable_numbers(context: str) -> str:
    """Add the billions form next to raw dollar amounts in SQL rows, so the judge can match
    '$62.4 billion' in an answer to '62,360,000,000' in the data. No new information."""
    def add(m):
        v = float(m.group(0).replace(",", ""))
        return f"{m.group(0)} (= ${v / 1e9:,.1f} billion)"
    return BIG_NUMBER.sub(add, context)


def numbers_in(text: str) -> list[float]:
    out = []
    for m in NUMBER.findall(text):
        try:
            v = float(m.replace(",", ""))
        except ValueError:
            continue
        out += [v, v / 1e9] if abs(v) >= 1e6 else [v]   # full dollars -> billions too
    return out


def has_value(text: str, expected: float) -> bool:
    # References are rounded to one decimal place, so an answer that rounds the same
    # underlying number the other way (14.65 -> 14.6 vs 14.7) must still count
    tol = max(0.101, 0.006 * abs(expected))
    return any(abs(v - expected) <= tol for v in numbers_in(text))


def has_entity(text: str, entity: str) -> bool:
    t = text.lower()
    names = ENTITY_ALIASES.get(entity.lower(), [entity.lower()])
    return any(n in t for n in names)


def exact_scores(test: dict, run: dict) -> dict:
    ans = run["answer"] or ""
    vals = [has_value(ans, v) for v in test["expected_values"]]
    ents = [has_entity(ans, e) for e in test["expected_entities"]]
    refused = bool(NOT_FOUND.search(ans)) or not ans.strip()
    row = {"id": test["id"], "system": run["system"], "category": test["category"],
           "seconds": run["seconds"],
           "citations": len(run["citations"]), "flagged": len(run["flagged"])}
    if test["category"] == "unanswerable":
        row["refusal_ok"] = float(refused)
        return row
    if run["system"] == "agent":            # the baseline always uses vector search
        row["route_ok"] = float(set(test["expected_tools"]) <= set(run["tools"]))
    row["values"] = sum(vals) / len(vals) if vals else None
    row["entities"] = sum(ents) / len(ents) if ents else None
    row["correct"] = float(all(vals) and all(ents)) if (vals or ents) else None
    row["false_refusal"] = float(refused)
    row["cited"] = float(len(run["citations"]) > 0)
    if test.get("source_chunk"):
        row["source_hit"] = float(test["source_chunk"] in run["retrieved_chunks"])
    return row


# ------------------------------------------------------------------------------ RAGAS
def ragas_metrics(temperature: float = 0.0):
    """RAGAS judge = the local LLM. Requests go through OllamaNativeTransport, which sends
    them to Ollama's native API with the output schema enforced (see eval/ollama_judge.py);
    through the OpenAI-compatible endpoint the schema was silently ignored and the judge
    sometimes answered off-format, which made exactly the low-scoring cases fail."""
    import httpx
    import instructor
    from openai import AsyncOpenAI
    from ragas.embeddings import OpenAIEmbeddings
    from ragas.llms import InstructorLLM
    from ragas.metrics.collections import (AnswerRelevancy, ContextPrecisionWithReference,
                                           ContextRecall, Faithfulness)
    from eval.ollama_judge import OllamaNativeTransport
    transport = OllamaNativeTransport(OLLAMA_URL)
    client = AsyncOpenAI(base_url=f"{OLLAMA_URL}/v1", api_key="ollama", timeout=900,
                         http_client=httpx.AsyncClient(transport=transport, timeout=900))
    patched = instructor.from_openai(client, mode=instructor.Mode.JSON_SCHEMA)
    llm = InstructorLLM(client=patched, model=LLM_MODEL, provider="openai",
                        temperature=temperature, max_tokens=4096)
    emb = OpenAIEmbeddings(client=client, model=EMBED_MODEL)
    return {"faithfulness": Faithfulness(llm=llm),
            "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=emb),
            "context_precision": ContextPrecisionWithReference(llm=llm),
            "context_recall": ContextRecall(llm=llm)}


async def ragas_one(metrics: dict, name: str, test: dict, run: dict):
    q, a, ref = test["question"], run["answer"], test["reference"]
    ctx = [readable_numbers(c) for c in run["contexts"]]
    if not a or (name != "answer_relevancy" and not ctx):
        return None
    m = metrics[name]
    if name == "faithfulness":
        r = await m.ascore(user_input=q, response=a, retrieved_contexts=ctx)
    elif name == "answer_relevancy":
        r = await m.ascore(user_input=q, response=a)
    elif name == "context_precision":
        r = await m.ascore(user_input=q, reference=ref, retrieved_contexts=ctx)
    else:
        r = await m.ascore(user_input=q, retrieved_contexts=ctx, reference=ref)
    return float(r.value)


def fingerprint(test: dict, run: dict) -> str:
    """Scores are cached per question AND per answer, so a re-run answer is re-judged."""
    blob = json.dumps([test["question"], test["reference"], run["answer"], run["contexts"]])
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


async def run_ragas(tests: dict, runs: list[dict], smoke: bool) -> tuple[pd.DataFrame, dict]:
    cache = {}
    if RAGAS_CACHE.exists():
        for r in map(json.loads, RAGAS_CACHE.read_text().splitlines()):
            cache[(r["system"], r["id"], r["metric"], r.get("key"))] = r["value"]
    keys = {(r["system"], r["id"]): fingerprint(tests[r["id"]], r) for r in runs}
    metrics, retry_metrics = ragas_metrics(0.0), ragas_metrics(0.3)
    todo = [(run, name) for run in runs if tests[run["id"]]["category"] != "unanswerable"
            for name in RAGAS_METRICS
            if (run["system"], run["id"], name, keys[(run["system"], run["id"])]) not in cache]
    if smoke:
        todo = todo[:2 * len(RAGAS_METRICS)]
        for run in {r["id"]: r for r, _ in todo}.values():   # show what the judge sees
            print(f"\n{run['id']}: {tests[run['id']]['question']}\nanswer: {run['answer']}")
            for c in run["contexts"][:2]:
                print("context:", readable_numbers(c)[:400].replace("\n", " / "))
        print()
    print(f"RAGAS: {len(todo)} metric evaluations to run")
    failed = {}
    with RAGAS_CACHE.open("a") as f:
        for k, (run, name) in enumerate(todo, 1):
            key = keys[(run["system"], run["id"])]
            value, error = None, None
            for ms in (metrics, retry_metrics):     # retry once with a little randomness
                try:
                    value, error = await ragas_one(ms, name, tests[run["id"]], run), None
                    break
                except Exception as e:
                    error = f"{type(e).__name__}: {str(e)[:120]}"
            if error:
                failed[(run["system"], name)] = failed.get((run["system"], name), 0) + 1
                print(f"  {run['system']} {run['id']} {name}: failed twice ({error})")
                continue
            cache[(run["system"], run["id"], name, key)] = value
            f.write(json.dumps({"system": run["system"], "id": run["id"], "metric": name,
                                "key": key, "value": value}) + "\n")
            f.flush()
            if k % 10 == 0 or smoke:
                print(f"  [{k}/{len(todo)}] {run['system']} {run['id']} {name} = {value}")
    rows = [{"system": s, "id": i, "metric": m, "value": v} for (s, i, m, key), v in cache.items()
            if keys.get((s, i)) == key]                # only scores for the current answers
    return pd.DataFrame(rows, columns=["system", "id", "metric", "value"]), failed


# ---------------------------------------------------------------------------- report
def fmt(x) -> str:
    return "" if pd.isna(x) else f"{x:.2f}"


def report(exact: pd.DataFrame, ragas: pd.DataFrame | None) -> str:
    ans = exact[exact["category"] != "unanswerable"]
    cols = {"route_ok": "Route", "correct": "Correct", "values": "Values", "entities": "Entities",
            "false_refusal": "Wrong refusals", "cited": "Cited", "source_hit": "Source passage retrieved"}
    by = ans.groupby(["category", "system"])[[c for c in cols if c in ans]].mean().rename(columns=cols)
    overall = ans.groupby("system")[[c for c in cols if c in ans]].mean().rename(columns=cols)
    extra = exact.groupby("system").agg(seconds=("seconds", "mean"), citations=("citations", "sum"),
                                        flagged=("flagged", "sum"))
    extra["Invalid citations"] = extra["flagged"] / (extra["citations"] + extra["flagged"]).clip(lower=1)
    refusal = exact[exact["category"] == "unanswerable"].groupby("system")["refusal_ok"].mean()
    overall["Correct refusals"] = refusal
    overall["Invalid citations"] = extra["Invalid citations"]
    overall["Mean seconds"] = extra["seconds"]
    if ragas is not None and len(ragas):
        rg = ragas.merge(exact[["id", "system", "category"]], on=["id", "system"])
        overall = overall.join(rg.pivot_table(index="system", columns="metric", values="value", aggfunc="mean"))
        by = by.join(rg.pivot_table(index=["category", "system"], columns="metric", values="value", aggfunc="mean"))
    n = exact.groupby("category")["id"].nunique()
    out = [f"Questions: {int(n.sum())} ({', '.join(f'{c} {k}' for c, k in n.items())})", "",
           "**Overall (answerable questions; refusals on unanswerable ones)**", "",
           overall.T.to_markdown(floatfmt=".2f"), "", "**By category**", "",
           by.reset_index().to_markdown(index=False, floatfmt=".2f")]
    return "\n".join(out)


def main(use_ragas: bool, smoke: bool):
    tests = {t["id"]: t for t in map(json.loads, (EVAL / "testset.jsonl").read_text().splitlines())}
    runs = [r for r in map(json.loads, (RESULTS / "runs.jsonl").read_text().splitlines()) if r["id"] in tests]
    exact = pd.DataFrame([exact_scores(tests[r["id"]], r) for r in runs])
    exact.to_csv(RESULTS / "exact_scores.csv", index=False)
    ragas, failed = asyncio.run(run_ragas(tests, runs, smoke)) if use_ragas else (None, {})
    if smoke:
        return
    text = report(exact, ragas)
    if ragas is not None:
        n = ragas.dropna(subset=["value"]).groupby(["system", "metric"]).size()
        text += "\n\nRAGAS evaluations completed per system and metric (failed after retry in brackets): " + \
            "; ".join(f"{s} {m} {k} ({failed.get((s, m), 0)})" for (s, m), k in n.items())
    (RESULTS / "summary.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ragas", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    main(a.ragas, a.smoke)
