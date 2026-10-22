"""Extract business relationships and risk topics from 10-K passages with the local LLM.

For each company's latest report, every passage from Item 1 (Business) and Item 1A
(Risk Factors), or Intel's whole report, goes to the LLM once. It returns JSON with:
  competitors, products, segments, partners (with role), risk_topics (from a fixed list)

Results are cached in data/graph/llm_cache.jsonl, so an interrupted run resumes where
it stopped. Names are then normalised ("Advanced Micro Devices, Inc." -> "AMD") and
aggregated per company with evidence chunk ids.

Output (data/graph/): competitors.csv, products.csv, segments.csv, partners.csv,
                      risk_topics.csv (replaces the embedding-based version)
Usage:  python ingest/extract_entities.py                    full run
        python ingest/extract_entities.py --limit 40         trial on a spread-out sample
        python ingest/extract_entities.py --limit 30 --grep compet   trial on matching passages
"""
import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import COMPANIES, DATA, INDEX  # noqa: E402
from core.ollama import chat_json  # noqa: E402
from ingest.classify_risks import TOPICS  # noqa: E402

GRAPH = DATA / "graph"
CACHE = GRAPH / "llm_cache.jsonl"
ROLES = ["supplier", "manufacturer", "customer", "partner"]

SCHEMA = {
    "type": "object",
    "properties": {
        "competitors": {"type": "array", "items": {"type": "string"}},
        "products": {"type": "array", "items": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "category": {"type": "string"}},
            "required": ["name", "category"]}},
        "segments": {"type": "array", "items": {"type": "string"}},
        "partners": {"type": "array", "items": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "role": {"type": "string", "enum": ROLES}},
            "required": ["name", "role"]}},
        "risk_topics": {"type": "array", "items": {"type": "string", "enum": list(TOPICS)}},
    },
    "required": ["competitors", "products", "segments", "partners", "risk_topics"],
}

SYSTEM = f"""You extract facts from one passage of a company's annual report (Form 10-K).
Return JSON with these fields, using ONLY what the passage explicitly states:
- competitors: names of specific companies the passage says {{company}} competes with.
  Only named organisations, never generic groups such as "large technology companies".
- products: named products, services or platforms that {{company}} itself offers, each
  with a short category (for example "smartphone", "cloud platform", "GPU").
- segments: names of {{company}}'s reportable business segments, if the passage names them.
- partners: named organisations {{company}} relies on or sells to, with role
  supplier, manufacturer, customer or partner.
- risk_topics: the risk topics the passage substantially discusses, chosen only from:
  {"; ".join(TOPICS)}. Use an empty list if the passage is not about risks.
Use empty lists when nothing applies. Do not guess."""

# Canonical names: our eight companies plus frequent outside organisations
ALIASES = {
    "apple": "Apple", "microsoft": "Microsoft", "alphabet": "Alphabet", "google": "Alphabet",
    "amazon": "Amazon", "amazon.com": "Amazon", "amazon web services": "Amazon", "aws": "Amazon",
    "meta": "Meta Platforms", "meta platforms": "Meta Platforms", "facebook": "Meta Platforms",
    "nvidia": "NVIDIA", "intel": "Intel", "amd": "AMD", "advanced micro devices": "AMD",
    "taiwan semiconductor manufacturing": "TSMC", "tsmc": "TSMC",
    "samsung electronics": "Samsung", "samsung": "Samsung", "qualcomm": "Qualcomm",
    "broadcom": "Broadcom", "huawei": "Huawei", "huawei technologies": "Huawei",
    "oracle": "Oracle", "ibm": "IBM", "international business machines": "IBM",
    "bytedance": "ByteDance", "tiktok": "ByteDance", "openai": "OpenAI",
    "arm": "Arm", "arm holdings": "Arm", "foxconn": "Foxconn", "hon hai": "Foxconn",
    "hon hai precision industry": "Foxconn", "sk hynix": "SK hynix", "micron": "Micron",
    "micron technology": "Micron", "salesforce": "Salesforce", "alibaba": "Alibaba",
    # platforms named as competitors -> the company behind them
    "android": "Alphabet", "chrome": "Alphabet", "youtube": "Alphabet",
    "windows": "Microsoft", "xbox": "Microsoft", "azure": "Microsoft",
    "linkedin": "Microsoft", "playstation": "Sony", "sony": "Sony",
    "instagram": "Meta Platforms", "whatsapp": "Meta Platforms",
}
SUFFIX = re.compile(r"[,.]?\s+(inc|incorporated|corp|corporation|co|company|ltd|limited|llc|"
                    r"plc|n\.v|s\.a|ag|gmbh|holdings|group)\.?$", re.I)
GENERIC = re.compile(r"\b(compan(y|ies)|providers?|competitors?|firms?|others?|vendors?|"
                     r"manufacturers?|suppliers?|customers?|partners?|startups?|businesses|"
                     r"players|entities|platforms|governments?)\b", re.I)


TRADEMARKS = re.compile(r"[®™©]|\((R|TM|C)\)", re.I)


def canonical_org(name: str) -> str | None:
    name = TRADEMARKS.sub("", name)
    name = re.sub(r"\s+", " ", name).strip(" ,.;'\"")
    for _ in range(2):
        name = SUFFIX.sub("", name).strip(" ,.")
    if name.lower() in ALIASES:                    # known names first ("Meta Platforms")
        return ALIASES[name.lower()]
    if len(name) < 2 or GENERIC.search(name) or not (name[0].isupper() or name[0].isdigit()):
        return None
    return name


def canonical_label(text: str) -> str:
    return re.sub(r"\s+", " ", TRADEMARKS.sub("", text)).strip(" ,.;'\"*")


def candidates(chunks: pd.DataFrame) -> pd.DataFrame:
    latest = chunks.groupby("ticker")["fiscal_year"].transform("max") == chunks["fiscal_year"]
    return chunks[latest & chunks["item"].isin(["1", "1A", "ALL"])].reset_index(drop=True)


def load_cache() -> dict:
    if not CACHE.exists():
        return {}
    return {r["chunk_id"]: r["result"] for r in map(json.loads, CACHE.read_text().splitlines())}


def run_llm(todo: pd.DataFrame, cache: dict):
    GRAPH.mkdir(parents=True, exist_ok=True)
    t0, done = time.time(), 0
    with CACHE.open("a") as f:
        for row in todo.itertuples():
            if row.chunk_id in cache:
                continue
            system = SYSTEM.replace("{company}", row.company)
            user = (f"Company: {row.company}\nReport: 10-K, fiscal year {row.fiscal_year}, "
                    f"{row.section}\n\nPassage:\n{row.text}")
            try:
                result = chat_json(system, user, SCHEMA, max_tokens=1500)
            except Exception as e:                      # keep going; failures are retried next run
                print(f"  {row.chunk_id}: failed ({e})")
                continue
            cache[row.chunk_id] = result
            f.write(json.dumps({"chunk_id": row.chunk_id, "result": result}) + "\n")
            f.flush()
            done += 1
            if done % 25 == 0:
                rate = (time.time() - t0) / done
                left = sum(1 for c in todo["chunk_id"] if c not in cache) * rate / 60
                print(f"  {done} passages, {rate:.1f}s each, about {left:.0f} min left")


def aggregate(todo: pd.DataFrame, cache: dict):
    comp, prod, seg, part, risk, categories = (defaultdict(list) for _ in range(6))
    names = {c: n for c, n in COMPANIES.items()}
    for row in todo.itertuples():
        r = cache.get(row.chunk_id)
        if not r:
            continue
        key = (row.ticker, row.fiscal_year)
        own = canonical_org(names[row.ticker])
        for c in r.get("competitors", []):
            org = canonical_org(c)
            if org and org != own:
                comp[key + (org,)].append(row.chunk_id)
        for p in r.get("products", []):
            name = canonical_label(p.get("name", ""))
            if 2 <= len(name) <= 60:
                prod[key + (name,)].append(row.chunk_id)
                categories[key + (name,)].append(canonical_label(p.get("category", "")).lower())
        for s in r.get("segments", []):
            name = canonical_label(s)
            if 2 <= len(name) <= 60:
                seg[key + (name,)].append(row.chunk_id)
        for p in r.get("partners", []):
            org = canonical_org(p.get("name", ""))
            if org and org != own and p.get("role") in ROLES:
                part[key + (org, p["role"])].append(row.chunk_id)
        for t in r.get("risk_topics", []):
            if t in TOPICS:
                risk[key + (t,)].append(row.chunk_id)

    def frame(d, cols, min_mentions=1):
        rows = []
        for k, v in d.items():
            passages = list(dict.fromkeys(v))          # count passages, not repeats within one
            if len(passages) >= min_mentions:
                rows.append(dict(zip(cols, k), mentions=len(passages),
                                 evidence=";".join(passages)[:2000]))
        return pd.DataFrame(rows, columns=cols + ["mentions", "evidence"])

    base = ["ticker", "fiscal_year"]
    out = {"competitors": frame(comp, base + ["competitor"]),
           "products": frame(prod, base + ["product"]),
           "segments": frame(seg, base + ["segment"], min_mentions=2),
           "partners": frame(part, base + ["organization", "role"]),
           "risk_topics": frame(risk, base + ["topic"], min_mentions=2)}
    # Emphasis, not just presence: 10-K risk sections mention nearly every topic, so record
    # each topic's share of the company's risk passages and its rank within the company.
    rt = out["risk_topics"].rename(columns={"mentions": "passages"})
    risk_passages = defaultdict(set)
    for (ticker, year, _topic), ids in risk.items():
        risk_passages[(ticker, year)].update(ids)
    rt["share"] = [round(p / max(len(risk_passages[(t, y)]), 1), 3)
                   for t, y, p in zip(rt["ticker"], rt["fiscal_year"], rt["passages"])]
    rt["rank"] = (rt.groupby(["ticker", "fiscal_year"])["passages"]
                  .rank(ascending=False, method="min").astype(int))
    out["risk_topics"] = rt[["ticker", "fiscal_year", "topic", "passages", "share", "rank",
                             "evidence"]]
    out["products"].insert(3, "category", [                   # most common category per product
        pd.Series(categories[(t, y, n)]).mode().iloc[0] if categories[(t, y, n)] else ""
        for t, y, n in zip(out["products"]["ticker"], out["products"]["fiscal_year"],
                           out["products"]["product"])])
    for name, df in out.items():
        df.to_csv(GRAPH / f"{name}.csv", index=False)
        print(f"{name:12s} {len(df):5d} rows")
    return out


def main(limit: int | None, grep: str | None):
    chunks = pd.read_parquet(INDEX / "chunks.parquet")
    todo = candidates(chunks)
    if grep:
        todo = todo[todo["text"].str.contains(grep, case=False)]
    if limit:                                     # spread the trial across each report
        per = max(1, limit // todo["ticker"].nunique())
        todo = (todo.sample(frac=1, random_state=0).groupby("ticker").head(per)
                .sort_index().reset_index(drop=True))
    cache = load_cache()
    remaining = sum(1 for c in todo["chunk_id"] if c not in cache)
    print(f"{len(todo):,} passages from the latest reports; {remaining:,} not yet processed")
    run_llm(todo, cache)
    out = aggregate(todo, cache)
    top = (out["competitors"].sort_values("mentions", ascending=False)
           .groupby("ticker").head(5).groupby("ticker")["competitor"].apply(", ".join))
    print("\nMost-mentioned competitors per company:\n" + top.to_string())
    rt = out["risk_topics"]
    top_risks = (rt[rt["rank"] <= 3].sort_values(["ticker", "rank"])
                 .groupby("ticker")["topic"].apply(" | ".join))
    print("\nTop three risk topics per company (by share of risk passages):")
    for ticker, topics in top_risks.items():
        print(f"  {ticker:5s} {topics}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="process only about N passages (trial run)")
    ap.add_argument("--grep", help="only passages containing this text (trial runs)")
    a = ap.parse_args()
    main(a.limit, a.grep)
