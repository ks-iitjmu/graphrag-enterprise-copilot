"""Tag each company's risk-factor passages with a fixed set of risk topics.

No LLM calls: every topic has a short description, embedded once. Each risk passage
(Item 1A, or Intel's whole report), using its vector already stored in the FAISS index,
is assigned to its most similar topic when the similarity clears a threshold. A company "cites" a topic in a year when at least
MIN_PASSAGES passages are assigned to it, which filters out one-off mentions.

Output: data/graph/risk_topics.csv (ticker, fiscal_year, topic, passages, evidence chunk ids)
Usage:  python ingest/classify_risks.py
"""
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import DATA, INDEX  # noqa: E402
from core.ollama import QUERY_PREFIX, embed  # noqa: E402

GRAPH = DATA / "graph"
MIN_SIMILARITY = 0.55
MIN_PASSAGES = 2

TOPICS = {
    "Export controls and trade restrictions":
        "export controls, trade restrictions, tariffs, sanctions, licences required to sell chips or technology to China and other countries",
    "Supply chain and manufacturing concentration":
        "dependence on a limited number of suppliers, contract manufacturers or foundries, concentration of manufacturing in Asia or Taiwan, component shortages",
    "Geopolitical tensions":
        "geopolitical tensions, conflicts, Taiwan, China relations, war, political instability affecting operations",
    "Cybersecurity and data breaches":
        "cyberattacks, security breaches, ransomware, unauthorised access to systems or customer data",
    "Data privacy and protection regulation":
        "privacy laws such as GDPR, data protection regulation, restrictions on collecting, using or transferring personal data",
    "AI development and regulation":
        "risks from developing and deploying artificial intelligence, AI regulation, responsible AI, harmful or inaccurate AI outputs",
    "Antitrust and competition law":
        "antitrust investigations, competition law, Digital Markets Act, regulators challenging market dominance or acquisitions",
    "Intense competition":
        "intense competition, competitors with greater resources, new entrants, pricing pressure, losing market share",
    "Customer concentration":
        "a small number of customers account for a large share of revenue, dependence on major customers or partners",
    "Litigation and intellectual property":
        "lawsuits, patent infringement claims, intellectual property disputes, legal proceedings",
    "Talent and key personnel":
        "ability to attract and retain key employees, engineers and executives, competition for talent",
    "Foreign exchange and macroeconomic conditions":
        "currency exchange rate fluctuations, inflation, interest rates, recession, macroeconomic conditions affecting demand",
    "Tax law changes":
        "changes in tax laws, tax disputes, global minimum tax, effective tax rate",
    "Climate and sustainability":
        "climate change, extreme weather, environmental regulation, sustainability and emissions commitments",
    "Capital intensity and large investments":
        "large capital expenditures, investments in data centers or fabs that may not generate returns, infrastructure spending",
}


def main():
    chunks = pd.read_parquet(INDEX / "chunks.parquet")
    index = faiss.read_index(str(INDEX / "faiss.index"))
    mask = chunks["item"].isin(["1A", "ALL"]).values
    passage_vecs = index.reconstruct_n(0, index.ntotal)[mask]   # reuse stored embeddings
    risk = chunks[mask].reset_index(drop=True)
    print(f"{len(risk):,} risk passages to classify")

    names = list(TOPICS)
    topic_vecs = embed([TOPICS[n] for n in names], prefix=QUERY_PREFIX)
    sims = passage_vecs @ topic_vecs.T
    best, score = sims.argmax(1), sims.max(1)
    risk["topic"] = [names[i] for i in best]
    risk["similarity"] = score
    kept = risk[risk["similarity"] >= MIN_SIMILARITY]
    print(f"{len(kept):,} passages assigned a topic "
          f"(similarity >= {MIN_SIMILARITY}); median similarity {np.median(score):.2f}")

    agg = (kept.sort_values("similarity", ascending=False)
           .groupby(["ticker", "fiscal_year", "topic"])
           .agg(passages=("chunk_id", "count"),
                evidence=("chunk_id", lambda x: ";".join(x.head(3))))
           .reset_index())
    agg = agg[agg["passages"] >= MIN_PASSAGES]
    GRAPH.mkdir(parents=True, exist_ok=True)
    agg.to_csv(GRAPH / "risk_topics.csv", index=False)
    print(f"Wrote {GRAPH / 'risk_topics.csv'}: {len(agg)} company-year-topic links\n")
    latest = agg[agg.groupby("ticker")["fiscal_year"].transform("max") == agg["fiscal_year"]]
    print(latest.pivot_table(index="topic", columns="ticker", values="passages",
                             fill_value=0).astype(int).to_string())


if __name__ == "__main__":
    main()
