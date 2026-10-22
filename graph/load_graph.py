"""Load the knowledge graph into Neo4j.

Schema
  (:Company:Organization {ticker, name, cik, industry, fiscal_year_end, state_of_incorporation})
  (:Organization {name})          companies outside the eight, e.g. TSMC, Samsung, Huawei
  (:Product {name, category})
  (:Segment {id, name})
  (:Report {id, ticker, fiscal_year, filed, url})
  (:Subsidiary {name})
  (:Jurisdiction {name, country})
  (:RiskTopic {name, description})

  (Company)-[:FILED]->(Report)
  (Company)-[:HAS_SUBSIDIARY {fiscal_years}]->(Subsidiary)-[:INCORPORATED_IN]->(Jurisdiction)
  (Company)-[:CITES_RISK {fiscal_year, passages, share, rank, evidence}]->(RiskTopic)
  (Company)-[:COMPETES_WITH {fiscal_year, mentions, evidence}]->(Organization)
  (Company)-[:OFFERS {fiscal_year, mentions, evidence}]->(Product)
  (Company)-[:HAS_SEGMENT {fiscal_year, mentions, evidence}]->(Segment)
  (Company)-[:HAS_SUPPLIER {role} | :HAS_CUSTOMER | :HAS_PARTNER {fiscal_year, ...}]->(Organization)

Facts from Steps 3a (subsidiaries) and 3b (LLM extraction: competitors, products,
segments, partners, risk topics). Every LLM-derived relationship keeps the chunk ids
of the passages it came from, so answers can cite their source.

Re-running is safe: everything is MERGEd. Use --reset to wipe the database first.
Usage:  python graph/load_graph.py [--reset]
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import DATA, RAW  # noqa: E402
from core.neo4j_db import driver  # noqa: E402
from ingest.classify_risks import TOPICS  # noqa: E402

GRAPH = DATA / "graph"
CONSTRAINTS = [
    "CREATE CONSTRAINT company_ticker IF NOT EXISTS FOR (c:Company) REQUIRE c.ticker IS UNIQUE",
    "CREATE CONSTRAINT report_id IF NOT EXISTS FOR (r:Report) REQUIRE r.id IS UNIQUE",
    "CREATE CONSTRAINT subsidiary_name IF NOT EXISTS FOR (s:Subsidiary) REQUIRE s.name IS UNIQUE",
    "CREATE CONSTRAINT jurisdiction_name IF NOT EXISTS FOR (j:Jurisdiction) REQUIRE j.name IS UNIQUE",
    "CREATE CONSTRAINT risk_topic_name IF NOT EXISTS FOR (t:RiskTopic) REQUIRE t.name IS UNIQUE",
    "CREATE CONSTRAINT organization_name IF NOT EXISTS FOR (o:Organization) REQUIRE o.name IS UNIQUE",
    "CREATE CONSTRAINT product_name IF NOT EXISTS FOR (p:Product) REQUIRE p.name IS UNIQUE",
    "CREATE CONSTRAINT segment_id IF NOT EXISTS FOR (s:Segment) REQUIRE s.id IS UNIQUE",
]
PARTNER_RELS = {"supplier": "HAS_SUPPLIER", "manufacturer": "HAS_SUPPLIER",
                "customer": "HAS_CUSTOMER", "partner": "HAS_PARTNER"}


def records(name: str) -> list[dict]:
    path = GRAPH / f"{name}.csv"
    if not path.exists():
        print(f"  ({name}.csv not found, skipped)")
        return []
    df = pd.read_csv(path).fillna("")
    if "evidence" in df:
        df["evidence"] = df["evidence"].astype(str)
    return df.to_dict("records")


def run(db, query: str, rows: list | None = None, **params):
    db.execute_query(query, rows=rows or [], **params, database_="neo4j")


def main(reset: bool):
    companies = list(json.loads((RAW / "companies.json").read_text()).values())
    manifest = json.loads((RAW / "manifest.json").read_text())
    subs = records("subsidiaries")
    risks = records("risk_topics")

    with driver() as db:
        db.verify_connectivity()
        if reset:
            run(db, "MATCH (n) DETACH DELETE n")
        for c in CONSTRAINTS:
            run(db, c)

        run(db, """UNWIND $rows AS r
                   MERGE (c:Company {ticker: r.ticker})
                   SET c:Organization, c.name = r.name, c.cik = r.cik, c.industry = r.industry,
                       c.fiscal_year_end = r.fiscal_year_end,
                       c.state_of_incorporation = r.state_of_incorporation""", companies)

        run(db, """UNWIND $rows AS r
                   MATCH (c:Company {ticker: r.ticker})
                   MERGE (rep:Report {id: r.ticker + '_' + toString(r.fiscal_year)})
                   SET rep.ticker = r.ticker, rep.fiscal_year = r.fiscal_year,
                       rep.filed = r.filed, rep.url = r.url
                   MERGE (c)-[:FILED]->(rep)""", manifest)

        for i in range(0, len(subs), 1000):
            run(db, """UNWIND $rows AS r
                       MATCH (c:Company {ticker: r.ticker})
                       MERGE (s:Subsidiary {name: r.subsidiary})
                       MERGE (j:Jurisdiction {name: r.jurisdiction})
                       SET j.country = r.country
                       MERGE (s)-[:INCORPORATED_IN]->(j)
                       MERGE (c)-[h:HAS_SUBSIDIARY]->(s)
                       ON CREATE SET h.fiscal_years = [r.fiscal_year]
                       ON MATCH SET h.fiscal_years = CASE WHEN r.fiscal_year IN h.fiscal_years
                           THEN h.fiscal_years ELSE h.fiscal_years + r.fiscal_year END""",
                subs[i:i + 1000])

        run(db, """UNWIND $rows AS r
                   MERGE (t:RiskTopic {name: r.name}) SET t.description = r.description""",
            [{"name": k, "description": v} for k, v in TOPICS.items()])
        run(db, """UNWIND $rows AS r
                   MATCH (c:Company {ticker: r.ticker}), (t:RiskTopic {name: r.topic})
                   MERGE (c)-[x:CITES_RISK {fiscal_year: r.fiscal_year}]->(t)
                   SET x.passages = r.passages, x.share = r.share, x.rank = r.rank,
                       x.evidence = split(r.evidence, ';')""", risks)

        run(db, """UNWIND $rows AS r
                   MATCH (c:Company {ticker: r.ticker})
                   MERGE (o:Organization {name: r.competitor})
                   MERGE (c)-[x:COMPETES_WITH {fiscal_year: r.fiscal_year}]->(o)
                   SET x.mentions = r.mentions, x.evidence = split(r.evidence, ';')""",
            records("competitors"))
        run(db, """UNWIND $rows AS r
                   MATCH (c:Company {ticker: r.ticker})
                   MERGE (p:Product {name: r.product})
                   SET p.category = CASE WHEN p.category IS NULL OR p.category = ''
                                         THEN r.category ELSE p.category END
                   MERGE (c)-[x:OFFERS {fiscal_year: r.fiscal_year}]->(p)
                   SET x.mentions = r.mentions, x.evidence = split(r.evidence, ';')""",
            records("products"))
        run(db, """UNWIND $rows AS r
                   MATCH (c:Company {ticker: r.ticker})
                   MERGE (s:Segment {id: r.ticker + ':' + r.segment})
                   SET s.name = r.segment
                   MERGE (c)-[x:HAS_SEGMENT {fiscal_year: r.fiscal_year}]->(s)
                   SET x.mentions = r.mentions, x.evidence = split(r.evidence, ';')""",
            records("segments"))
        partners = records("partners")
        for role, rel in PARTNER_RELS.items():
            run(db, f"""UNWIND $rows AS r
                        MATCH (c:Company {{ticker: r.ticker}})
                        MERGE (o:Organization {{name: r.organization}})
                        MERGE (c)-[x:{rel} {{fiscal_year: r.fiscal_year, role: r.role}}]->(o)
                        SET x.mentions = r.mentions, x.evidence = split(r.evidence, ';')""",
                [p for p in partners if p["role"] == role])

        counts, _, _ = db.execute_query(
            """MATCH (n) WITH CASE WHEN n:Company THEN 'Company' ELSE labels(n)[0] END AS label,
                              count(*) AS n
               RETURN label, n ORDER BY label""", database_="neo4j")
        rels, _, _ = db.execute_query(
            """MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS n ORDER BY rel""",
            database_="neo4j")
    print("Nodes:        ", ", ".join(f"{r['label']} {r['n']:,}" for r in counts))
    print("Relationships:", ", ".join(f"{r['rel']} {r['n']:,}" for r in rels))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="delete all nodes first")
    main(ap.parse_args().reset)
