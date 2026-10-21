"""Load the structured part of the knowledge graph into Neo4j.

Schema
  (:Company {ticker, name, cik, industry, fiscal_year_end, state_of_incorporation})
  (:Report {id, ticker, fiscal_year, filed, url})
  (:Subsidiary {name})
  (:Jurisdiction {name, country})
  (:RiskTopic {name, description})

  (Company)-[:FILED]->(Report)
  (Company)-[:HAS_SUBSIDIARY {fiscal_years}]->(Subsidiary)-[:INCORPORATED_IN]->(Jurisdiction)
  (Company)-[:CITES_RISK {fiscal_year, passages, evidence}]->(RiskTopic)

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
]


def run(db, query: str, rows: list | None = None, **params):
    db.execute_query(query, rows=rows or [], **params, database_="neo4j")


def main(reset: bool):
    companies = list(json.loads((RAW / "companies.json").read_text()).values())
    manifest = json.loads((RAW / "manifest.json").read_text())
    subs = pd.read_csv(GRAPH / "subsidiaries.csv").to_dict("records")
    risks = pd.read_csv(GRAPH / "risk_topics.csv").to_dict("records")

    with driver() as db:
        db.verify_connectivity()
        if reset:
            run(db, "MATCH (n) DETACH DELETE n")
        for c in CONSTRAINTS:
            run(db, c)

        run(db, """UNWIND $rows AS r
                   MERGE (c:Company {ticker: r.ticker})
                   SET c.name = r.name, c.cik = r.cik, c.industry = r.industry,
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
                   SET x.passages = r.passages, x.evidence = split(r.evidence, ';')""", risks)

        counts, _, _ = db.execute_query(
            """MATCH (n) WITH labels(n)[0] AS label, count(*) AS n
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
