"""Question -> Cypher over the Neo4j knowledge graph -> rows with evidence.

Safeguards: write clauses are rejected before execution, the query runs in a read-only
transaction, a row limit is added when missing, and a failed or empty query is retried
with feedback. Exact names (companies, organisations, risk topics, segments) are read
from the graph and put in the prompt, because small models misspell names otherwise.

    python tools/cypher_tool.py "Which companies name AMD as a competitor?"
"""
import re
import sys
from functools import lru_cache
from pathlib import Path

from neo4j import READ_ACCESS

sys.path.append(str(Path(__file__).resolve().parents[1]))
from core.neo4j_db import driver  # noqa: E402
from core.ollama import chat_json  # noqa: E402
from tools.base import ToolResult  # noqa: E402

MAX_ROWS = 100
RETRIES = 2
WRITE_CLAUSES = re.compile(r"\b(create|merge|delete|detach|set|remove|drop|load\s+csv|"
                           r"foreach|call\s+db\.|call\s+apoc\.(create|merge|refactor|periodic))\b",
                           re.I)
CHUNK_ID = re.compile(r"^[A-Z]{2,5}_\d{4}_[0-9A-Z]+_\d{4}$")

SCHEMA = """Neo4j knowledge graph built from 10-K annual reports of 8 companies.

Nodes
  (:Company:Organization {ticker, name, industry})  the 8 companies; ticker is one of
      AAPL, MSFT, GOOGL, AMZN, META, NVDA, INTC, AMD
  (:Organization {name})        other companies (competitors, suppliers, partners)
  (:Report {id, ticker, fiscal_year, filed, url})
  (:Subsidiary {name}), (:Jurisdiction {name, country})
  (:Product {name, category}), (:Segment {id, name})
  (:RiskTopic {name, description})

Relationships
  (Company)-[:FILED]->(Report)
  (Company)-[:HAS_SUBSIDIARY {fiscal_years}]->(Subsidiary)-[:INCORPORATED_IN]->(Jurisdiction)
  (Company)-[:COMPETES_WITH {fiscal_year, mentions, evidence}]->(Organization)
  (Company)-[:OFFERS {fiscal_year, mentions, evidence}]->(Product)
  (Company)-[:HAS_SEGMENT {fiscal_year, mentions, evidence}]->(Segment)
  (Company)-[:HAS_SUPPLIER {fiscal_year, role, mentions, evidence}]->(Organization)
      role is "supplier" or "manufacturer"
  (Company)-[:HAS_CUSTOMER {fiscal_year, mentions, evidence}]->(Organization)
  (Company)-[:HAS_PARTNER {fiscal_year, mentions, evidence}]->(Organization)
  (Company)-[:CITES_RISK {fiscal_year, passages, share, rank, evidence}]->(RiskTopic)
      share = fraction of the company's risk passages on this topic; rank 1 = biggest

Notes: relationship evidence is a list of passage ids; return it as `evidence` whenever
the answer uses COMPETES_WITH, OFFERS, HAS_SEGMENT, HAS_SUPPLIER, HAS_CUSTOMER,
HAS_PARTNER or CITES_RISK. A competitor that is one of the 8 companies is the same node
as that Company. Match products with toLower(p.name) CONTAINS or toLower(p.category)
CONTAINS, because product names vary. Jurisdiction.country groups US states into
"United States"."""

EXAMPLES = """Examples:
Q: Which companies name AMD as a competitor?
Cypher: MATCH (c:Company)-[r:COMPETES_WITH]->(o:Organization {name: 'AMD'}) RETURN c.name AS company, r.mentions AS mentions, r.evidence AS evidence ORDER BY mentions DESC
Q: In which countries does Microsoft have subsidiaries?
Cypher: MATCH (:Company {ticker: 'MSFT'})-[:HAS_SUBSIDIARY]->(s:Subsidiary)-[:INCORPORATED_IN]->(j:Jurisdiction) RETURN j.country AS country, count(DISTINCT s) AS subsidiaries ORDER BY subsidiaries DESC
Q: Which companies emphasise export controls most in their risk factors?
Cypher: MATCH (c:Company)-[r:CITES_RISK]->(t:RiskTopic {name: 'Export controls and trade restrictions'}) RETURN c.name AS company, r.rank AS rank, r.share AS share, r.evidence AS evidence ORDER BY r.share DESC
Q: Which companies rely on TSMC?
Cypher: MATCH (c:Company)-[r:HAS_SUPPLIER|HAS_PARTNER]->(o:Organization {name: 'TSMC'}) RETURN c.name AS company, type(r) AS relationship, r.role AS role, r.evidence AS evidence
Q: What wearable products does Apple offer?
Cypher: MATCH (:Company {ticker: 'AAPL'})-[r:OFFERS]->(p:Product) WHERE toLower(p.category) CONTAINS 'watch' OR toLower(p.category) CONTAINS 'wear' OR toLower(p.category) CONTAINS 'headphone' RETURN p.name AS product, p.category AS category, r.evidence AS evidence
Q: Which companies have subsidiaries incorporated in Japan?
Cypher: MATCH (c:Company)-[:HAS_SUBSIDIARY]->(s:Subsidiary)-[:INCORPORATED_IN]->(j:Jurisdiction {country: 'Japan'}) RETURN c.name AS company, count(DISTINCT s) AS subsidiaries ORDER BY subsidiaries DESC
Q: Which pairs of companies share a manufacturer?
Cypher: MATCH (a:Company)-[:HAS_SUPPLIER {role: 'manufacturer'}]->(o:Organization)<-[:HAS_SUPPLIER {role: 'manufacturer'}]-(b:Company) WHERE a.ticker < b.ticker RETURN a.name AS company_a, b.name AS company_b, o.name AS manufacturer"""

OUTPUT = {"type": "object",
          "properties": {"cypher": {"type": "string"}},
          "required": ["cypher"]}


@lru_cache(maxsize=1)
def known_names() -> str:
    """Exact names from the graph, so generated queries match existing nodes."""
    with driver() as db:
        def names(q):
            return [r["n"] for r in db.execute_query(q, database_="neo4j").records]
        orgs = names("MATCH (o:Organization) RETURN o.name AS n ORDER BY n")
        risks = names("MATCH (t:RiskTopic) RETURN t.name AS n ORDER BY n")
        segs = names("MATCH (c:Company)-[:HAS_SEGMENT]->(s) RETURN c.ticker + ': ' + s.name AS n ORDER BY n")
    return (f"Organization names: {', '.join(orgs)}\n"
            f"RiskTopic names: {'; '.join(risks)}\n"
            f"Segments (ticker: name): {'; '.join(segs)}")


def system_prompt() -> str:
    return (f"You write one read-only Cypher query that answers the user's question.\n"
            f"{SCHEMA}\n\n{known_names()}\n\n"
            "Rules: MATCH/OPTIONAL MATCH/WHERE/WITH/RETURN/ORDER BY/LIMIT only; never write "
            "data; use exact names from the lists above; give returned columns clear aliases; "
            "whenever you return relationship properties (mentions, evidence, share, rank), "
            "name the relationship in the pattern, e.g. -[r:COMPETES_WITH]->. "
            "If the graph cannot answer the question, return an empty string as cypher.\n\n"
            f"{EXAMPLES}")


def repair_cypher(q: str, error: str) -> str | None:
    """Fix the most common small-model mistake: returning r.x from an unnamed relationship."""
    m = re.search(r"Variable `(\w+)` not defined", error)
    if m and len(re.findall(r"-\[:", q)) == 1:
        return q.replace("-[:", f"-[{m.group(1)}:", 1)
    return None


def check_cypher(q: str) -> str:
    q = q.strip().rstrip(";").strip()
    if not q:
        raise ValueError("no query generated: the knowledge graph may not cover this question")
    if ";" in q:
        raise ValueError("only one statement is allowed")
    if WRITE_CLAUSES.search(q):
        raise ValueError("only read-only queries are allowed")
    if not re.search(r"\blimit\s+\d+\s*$", q, re.I):
        q += f" LIMIT {MAX_ROWS}"
    return q


def run_cypher(q: str) -> tuple[list, list]:
    with driver() as db, db.session(database="neo4j", default_access_mode=READ_ACCESS) as s:
        def work(tx):
            result = tx.run(q)
            return list(result.keys()), [list(r.values()) for r in result]
        return s.execute_read(work)


def collect_evidence(rows: list) -> list:
    ids = []
    for row in rows:
        for v in row:
            for x in (v if isinstance(v, list) else [v]):
                if isinstance(x, str) and CHUNK_ID.match(x) and x not in ids:
                    ids.append(x)
    return ids


def answer(question: str) -> ToolResult:
    res = ToolResult(tool="cypher", question=question)
    system, user = system_prompt(), f"Question: {question}"
    for attempt in range(1, RETRIES + 2):
        res.attempts = attempt
        q = chat_json(system, user, OUTPUT, max_tokens=600).get("cypher", "")
        try:
            res.query = check_cypher(q)
            res.columns, res.rows = run_cypher(res.query)
            res.error = ""
            if res.rows:
                res.evidence = collect_evidence(res.rows)
                return res
            feedback = ("returned no rows. Check that names match the lists exactly and that "
                        "the relationship direction and type are right")
        except Exception as e:                       # syntax errors, rejected writes, ...
            res.query, res.error = q, str(e).split("\n")[0][:300]
            if "no query" in res.error:
                return res
            fixed = repair_cypher(q, str(e))
            if fixed:
                try:
                    res.query = check_cypher(fixed)
                    res.columns, res.rows = run_cypher(res.query)
                    res.error = ""
                    res.evidence = collect_evidence(res.rows)
                    if res.rows:
                        return res
                except Exception:
                    pass
            feedback = f"failed with: {res.error}"
        user = (f"Question: {question}\nYour previous query:\n{q}\n{feedback}.\n"
                "Write a corrected query.")
    return res


if __name__ == "__main__":
    r = answer(" ".join(sys.argv[1:]))
    print(r.as_context())
    if r.evidence:
        print("evidence:", ", ".join(r.evidence[:10]))
