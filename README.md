# GraphRAG Enterprise Copilot

An agentic assistant that answers questions over company annual reports by routing each
question to the right source: vector search over the report text (FAISS), a knowledge
graph of companies, subsidiaries, competitors and products (Neo4j, queried with Cypher),
or a SQL database of reported financials. Built with LangGraph, served with FastAPI,
deployed on Azure, and evaluated with RAGAS.

## Data

SEC 10-K annual reports for Apple, Microsoft, Alphabet, Amazon, Meta, NVIDIA, Intel and
AMD (three most recent fiscal years), their Exhibit 21 subsidiary lists, and XBRL
financial facts, all downloaded from SEC EDGAR. All data is public; none is simulated.

## Status

- [x] Step 1: download filings and build the financial SQL database
- [x] Step 2: chunk and embed report text into FAISS
- [ ] Step 3: build the knowledge graph in Neo4j (3a structured facts done; 3b LLM extraction next)
- [ ] Step 4: vector, Cypher and SQL tools
- [ ] Step 5: LangGraph agent with routing and retries
- [ ] Step 6: evaluation with RAGAS against a vector-only baseline
- [ ] Step 7: FastAPI service
- [ ] Step 8: deployment on Azure

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then edit: your name and email for the SEC, a Neo4j password
set -a && source .env && set +a
docker compose up -d          # Neo4j at http://localhost:7474
```

## Step 1: data

```bash
python ingest/sec_download.py   # ~24 annual reports, subsidiary lists, XBRL facts
python ingest/build_sql.py      # data/finance.db
```

`finance.db` has a `financials` table (one row per company, fiscal year and metric, with
the XBRL concept it came from) and an `annual_financials` view with one row per company
and year: revenue, cost of revenue, gross profit, R&D, operating income, net income,
diluted EPS, operating cash flow, capital expenditure, share repurchases, total assets,
total liabilities, cash and stockholders' equity.

Companies tag the same item with different XBRL concepts, so each metric lists candidate
concepts in priority order. Values restated in later filings keep the latest figure, and
only full-year values are kept.

## Step 2: vector index

```bash
python ingest/parse_10k.py            # 10-K HTML -> clean text split by Item
python ingest/build_vector_index.py   # chunks -> embeddings (nomic-embed-text) -> FAISS
python retrieval/vector_search.py "export controls on AI chips" --ticker NVDA
```

**Parsing.** 10-Ks are inline-XBRL HTML: hidden XBRL blocks, sentences split across many
`<span>` tags, and a table of contents that repeats every Item heading. The parser
removes hidden elements, rebuilds paragraphs and table rows as lines, finds every
"Item N" heading, merges repeated headings of the same item (Microsoft prints a running
"PART II Item 7" header on every page) and keeps the longest section per item, which
skips the table of contents. Items 6, 15 and 16 (reserved, exhibit list, summary) are
dropped. Intel does not organise its 10-K by Item at all; it uses its own chapters and a
cross-reference table, so when Item sections cover less than 30% of a report, the whole
report is kept as one section (item "ALL") instead of being split wrongly. Item 15 is
kept when it is large, because NVIDIA places its financial statements and notes there
rather than in Item 8.

**Chunking.** About 1,500 characters per chunk, cut at line boundaries with a
200-character overlap. Each chunk is embedded together with a header such as
"NVIDIA (NVDA) 10-K, fiscal year 2025, Item 1A: Risk Factors", so retrieval respects
which company and year a passage belongs to.

**Index.** `nomic-embed-text` through Ollama, with its separate document and query
prefixes; vectors are normalised and stored in a FAISS inner-product index (cosine
similarity). Search supports filters by company, fiscal year and section, and skips
passages that are near-copies of one already returned (10-Ks repeat paragraphs across
sections and years).

## Step 3a: knowledge graph (structured facts)

```bash
python ingest/parse_subsidiaries.py   # Exhibit 21 lists -> data/graph/subsidiaries.csv
python ingest/classify_risks.py       # risk passages -> topics, data/graph/risk_topics.csv
python graph/load_graph.py --reset    # load into Neo4j
```

| Node | From |
|---|---|
| `Company`, `Report` | SEC company data and the filing manifest (with source URLs for citations) |
| `Subsidiary`, `Jurisdiction` | Exhibit 21 of each 10-K: tables, ownership-percentage tables and plain lists are all parsed; US states map to country "United States" |
| `RiskTopic` | 15 fixed topics; each risk-factor passage is assigned to its most similar topic description using the stored embeddings (no LLM calls), and a company cites a topic in a year when at least two passages are assigned to it |

Relationships: `(Company)-[:FILED]->(Report)`,
`(Company)-[:HAS_SUBSIDIARY {fiscal_years}]->(Subsidiary)-[:INCORPORATED_IN]->(Jurisdiction)`,
`(Company)-[:CITES_RISK {fiscal_year, passages, evidence}]->(RiskTopic)`, where `evidence`
holds the chunk ids of the strongest passages so answers can cite them.
