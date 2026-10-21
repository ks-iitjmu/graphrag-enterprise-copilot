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
- [ ] Step 2: chunk and embed report text into FAISS
- [ ] Step 3: build the knowledge graph in Neo4j
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
