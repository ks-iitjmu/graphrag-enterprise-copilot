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
- [x] Step 3: build the knowledge graph in Neo4j
- [x] Step 4: vector, Cypher and SQL tools
- [x] Step 5: LangGraph agent with routing and retries
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
| `RiskTopic` | 15 fixed topics, tagged by the LLM in Step 3b (`classify_risks.py` holds the topic list and an embedding-based first version, kept for comparison) |

Relationships: `(Company)-[:FILED]->(Report)`,
`(Company)-[:HAS_SUBSIDIARY {fiscal_years}]->(Subsidiary)-[:INCORPORATED_IN]->(Jurisdiction)`,
`(Company)-[:CITES_RISK {fiscal_year, passages, evidence}]->(RiskTopic)`, where `evidence`
holds the chunk ids of the strongest passages so answers can cite them.

## Step 3b: knowledge graph (LLM extraction)

```bash
python ingest/extract_entities.py --limit 40   # trial run on ~40 passages, check quality and speed
python ingest/extract_entities.py              # full run (resumes from the cache)
python graph/load_graph.py --reset
```

The local LLM (Qwen3 14B through Ollama, structured JSON output, temperature 0) reads
every passage of Item 1 (Business) and Item 1A (Risk Factors) in each company's latest
10-K, or Intel's whole report, and returns competitors, products, segments, partners
(supplier, manufacturer, customer or partner) and risk topics from the fixed list. The
schema is enforced by Ollama, so every answer is valid JSON.

Names are normalised ("Advanced Micro Devices, Inc." -> AMD, "Taiwan Semiconductor
Manufacturing Company Limited" -> TSMC), generic phrases ("large cloud providers") are
dropped, and when a competitor or partner is one of the eight companies it links to that
company's node. Segments and risk topics need at least two supporting passages. Every
relationship stores the chunk ids it came from.

Why the LLM rather than embeddings for risk topics: tagging by embedding similarity
assigned every passage to some topic (3,133 of 3,133), inflated Intel (whose whole report,
not just its risk chapter, was classified), and almost never found "intense competition",
because the topic descriptions overlap. The LLM can return several topics or none.

10-K risk sections mention nearly every topic, so each `CITES_RISK` link also stores
`share` (the topic's fraction of the company's risk passages) and `rank` (its position in
the company's risk profile). Queries can then ask what a company emphasises, not just what
it mentions.

## Step 4: retrieval tools

```bash
python tools/sql_tool.py "Which company had the highest net margin in its latest fiscal year?"
python tools/cypher_tool.py "Which companies name AMD as a competitor?"
python tools/vector_tool.py "What does NVIDIA say about export controls?"
```

Every tool returns a `ToolResult` with its results and the evidence to cite.

| Tool | How it works | Safeguards |
|---|---|---|
| `sql_tool` | The LLM writes a SQLite query against `annual_financials` from the schema, fiscal-year conventions and worked examples | read-only connection; a single SELECT/WITH statement; automatic row limit; on an error the query is retried with the database message |
| `cypher_tool` | The LLM writes a Cypher query from the graph schema, the exact organisation, risk-topic and segment names read from the graph, and worked examples | write clauses rejected; read-only transaction; row limit; retried on errors and on empty results; evidence passage ids collected from the rows |
| `vector_tool` | Semantic search filtered by the companies and years named in the question, balanced across companies when several are named | near-duplicate passages removed; falls back without the year filter if nothing matches |

## Step 5: the agent (LangGraph)

```bash
python agent/graph.py "Which companies emphasise export-control risk, and how did their revenue change from 2024 to 2025?"
python agent/graph.py -v "Why did Intel's revenue decline?"      # -v shows queries and context
```

```
question -> route -> run_tools --(something found)--> answer -> check_citations -> result
                          \--(nothing found)--> vector fallback --/
```

- **route**: the LLM picks one or more sources (sql, cypher, vector) and writes a
  self-contained sub-question for each, so a question such as "which companies
  emphasise export controls, and how did their revenue change?" becomes a graph
  question plus a financial-data question.
- **run_tools**: the graph runs first; when a later sub-question names no company, the
  companies the graph found are passed on to it (so "which companies emphasise export
  controls, and how did their revenue change?" fetches revenue for exactly those
  companies). Vector search takes its year filter only from the user's own words. A
  failing tool returns an error instead of stopping the agent.
- **fallback**: if every chosen tool failed or found nothing, vector search over the
  report text is tried before answering.
- **answer**: the LLM answers only from the retrieved context, cites every claim with a
  passage id or [SQL], and says so when the context does not answer the question. The
  prompt includes examples of correctly cited answers; if an answer still cites nothing
  although sources were retrieved, it is rewritten once with citations.
- **check_citations**: every bracketed citation is validated, including grouped ones
  such as [A, B]; ids that were never retrieved and pseudo-sources such as
  [cypher query] become "(unverified)", so an answer cannot point to a source that does
  not exist. Answers that say the context lacks the information are not forced to cite.
