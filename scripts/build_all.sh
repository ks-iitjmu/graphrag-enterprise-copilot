#!/usr/bin/env bash
# Build all data from scratch, in order. Safe to re-run: the SEC download and the LLM
# extraction reuse what they already have (the extraction resumes from its cache).
#   bash scripts/build_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a

step() { echo; echo "==== $1"; }
step "1/8  Checking the setup";            python scripts/check_setup.py
step "2/8  Downloading SEC filings";       python ingest/sec_download.py
step "3/8  Building the financial database"; python ingest/build_sql.py
step "4/8  Parsing the 10-K reports";      python ingest/parse_10k.py
step "5/8  Building the vector index";     python ingest/build_vector_index.py
step "6/8  Parsing subsidiary lists";      python ingest/parse_subsidiaries.py
step "7/8  LLM extraction (about 45 min on an RTX A4000; resumes if interrupted)"
python ingest/extract_entities.py
step "8/8  Loading the knowledge graph";   python graph/load_graph.py --reset
echo; echo "Done. Start the app with:  uvicorn api.main:app --host 0.0.0.0 --port 8000 --no-server-header"
