"""Check that everything the project needs is installed and running.

    python scripts/check_setup.py

Checks: Python version, packages, the .env settings, Ollama and both models, Neo4j, and
which data files have already been built. Each line says what to do if something is missing.
"""
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
OK, BAD, TODO = "[ OK ]", "[FAIL]", "[TODO]"
problems = 0


def line(status, msg):
    global problems
    problems += status == BAD
    print(f"{status} {msg}")


v = sys.version_info
line(OK if v >= (3, 11) else BAD, f"Python {v.major}.{v.minor}" + ("" if v >= (3, 11) else " (3.11 or newer needed)"))
line(OK if sys.prefix != sys.base_prefix else BAD,
     "virtual environment active" if sys.prefix != sys.base_prefix else "no virtual environment: run  source .venv/bin/activate")

missing = [m for m in ["requests", "pandas", "pyarrow", "bs4", "lxml", "faiss", "neo4j", "langgraph",
                       "ragas", "instructor", "openai", "httpx", "fastapi", "uvicorn", "tabulate"]
           if importlib.util.find_spec(m) is None]
line(BAD if missing else OK, f"packages missing: {', '.join(missing)}  ->  pip install -r requirements.txt"
     if missing else "all Python packages installed")

for var, hint in [("SEC_USER_AGENT", 'your name and email, e.g. "Jane Doe jane@example.com"'),
                  ("NEO4J_PASSWORD", "the password you chose for Neo4j")]:
    val = os.environ.get(var, "")
    line(OK if val else BAD, f"{var} set" if val else
         f"{var} not set: put {hint} in .env, then run  set -a && source .env && set +a")

try:
    import requests
    from config import EMBED_MODEL, LLM_MODEL, OLLAMA_URL
    tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).json()
    names = [m["name"] for m in tags.get("models", [])]
    line(OK, f"Ollama running at {OLLAMA_URL}")
    for model in (LLM_MODEL, EMBED_MODEL):
        found = any(n.startswith(model) for n in names)
        line(OK if found else BAD, f"model {model}" + ("" if found else f" missing  ->  ollama pull {model}"))
except Exception:
    line(BAD, "Ollama not reachable  ->  install from ollama.com, then  systemctl status ollama")

try:
    from core.neo4j_db import driver
    with driver() as db:
        db.verify_connectivity()
        n = db.execute_query("MATCH (n) RETURN count(n) AS n", database_="neo4j").records[0]["n"]
    line(OK, f"Neo4j running ({n:,} nodes)" + ("" if n else ": empty until the graph is loaded"))
except BaseException:
    line(BAD, "Neo4j not reachable  ->  docker compose up -d  (wait ~20 s)")

print("\nData built so far:")
for path, step in [("data/raw/manifest.json", "download: python ingest/sec_download.py"),
                   ("data/finance.db", "financial database: python ingest/build_sql.py"),
                   ("data/index/faiss.index", "vector index: parse_10k.py + build_vector_index.py"),
                   ("data/graph/subsidiaries.csv", "subsidiaries: python ingest/parse_subsidiaries.py"),
                   ("data/graph/competitors.csv", "LLM extraction: python ingest/extract_entities.py (~45 min)")]:
    exists = (ROOT / path).exists()
    print(f"{OK if exists else TODO} {path}" + ("" if exists else f"   <- {step}"))
print("\n" + ("Ready." if not problems else f"{problems} problem(s) to fix first.")
      + "  Build everything at once with:  bash scripts/build_all.sh")
