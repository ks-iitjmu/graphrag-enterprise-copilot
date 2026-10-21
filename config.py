"""Project-wide settings."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"                 # downloaded filings and XBRL facts
SQL_DB = DATA / "finance.db"       # SQLite database of financial facts
SECTIONS = DATA / "sections"       # parsed 10-K sections (one JSON per report)
INDEX = DATA / "index"             # FAISS index and chunk metadata

# Companies covered: ticker -> display name
COMPANIES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet", "AMZN": "Amazon",
    "META": "Meta Platforms", "NVDA": "NVIDIA", "INTC": "Intel", "AMD": "AMD",
}
N_ANNUAL_REPORTS = 3   # most recent 10-K filings per company
N_FINANCIAL_YEARS = 5  # fiscal years of financial data kept in SQL

# The SEC requires a User-Agent naming you and giving a contact email.
# Set it once per terminal:  export SEC_USER_AGENT="Kunal Sharma your.email@example.com"
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "")

# Local models served by Ollama
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3:14b")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")

# Chunking: about 300 words per chunk, with overlap so sentences are not cut off
CHUNK_CHARS = 1500
CHUNK_OVERLAP = 200
