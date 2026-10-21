"""Project-wide settings."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"                 # downloaded filings and XBRL facts
SQL_DB = DATA / "finance.db"       # SQLite database of financial facts

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
