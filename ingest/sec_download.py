"""Download annual reports and financial facts from SEC EDGAR.

For each company in config.COMPANIES:
  - the most recent 10-K filings (main document, HTML)
  - Exhibit 21 of each filing: the official list of subsidiaries
  - XBRL "company facts": every reported financial value, used for the SQL database

Output (data/raw/):
  companies.json                   CIK, name, industry, fiscal year end per ticker
  <TICKER>/companyfacts.json
  <TICKER>/<fiscal_year>_10k.htm
  <TICKER>/<fiscal_year>_ex21.htm
  manifest.json                    one record per downloaded filing

The SEC asks for at most 10 requests per second and a User-Agent with contact details.
Usage:  export SEC_USER_AGENT="Your Name you@example.com"
        python ingest/sec_download.py
"""
import json
import re
import sys
import time
from pathlib import Path

import requests

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import COMPANIES, N_ANNUAL_REPORTS, RAW, SEC_USER_AGENT  # noqa: E402

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{name}"
FILING_INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{acc_dashed}-index.htm"


class Edgar:
    def __init__(self, user_agent: str):
        if "@" not in user_agent:
            sys.exit('Set SEC_USER_AGENT first, e.g.\n'
                     '  export SEC_USER_AGENT="Your Name you@example.com"')
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})

    def get(self, url: str) -> requests.Response:
        for attempt in range(4):
            time.sleep(0.15)                      # stay well under 10 requests/second
            r = self.s.get(url, timeout=60)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 500, 502, 503):
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
        r.raise_for_status()

    def json(self, url: str) -> dict:
        return self.get(url).json()


def ticker_to_cik(edgar: Edgar) -> dict:
    data = edgar.json(TICKERS_URL)
    return {row["ticker"]: int(row["cik_str"]) for row in data.values()}


def recent_10ks(edgar: "Edgar", submissions: dict, n: int) -> list:
    """Most recent original 10-K filings (amendments excluded).

    The submissions API lists only about the last 1,000 filings in "recent". Companies
    with many insider-trading forms push older 10-Ks into extra pages, read here."""
    def rows_of(block):
        return [dict(form=f, accession=a, filed=d, period=p, primary=doc)
                for f, a, d, p, doc in zip(block["form"], block["accessionNumber"],
                                           block["filingDate"], block["reportDate"],
                                           block["primaryDocument"])
                if f == "10-K"]

    rows = rows_of(submissions["filings"]["recent"])
    for page in submissions["filings"].get("files", []):
        if len(rows) >= n:
            break
        rows += rows_of(edgar.json(f"https://data.sec.gov/submissions/{page['name']}"))
    return sorted(rows, key=lambda x: x["filed"], reverse=True)[:n]


def find_exhibit21(edgar: "Edgar", cik: int, acc: str, acc_dashed: str) -> str | None:
    """Exhibit 21 (list of subsidiaries) of a filing.

    Read the filing index page, where the SEC labels each document with its official
    type (EX-21, EX-21.1, ...). File names vary between companies, types do not."""
    page = edgar.get(FILING_INDEX_URL.format(cik=cik, acc=acc, acc_dashed=acc_dashed)).text
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", page, flags=re.S | re.I):
        if re.search(r">\s*EX-21(\.\d+)?\s*<", row, flags=re.I):
            href = re.search(r'href="([^"]+)"', row, flags=re.I)
            if href:
                return href.group(1).replace("/ix?doc=", "").rsplit("/", 1)[-1]
    return None


def main():
    edgar = Edgar(SEC_USER_AGENT)
    RAW.mkdir(parents=True, exist_ok=True)
    ciks = ticker_to_cik(edgar)
    companies, manifest = {}, []

    for ticker, display in COMPANIES.items():
        cik = ciks[ticker]
        out = RAW / ticker
        out.mkdir(exist_ok=True)
        sub = edgar.json(SUBMISSIONS_URL.format(cik=cik))
        companies[ticker] = {"ticker": ticker, "name": display, "cik": cik,
                             "sec_name": sub.get("name"),
                             "industry": sub.get("sicDescription"),
                             "fiscal_year_end": sub.get("fiscalYearEnd"),
                             "state_of_incorporation": sub.get("stateOfIncorporation")}

        facts = edgar.get(FACTS_URL.format(cik=cik)).content
        (out / "companyfacts.json").write_bytes(facts)

        for f in recent_10ks(edgar, sub, N_ANNUAL_REPORTS):
            acc = f["accession"].replace("-", "")
            fy = f["period"][:4]                      # fiscal year named by period-end year
            doc = edgar.get(ARCHIVE_URL.format(cik=cik, acc=acc, name=f["primary"]))
            (out / f"{fy}_10k.htm").write_bytes(doc.content)

            ex21 = find_exhibit21(edgar, cik, acc, f["accession"])
            if ex21:
                (out / f"{fy}_ex21.htm").write_bytes(
                    edgar.get(ARCHIVE_URL.format(cik=cik, acc=acc, name=ex21)).content)

            manifest.append({**f, "ticker": ticker, "cik": cik, "fiscal_year": int(fy),
                             "url": ARCHIVE_URL.format(cik=cik, acc=acc, name=f["primary"]),
                             "exhibit21": ex21})
            print(f"{ticker} FY{fy}: 10-K filed {f['filed']}, "
                  f"{len(doc.content) / 1e6:.1f} MB, exhibit 21: {'yes' if ex21 else 'no'}")

    (RAW / "companies.json").write_text(json.dumps(companies, indent=2))
    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nDownloaded {len(manifest)} annual reports for {len(companies)} companies.")


if __name__ == "__main__":
    main()
