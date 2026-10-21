"""Parse Exhibit 21 (list of subsidiaries) of every 10-K into one table.

Companies format the exhibit differently:
  - two-column tables (name | jurisdiction), sometimes with an ownership-percentage column
  - plain lines such as "Facebook Holdings, LLC (Delaware)"
  - name and jurisdiction on alternating lines (Microsoft, AMD), paired by recognising
    a line that is a known US state or country
All are reduced to (ticker, fiscal_year, subsidiary, jurisdiction, country).

Output: data/graph/subsidiaries.csv
Usage:  python ingest/parse_subsidiaries.py
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import DATA, RAW  # noqa: E402
from ingest.parse_10k import html_to_lines  # noqa: E402

GRAPH = DATA / "graph"
US_STATES = {
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut",
    "Delaware", "District of Columbia", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois",
    "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland",
    "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana",
    "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania",
    "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah",
    "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming",
}
COUNTRIES = {
    "United States", "Argentina", "Australia", "Austria", "Bahamas", "Bahrain", "Bangladesh",
    "Belgium", "Bermuda", "Brazil", "British Virgin Islands", "Bulgaria", "Canada",
    "Cayman Islands", "Chile", "China", "Colombia", "Costa Rica", "Croatia", "Cyprus",
    "Czech Republic", "Denmark", "Egypt", "Estonia", "Finland", "France", "Germany", "Ghana",
    "Gibraltar", "Greece", "Guernsey", "Hong Kong", "Hungary", "Iceland", "India", "Indonesia",
    "Ireland", "Isle of Man", "Israel", "Italy", "Japan", "Jersey", "Jordan", "Kenya",
    "Latvia", "Lithuania", "Luxembourg", "Macau", "Malaysia", "Malta", "Mauritius", "Mexico",
    "Morocco", "Netherlands", "New Zealand", "Nigeria", "Norway", "Pakistan", "Panama", "Peru",
    "Philippines", "Poland", "Portugal", "Puerto Rico", "Qatar", "Romania", "Russia",
    "Saudi Arabia", "Serbia", "Singapore", "Slovakia", "Slovenia", "South Africa",
    "South Korea", "Spain", "Sri Lanka", "Sweden", "Switzerland", "Taiwan", "Thailand",
    "Turkey", "Türkiye", "Ukraine", "United Arab Emirates", "United Kingdom", "Uruguay",
    "Vietnam", "Viet Nam",
}
ALIASES = {"The Netherlands": "Netherlands", "Holland": "Netherlands", "UK": "United Kingdom",
           "England": "United Kingdom", "England and Wales": "United Kingdom",
           "Scotland": "United Kingdom", "U.K.": "United Kingdom", "USA": "United States",
           "U.S.A.": "United States", "Republic of Korea": "South Korea", "Korea": "South Korea",
           "People's Republic of China": "China", "PRC": "China", "Hong Kong SAR": "Hong Kong",
           "Grand Cayman": "Cayman Islands", "British Virgin Islands": "British Virgin Islands"}
HEADER_WORDS = re.compile(r"\b(name|jurisdiction|incorporat|organiz|subsidiar|state or|"
                          r"country|domicile|percent|ownership|exhibit)\b", re.I)


def clean_jurisdiction(j: str) -> str:
    j = re.sub(r"\s+", " ", j).strip(" .,;*")
    j = re.sub(r"^(the )?(state of|commonwealth of|republic of) ", "", j, flags=re.I)
    j = re.sub(r",\s*(U\.?S\.?A?\.?|United States( of America)?)$", "", j, flags=re.I)  # "Delaware, U.S."
    if re.fullmatch(r"U\.?S\.?A?\.?|United States of America", j, flags=re.I):
        j = "United States"
    return ALIASES.get(j, j)


def clean_name(name: str) -> str:
    name = re.sub(r"\s*\(\d+\)\s*$", "", name)          # footnote markers: "Corporation(1)"
    return re.sub(r"\s+", " ", name).strip(" ,;*")


def is_jurisdiction(text: str) -> bool:
    j = clean_jurisdiction(text)
    return j in US_STATES or j in COUNTRIES


def country_of(jurisdiction: str) -> str:
    return "United States" if jurisdiction in US_STATES else jurisdiction


def plausible(name: str, jurisdiction: str) -> bool:
    return (3 <= len(name) <= 150 and 2 <= len(jurisdiction) <= 40
            and not re.search(r"\d", jurisdiction) and not HEADER_WORDS.search(jurisdiction))


def parse_exhibit(html: bytes) -> list[tuple[str, str]]:
    out, pending = [], None        # pending: a possible name awaiting a jurisdiction line
    for line in html_to_lines(html):
        name = jurisdiction = None
        if "|" in line:
            cells = [c.strip() for c in line.split("|")]
            cells = [c for c in cells if c and not re.fullmatch(r"[\d.%\s]+", c)]  # drop % columns
            if len(cells) >= 2:
                name, jurisdiction = cells[0], cells[-1]
        elif is_jurisdiction(line) and pending:                # alternating-line format
            name, jurisdiction = pending, line
        else:
            m = (re.match(r"^(.*?)\s*\(([^()]+)\)\s*$", line)
                 or re.match(r"^(.*?)\s+[–—-]\s+([^–—-]+)$", line))
            if m and is_jurisdiction(m.group(2)):
                name, jurisdiction = m.group(1), m.group(2)
        if name is None:
            pending = clean_name(line) if 3 <= len(line) <= 150 else None
            continue
        pending = None
        if HEADER_WORDS.search(name) and HEADER_WORDS.search(jurisdiction):
            continue                                            # table header row
        name, jurisdiction = clean_name(name), clean_jurisdiction(jurisdiction)
        if plausible(name, jurisdiction) and not is_jurisdiction(name):
            out.append((name, jurisdiction))
    return list(dict.fromkeys(out))                             # unique, order kept


def main():
    manifest = json.loads((RAW / "manifest.json").read_text())
    rows = []
    for f in manifest:
        path = RAW / f["ticker"] / f"{f['fiscal_year']}_ex21.htm"
        if not path.exists():
            print(f"{f['ticker']} FY{f['fiscal_year']}: no exhibit 21")
            continue
        subs = parse_exhibit(path.read_bytes())
        rows += [{"ticker": f["ticker"], "fiscal_year": f["fiscal_year"], "subsidiary": n,
                  "jurisdiction": j, "country": country_of(j)} for n, j in subs]
        top = pd.Series([country_of(j) for _, j in subs]).value_counts().head(3)
        print(f"{f['ticker']} FY{f['fiscal_year']}: {len(subs):4d} subsidiaries | "
              + ", ".join(f"{c} {n}" for c, n in top.items()))
    GRAPH.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(GRAPH / "subsidiaries.csv", index=False)
    print(f"\nWrote {GRAPH / 'subsidiaries.csv'}: {len(rows):,} rows")


if __name__ == "__main__":
    main()
