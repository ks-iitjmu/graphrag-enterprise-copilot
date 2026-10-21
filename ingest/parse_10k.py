"""Parse downloaded 10-K HTML into clean text split by the standard Item sections.

10-Ks are inline-XBRL HTML: hidden XBRL blocks, text broken across many <span>s, and a
table of contents that repeats every Item heading. This script
  1. drops hidden elements, scripts and styles,
  2. rebuilds paragraphs and table rows as lines of text,
  3. finds every "Item N" heading, merges repeated headings of the same item (running
     page headers such as "PART II Item 7" on every page), and keeps, for each item,
     the longest section (the table of contents produces tiny sections),
  4. falls back to one whole-report section when Item headings cover too little of the
     document (some companies, such as Intel, use their own chapter structure).

Output: data/sections/<TICKER>_<FY>.json with metadata and a list of sections.
Usage:  python ingest/parse_10k.py
"""
import json
import re
import sys
import warnings
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# Inline-XBRL files start with an XML declaration but are HTML; the warning is noise.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import RAW, SECTIONS  # noqa: E402

ITEM_TITLES = {
    "1": "Business", "1A": "Risk Factors", "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity", "2": "Properties", "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures", "5": "Market for Common Equity",
    "6": "Reserved", "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants", "9A": "Controls and Procedures",
    "9B": "Other Information", "9C": "Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation", "12": "Security Ownership",
    "13": "Certain Relationships and Related Transactions", "14": "Principal Accountant Fees",
    "15": "Exhibits and Financial Statement Schedules", "16": "Form 10-K Summary",
}
SKIP_ITEMS = {"6", "15", "16"}   # reserved, exhibit lists, summary: no useful prose
MIN_COVERAGE = 0.30              # Item sections must cover 30% of the text, else fall back
HEADING = re.compile(r"^(?:part\s+[iv]+\s*[,.\-–—]?\s*)?item\s*(\d{1,2}[abc]?)\s*[.:\-–—]?\s*(.{0,120})$",
                     re.IGNORECASE)
BLOCKS = ["p", "div", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "br", "table"]


def html_to_lines(html: bytes) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    for tag in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
        tag.decompose()
    for tag in soup.find_all(re.compile(r"^ix:header$", re.I)):
        tag.decompose()
    for cell in soup.find_all(["td", "th"]):
        cell.append(" | ")
    for tag in soup.find_all(BLOCKS):
        tag.append("\n")
    text = soup.get_text("").replace("\xa0", " ").replace("\u200b", "")
    text = re.sub(r"\bTable of Contents\b", "", text)          # page-navigation link text
    lines = []
    for raw in text.split("\n"):
        line = re.sub(r"\s+", " ", raw)
        line = re.sub(r"(\s*\|\s*)+", " | ", line).strip(" |")   # tidy empty table cells
        if not line or re.fullmatch(r"[\d\s|$%().,-]{0,6}", line):  # empty lines, page numbers
            continue
        lines.append(line)
    return lines


def split_items(lines: list[str]) -> list[dict]:
    marks = []
    for i, line in enumerate(lines):
        if len(line) > 160:
            continue
        m = HEADING.match(line)
        if m and m.group(1).upper() in ITEM_TITLES:
            item = m.group(1).upper()
            if marks and marks[-1][1] == item:     # running page header: same item again
                continue
            marks.append((i, item))
    best = {}
    for (start, item), nxt in zip(marks, marks[1:] + [(len(lines), None)]):
        body = lines[start + 1:nxt[0]]
        size = sum(len(x) for x in body)
        if size > best.get(item, (0, None))[0]:
            best[item] = (size, start, body)
    sections = [{"item": item, "title": ITEM_TITLES[item], "start_line": start,
                 "text": "\n".join(body), "chars": size}
                for item, (size, start, body) in best.items()
                if item not in SKIP_ITEMS or (item == "15" and size > 50_000)]

    total = sum(len(x) for x in lines)
    if sum(s["chars"] for s in sections) < MIN_COVERAGE * total:
        return [{"item": "ALL", "title": "Full annual report (no standard Item headings)",
                 "start_line": 0, "text": "\n".join(lines), "chars": total}]
    return sorted(sections, key=lambda s: s["start_line"])


def main():
    companies = json.loads((RAW / "companies.json").read_text())
    manifest = json.loads((RAW / "manifest.json").read_text())
    SECTIONS.mkdir(parents=True, exist_ok=True)
    for f in manifest:
        lines = html_to_lines((RAW / f["ticker"] / f"{f['fiscal_year']}_10k.htm").read_bytes())
        sections = split_items(lines)
        doc = {"ticker": f["ticker"], "company": companies[f["ticker"]]["name"],
               "fiscal_year": f["fiscal_year"], "filed": f["filed"], "url": f["url"],
               "sections": sections}
        (SECTIONS / f"{f['ticker']}_{f['fiscal_year']}.json").write_text(json.dumps(doc))
        found = " ".join(s["item"] for s in sections)
        kept = sum(s["chars"] for s in sections)
        coverage = kept / max(sum(len(x) for x in lines), 1)
        print(f"{f['ticker']} FY{f['fiscal_year']}: {len(sections):2d} sections, "
              f"{kept / 1e3:,.0f}k chars ({coverage:.0%} of report) | {found}")


if __name__ == "__main__":
    main()
