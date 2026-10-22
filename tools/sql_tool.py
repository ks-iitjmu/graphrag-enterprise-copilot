"""Question -> SQL over data/finance.db -> rows.

Safeguards: the database is opened read-only, only a single SELECT/WITH statement is
accepted, a row limit is added when missing, and a failed query is retried with the
database error shown to the LLM.

    python tools/sql_tool.py "Which company had the highest net margin in fiscal 2025?"
"""
import re
import sqlite3
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import SQL_DB  # noqa: E402
from core.ollama import chat_json  # noqa: E402
from tools.base import ToolResult  # noqa: E402

MAX_ROWS = 200
RETRIES = 2
FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|"
                       r"pragma|vacuum|reindex|truncate)\b", re.I)

SCHEMA = """SQLite database of annual financial data from SEC filings (XBRL), 8 companies.

View annual_financials: one row per company and fiscal year.
  ticker TEXT          AAPL, MSFT, GOOGL, AMZN, META, NVDA, INTC, AMD
  company TEXT         Apple, Microsoft, Alphabet, Amazon, Meta Platforms, NVIDIA, Intel, AMD
  fiscal_year INTEGER  named by the calendar year in which the fiscal year ENDS
  period_end TEXT      fiscal year end date, YYYY-MM-DD
  revenue, cost_of_revenue, gross_profit, research_and_development, operating_income,
  net_income, operating_cash_flow, capital_expenditure, share_repurchases,
  total_assets, total_liabilities, cash_and_equivalents, stockholders_equity   REAL, in US dollars
  eps_diluted REAL     US dollars per share
Values are NULL when a company does not report that item (e.g. Amazon reports no gross
profit or R&D line). Fiscal years end in different months: Apple late September,
Microsoft June, NVIDIA late January, the others December.

Table companies(ticker, name, cik, industry, fiscal_year_end, state_of_incorporation)."""

EXAMPLES = """Examples:
Q: What was NVIDIA's revenue in fiscal 2025?
SQL: SELECT company, fiscal_year, revenue FROM annual_financials WHERE ticker = 'NVDA' AND fiscal_year = 2025
Q: Which company had the highest net margin in 2024?
SQL: SELECT company, fiscal_year, net_income * 1.0 / revenue AS net_margin FROM annual_financials WHERE fiscal_year = 2024 AND revenue > 0 ORDER BY net_margin DESC
Q: How did Intel's and AMD's revenue grow year over year?
SQL: SELECT company, fiscal_year, revenue, revenue * 1.0 / LAG(revenue) OVER (PARTITION BY ticker ORDER BY fiscal_year) - 1 AS growth FROM annual_financials WHERE ticker IN ('INTC', 'AMD') ORDER BY company, fiscal_year
Q: Compare R&D as a share of revenue in each company's latest fiscal year.
SQL: SELECT a.company, a.fiscal_year, a.research_and_development * 1.0 / a.revenue AS rd_share FROM annual_financials a WHERE a.fiscal_year = (SELECT MAX(fiscal_year) FROM annual_financials b WHERE b.ticker = a.ticker) ORDER BY rd_share DESC"""

SYSTEM = f"""You write one SQLite SELECT query that answers the user's question.
{SCHEMA}

Rules: a single SELECT (or WITH ... SELECT) statement; use the annual_financials view;
return the company name, fiscal_year and the values needed to answer; compute ratios
with * 1.0 to avoid integer division; when the question says "latest" use each company's
own most recent fiscal year. If the question cannot be answered from this data, return
an empty string as sql.

{EXAMPLES}"""

OUTPUT = {"type": "object",
          "properties": {"sql": {"type": "string"}},
          "required": ["sql"]}


def check_sql(sql: str) -> str:
    """Return a cleaned read-only query, or raise ValueError."""
    sql = sql.strip().rstrip(";").strip()
    if not sql:
        raise ValueError("no query generated: the question may not be answerable from financial data")
    if ";" in sql:
        raise ValueError("only one statement is allowed")
    if not re.match(r"^(select|with)\b", sql, re.I) or FORBIDDEN.search(sql):
        raise ValueError("only read-only SELECT queries are allowed")
    if not re.search(r"\blimit\s+\d+\s*$", sql, re.I):
        sql += f" LIMIT {MAX_ROWS}"
    return sql


def run_sql(sql: str) -> tuple[list, list]:
    con = sqlite3.connect(f"file:{SQL_DB}?mode=ro", uri=True)
    try:
        cur = con.execute(sql)
        return [d[0] for d in cur.description], [list(r) for r in cur.fetchall()]
    finally:
        con.close()


def answer(question: str) -> ToolResult:
    res = ToolResult(tool="sql", question=question)
    user = f"Question: {question}"
    for attempt in range(1, RETRIES + 2):
        res.attempts = attempt
        sql = chat_json(SYSTEM, user, OUTPUT, max_tokens=500).get("sql", "")
        try:
            res.query = check_sql(sql)
            res.columns, res.rows = run_sql(res.query)
            res.error = ""
            res.evidence = [f"SEC XBRL financial data (finance.db): {res.query}"]
            return res
        except (ValueError, sqlite3.Error) as e:
            res.query, res.error = sql, str(e)
            if isinstance(e, ValueError) and "no query" in str(e):
                return res
            user = (f"Question: {question}\nYour previous query:\n{sql}\nfailed with: {e}\n"
                    "Write a corrected query.")
    return res


if __name__ == "__main__":
    r = answer(" ".join(sys.argv[1:]))
    print(r.as_context() if not r.error else f"Failed after {r.attempts} attempts: {r.error}\n{r.query}")
