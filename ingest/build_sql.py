"""Build data/finance.db (SQLite) from the downloaded XBRL company facts.

Tables
  companies(ticker, name, cik, industry, fiscal_year_end, state_of_incorporation)
  financials(ticker, fiscal_year, period_end, metric, value, unit, xbrl_concept, filed)
View
  annual_financials: one row per company and fiscal year, one column per metric.
  This is the table the text-to-SQL tool queries.

Companies tag the same item with different XBRL concepts (for example revenue), so
each metric lists candidate concepts in priority order and the first available wins.
"""
import json
import sqlite3
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import N_FINANCIAL_YEARS, RAW, SQL_DB  # noqa: E402

# metric -> (candidate us-gaap concepts in priority order, unit, kind)
# kind "duration" = value over the fiscal year; "instant" = balance at year end
METRICS = {
    "revenue": (["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
                 "SalesRevenueNet"], "USD", "duration"),
    "cost_of_revenue": (["CostOfRevenue", "CostOfGoodsAndServicesSold"], "USD", "duration"),
    "gross_profit": (["GrossProfit"], "USD", "duration"),
    "research_and_development": (["ResearchAndDevelopmentExpense"], "USD", "duration"),
    "operating_income": (["OperatingIncomeLoss"], "USD", "duration"),
    "net_income": (["NetIncomeLoss"], "USD", "duration"),
    "eps_diluted": (["EarningsPerShareDiluted"], "USD/shares", "duration"),
    "operating_cash_flow": (["NetCashProvidedByUsedInOperatingActivities"], "USD", "duration"),
    "capital_expenditure": (["PaymentsToAcquirePropertyPlantAndEquipment",
                             "PaymentsToAcquireProductiveAssets"], "USD", "duration"),
    "share_repurchases": (["PaymentsForRepurchaseOfCommonStock"], "USD", "duration"),
    "total_assets": (["Assets"], "USD", "instant"),
    "total_liabilities": (["Liabilities"], "USD", "instant"),
    "cash_and_equivalents": (["CashAndCashEquivalentsAtCarryingValue"], "USD", "instant"),
    "stockholders_equity": (["StockholdersEquity"], "USD", "instant"),
}


def annual_values(facts: dict, concept: str, unit: str, kind: str) -> pd.DataFrame:
    rows = facts.get("facts", {}).get("us-gaap", {}).get(concept, {}).get("units", {}).get(unit, [])
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[df["form"].isin(["10-K", "10-K/A"])].copy()
    df["end"] = pd.to_datetime(df["end"])
    if kind == "duration":
        if "start" not in df:
            return df.iloc[0:0]
        days = (df["end"] - pd.to_datetime(df["start"])).dt.days
        df = df[days.between(350, 380)]            # full fiscal years only, no quarters
    elif "start" in df:
        df = df[df["start"].isna()]
    # A value is repeated in later filings as a comparative; keep the latest filing
    df = df.sort_values("filed").drop_duplicates("end", keep="last")
    df["fiscal_year"] = df["end"].dt.year         # fiscal years are named by period-end year
    return df[["fiscal_year", "end", "val", "filed"]]


def build_financials(ticker: str, facts: dict) -> pd.DataFrame:
    out = []
    for metric, (concepts, unit, kind) in METRICS.items():
        chosen = {}
        for concept in concepts:                   # earlier concepts take priority per year
            df = annual_values(facts, concept, unit, kind)
            for r in df.itertuples():
                chosen.setdefault(r.fiscal_year, (r.end, r.val, concept, r.filed))
        for fy, (end, val, concept, filed) in chosen.items():
            out.append({"ticker": ticker, "fiscal_year": int(fy),
                        "period_end": end.date().isoformat(), "metric": metric,
                        "value": float(val), "unit": unit, "xbrl_concept": concept,
                        "filed": filed})
    df = pd.DataFrame(out)
    if df.empty:
        return df
    latest = df["fiscal_year"].max()
    return df[df["fiscal_year"] > latest - N_FINANCIAL_YEARS]


def main():
    companies = json.loads((RAW / "companies.json").read_text())
    frames = []
    for ticker in companies:
        facts = json.loads((RAW / ticker / "companyfacts.json").read_text())
        df = build_financials(ticker, facts)
        frames.append(df)
        years = sorted(int(y) for y in df["fiscal_year"].unique()) if len(df) else []
        print(f"{ticker}: {len(df)} values, fiscal years {years}")
    fin = pd.concat(frames, ignore_index=True)

    SQL_DB.unlink(missing_ok=True)
    con = sqlite3.connect(SQL_DB)
    pd.DataFrame(companies.values())[["ticker", "name", "cik", "industry", "fiscal_year_end",
                                      "state_of_incorporation"]].to_sql("companies", con, index=False)
    fin.to_sql("financials", con, index=False)
    cols = ",\n  ".join(f"MAX(CASE WHEN metric = '{m}' THEN value END) AS {m}" for m in METRICS)
    con.execute(f"""CREATE VIEW annual_financials AS
SELECT f.ticker, c.name AS company, f.fiscal_year, MAX(f.period_end) AS period_end,
  {cols}
FROM financials f JOIN companies c USING (ticker)
GROUP BY f.ticker, f.fiscal_year""")
    con.commit()
    n = con.execute("SELECT COUNT(*) FROM annual_financials").fetchone()[0]
    print(f"\nWrote {SQL_DB}: {len(fin)} values, {n} company-years in annual_financials")
    print(pd.read_sql("SELECT company, fiscal_year, ROUND(revenue / 1e9, 1) AS revenue_bn, "
                      "ROUND(net_income / 1e9, 1) AS net_income_bn FROM annual_financials "
                      "ORDER BY company, fiscal_year DESC", con).to_string(index=False))
    con.close()


if __name__ == "__main__":
    main()
