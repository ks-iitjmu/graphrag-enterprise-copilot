"""Detect which companies and fiscal years a question refers to (used to filter searches)."""
import re

ALIASES = {
    "AAPL": ["apple", "aapl", "iphone"],
    "MSFT": ["microsoft", "msft", "azure", "linkedin"],
    "GOOGL": ["alphabet", "google", "googl", "youtube"],
    "AMZN": ["amazon", "amzn", "aws"],
    "META": ["meta", "facebook", "instagram", "whatsapp"],
    "NVDA": ["nvidia", "nvda"],
    "INTC": ["intel", "intc"],
    "AMD": ["amd", "advanced micro devices"],
}


def companies_in(text: str) -> list[str]:
    t = text.lower()
    return [ticker for ticker, names in ALIASES.items()
            if any(re.search(rf"\b{re.escape(n)}\b", t) for n in names)]


def years_in(text: str) -> list[int]:
    return sorted({int(y) for y in re.findall(r"\b(20[12]\d)\b", text)})
