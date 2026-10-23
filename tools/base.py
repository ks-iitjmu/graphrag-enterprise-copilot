"""Shared result type for the retrieval tools."""
import re
from dataclasses import dataclass, field


@dataclass
class ToolResult:
    tool: str                          # "sql", "cypher" or "vector"
    question: str
    query: str = ""                    # generated SQL or Cypher (empty for vector search)
    columns: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    passages: list = field(default_factory=list)   # vector search hits
    evidence: list = field(default_factory=list)   # chunk ids or source descriptions to cite
    error: str = ""
    attempts: int = 0

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.rows or self.passages)

    def as_context(self, max_rows: int = 40) -> str:
        """Compact text for the answer-writing LLM."""
        if self.error:
            return f"[{self.tool}] failed: {self.error}"
        if self.tool == "vector":
            return "\n\n".join(f"[{p['chunk_id']}] {p['header']}\n{p['text']}" for p in self.passages)
        lines = [" | ".join(map(str, self.columns))]
        lines += [" | ".join(_fmt(v, c) for v, c in zip(r, self.columns)) for r in self.rows[:max_rows]]
        if len(self.rows) > max_rows:
            lines.append(f"... {len(self.rows) - max_rows} more rows")
        return f"[{self.tool} query: {self.query}]\n" + "\n".join(lines)


RATIO_COLUMN = re.compile(r"margin|growth|share|ratio|rate|change", re.I)
PERCENT_COLUMN = re.compile(r"percent|pct", re.I)


def _fmt(v, column: str = "") -> str:
    # Ratios such as net_margin = -0.00505 are shown as percentages (-0.5%), so the
    # answer is written in the units people expect. Columns already in percent are kept.
    if (isinstance(v, float) and abs(v) <= 5 and RATIO_COLUMN.search(str(column))
            and not PERCENT_COLUMN.search(str(column))):
        return f"{v * 100:.1f}%"
    if isinstance(v, float) and abs(v) >= 1e6:      # dollars: give the billions form too
        sign = "-" if v < 0 else ""
        return f"{v:,.0f} (= {sign}${abs(v) / 1e9:,.2f} billion)"
    if isinstance(v, float):
        return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.4g}"
    if isinstance(v, list):
        return ", ".join(map(str, v[:5])) + (" ..." if len(v) > 5 else "")
    return str(v)
