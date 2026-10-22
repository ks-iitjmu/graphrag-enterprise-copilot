"""Shared result type for the retrieval tools."""
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
        lines += [" | ".join(_fmt(v) for v in r) for r in self.rows[:max_rows]]
        if len(self.rows) > max_rows:
            lines.append(f"... {len(self.rows) - max_rows} more rows")
        return f"[{self.tool} query: {self.query}]\n" + "\n".join(lines)


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.4g}"
    if isinstance(v, list):
        return ", ".join(map(str, v[:5])) + (" ..." if len(v) > 5 else "")
    return str(v)
