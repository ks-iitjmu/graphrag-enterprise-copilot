"""The copilot agent, built with LangGraph.

    route -> run_tools -> (fallback to vector search if nothing was found) -> answer
          -> check_citations (citations of sources that were never retrieved are flagged)

    from agent.graph import ask
    result = ask("Which companies name AMD as a competitor?")

Command line:
    python agent/graph.py "Which companies name AMD as a competitor?"
    python agent/graph.py --verbose "..."     shows routing, queries and contexts
"""
import argparse
import re
import sys
import time
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

sys.path.append(str(Path(__file__).resolve().parents[1]))
from agent.prompts import ANSWER_SCHEMA, ANSWER_SYSTEM, ROUTER_SCHEMA, ROUTER_SYSTEM, TOOLS  # noqa: E402
from core.entities import companies_in, years_in  # noqa: E402
from core.ollama import chat_json  # noqa: E402
from tools import cypher_tool, sql_tool, vector_tool  # noqa: E402
from tools.base import ToolResult  # noqa: E402

RUNNERS = {"sql": sql_tool.answer, "cypher": cypher_tool.answer, "vector": vector_tool.answer}
SOURCE_ID = re.compile(r"^([A-Z]{2,5}_\d{4}_[0-9A-Z]+_\d{4}|SQL)$")
BRACKETS = re.compile(r"\[([^\[\]]+)\]")
PSEUDO_SOURCE = re.compile(r"cypher|graph|query|vector|context|database|knowledge", re.I)
NOT_FOUND = re.compile(r"(does not|doesn't|do not|did not) (contain|provide|include|mention|cover)|"
                       r"not (available|provided|included|found|mentioned)|no information", re.I)
ORDER = ["cypher", "sql", "vector"]        # graph first, so its companies can inform the others
NAMES = {"AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet", "AMZN": "Amazon",
         "META": "Meta Platforms", "NVDA": "NVIDIA", "INTC": "Intel", "AMD": "AMD"}


def cited_ids(text: str) -> list[str]:
    ids = []
    for group in BRACKETS.findall(text):
        ids += [x.strip() for x in group.split(",") if SOURCE_ID.match(x.strip())]
    return ids


class AgentState(TypedDict, total=False):
    question: str
    tools: list[str]
    sub_questions: dict
    results: list[ToolResult]
    answer: str
    citations: list[str]
    removed_citations: list[str]
    timings: dict


def _timed(state: AgentState, step: str, start: float) -> dict:
    return {**state.get("timings", {}), step: round(time.time() - start, 2)}


def route(state: AgentState) -> AgentState:
    t0 = time.time()
    out = chat_json(ROUTER_SYSTEM, f"Question: {state['question']}", ROUTER_SCHEMA, max_tokens=300)
    tools = [t for t in dict.fromkeys(out.get("tools", [])) if t in TOOLS] or ["vector"]
    subs = out.get("sub_questions", {}) or {}
    sub_questions = {t: (subs.get(t) or state["question"]) for t in tools}
    return {"tools": tools, "sub_questions": sub_questions, "timings": _timed(state, "route", t0)}


def companies_from(result: ToolResult, limit: int = 4) -> list[str]:
    """Companies named in a tool's rows, in row order (graph results are ranked)."""
    found = []
    for row in result.rows:
        for ticker in companies_in(" ".join(map(str, row))):
            if ticker not in found:
                found.append(ticker)
    return found[:limit]


def run_tools(state: AgentState) -> AgentState:
    t0 = time.time()
    results, carried = [], []
    original_years = years_in(state["question"])
    for tool in sorted(state["tools"], key=ORDER.index):
        sub = state["sub_questions"][tool]
        if carried and not companies_in(sub):          # pass companies found by the graph on
            sub += f" (companies: {', '.join(NAMES[t] for t in carried)})"
        try:
            if tool == "vector":                        # years only from the user's own words
                res = vector_tool.answer(sub, tickers=companies_in(sub), years=original_years)
            else:
                res = RUNNERS[tool](sub)
        except Exception as e:                          # a broken tool must not break the agent
            res = ToolResult(tool=tool, question=sub, error=f"{type(e).__name__}: {e}")
        res.question = sub
        results.append(res)
        if tool == "cypher" and res.ok:
            carried = companies_from(res)
    return {"results": results, "timings": _timed(state, "tools", t0)}


def needs_fallback(state: AgentState) -> str:
    found = any(r.ok for r in state["results"])
    tried_vector = any(r.tool == "vector" for r in state["results"])
    return "answer" if found or tried_vector else "fallback"


def fallback(state: AgentState) -> AgentState:
    t0 = time.time()
    res = vector_tool.answer(state["question"])
    return {"results": state["results"] + [res], "tools": state["tools"] + ["vector"],
            "sub_questions": {**state["sub_questions"], "vector": state["question"]},
            "timings": _timed(state, "fallback", t0)}


def build_context(results: list[ToolResult]) -> str:
    parts = []
    for r in results:
        if r.tool == "sql" and r.ok:
            parts.append("[SQL] Financial data (SEC XBRL):\n" + r.as_context())
        elif r.tool == "cypher" and r.ok:
            parts.append("Knowledge graph rows (evidence = passage ids):\n" + r.as_context())
        elif r.tool == "vector" and r.ok:
            parts.append("Report passages:\n" + r.as_context())
        else:
            parts.append(f"[{r.tool}] returned nothing" + (f" ({r.error})" if r.error else ""))
    return "\n\n---\n\n".join(parts)


def allowed_citations(results: list[ToolResult]) -> set:
    """Ids the answer may cite: retrieved passage ids, plus SQL when financial data was found."""
    allowed = {"SQL"} if any(r.tool == "sql" and r.ok for r in results) else set()
    for r in results:
        if r.tool != "sql" and r.ok:
            allowed.update(r.evidence)
    return allowed


def answer(state: AgentState) -> AgentState:
    t0 = time.time()
    context = build_context(state["results"])
    user = f"Question: {state['question']}\n\nContext:\n{context}"
    text = chat_json(ANSWER_SYSTEM, user, ANSWER_SCHEMA, num_ctx=12288,
                     max_tokens=900).get("answer", "").strip()
    if allowed_citations(state["results"]) and not cited_ids(text) and not NOT_FOUND.search(text):
        # Sources were available but none was cited: ask once for a cited rewrite
        retry = (f"{user}\n\nYour previous answer cited no sources:\n{text}\n"
                 "Rewrite it, citing the source id in square brackets after every claim.")
        text = chat_json(ANSWER_SYSTEM, retry, ANSWER_SCHEMA, num_ctx=12288,
                         max_tokens=900).get("answer", "").strip() or text
    return {"answer": text, "timings": _timed(state, "answer", t0)}


def check_citations(state: AgentState) -> AgentState:
    """Validate every bracketed citation. Retrieved ids are kept (one per bracket);
    ids that were never retrieved, and pseudo-sources such as [cypher query], are
    replaced by "(unverified)", so an answer cannot point to a source that does not exist."""
    allowed = allowed_citations(state["results"])
    cited, flagged = [], []

    def fix(match):
        tokens = [x.strip() for x in match.group(1).split(",")]
        if not any(SOURCE_ID.match(x) or PSEUDO_SOURCE.search(x) for x in tokens):
            return match.group(0)                       # ordinary brackets, not a citation
        keep = []
        for x in tokens:
            if x in allowed:
                keep.append(x)
                cited.append(x)
            else:
                flagged.append(x)
        return "".join(f"[{x}]" for x in keep) if keep else "(unverified)"

    text = BRACKETS.sub(fix, state["answer"])
    return {"answer": re.sub(r"[ \t]{2,}", " ", text).strip(),
            "citations": list(dict.fromkeys(cited)), "removed_citations": sorted(set(flagged))}


def build_agent():
    g = StateGraph(AgentState)
    g.add_node("route", route)
    g.add_node("run_tools", run_tools)
    g.add_node("fallback", fallback)
    g.add_node("answer", answer)
    g.add_node("check_citations", check_citations)
    g.add_edge(START, "route")
    g.add_edge("route", "run_tools")
    g.add_conditional_edges("run_tools", needs_fallback, {"answer": "answer", "fallback": "fallback"})
    g.add_edge("fallback", "answer")
    g.add_edge("answer", "check_citations")
    g.add_edge("check_citations", END)
    return g.compile()


_AGENT = None


def ask(question: str) -> AgentState:
    global _AGENT
    if _AGENT is None:
        _AGENT = build_agent()
    return _AGENT.invoke({"question": question, "timings": {}})


def print_result(s: AgentState, verbose: bool = False):
    print(f"\nRoute: {', '.join(s['tools'])}")
    for r in s["results"]:
        status = "ok" if r.ok else f"no result {r.error}".strip()
        print(f"  {r.tool:6s} {status}; attempts {r.attempts}; {s['sub_questions'].get(r.tool, '')}")
        if verbose and r.query:
            print(f"         {r.query}")
    if verbose:
        print("\nContext:\n" + build_context(s["results"])[:4000])
    print("\nAnswer:\n" + s["answer"])
    if s.get("removed_citations"):
        print(f"\n(flagged as unverified, never retrieved: {', '.join(s['removed_citations'])})")
    print(f"\nTimings (s): {s.get('timings')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="+")
    ap.add_argument("--verbose", "-v", action="store_true")
    a = ap.parse_args()
    print_result(ask(" ".join(a.question)), a.verbose)
