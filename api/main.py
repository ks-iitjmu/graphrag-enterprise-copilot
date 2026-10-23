"""HTTP API for the copilot.

    uvicorn api.main:app --host 0.0.0.0 --port 8000 --no-server-header

Endpoints
  POST /ask              question -> answer, citations resolved to sources, route, timings
  GET  /sources/{id}     full text and metadata of a cited passage
  GET  /health           status of each dependency (Ollama, Neo4j, SQL database, vector index)
  GET  /                 web page for asking questions
  GET  /docs             interactive API documentation (can be disabled with ENABLE_DOCS=0)

Security controls live in api/security.py and are configured with environment variables:
optional access key, per-client rate limit, request-size limit, security headers with a
strict Content Security Policy, CORS closed by default, generic error messages. The tools
behind the agent are read-only: a read-only SQLite connection and read-only Neo4j
transactions with write clauses rejected.
"""
import logging
import re
import sqlite3
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import requests
from fastapi import Depends, FastAPI, HTTPException, Path as PathParam
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

sys.path.append(str(Path(__file__).resolve().parents[1]))
from api.security import (ALLOWED_ORIGINS, BUSY_WAIT_SECONDS, ENABLE_DOCS,  # noqa: E402
                          SecurityHeaders, limit_questions, require_api_key)
from config import INDEX, LLM_MODEL, OLLAMA_URL, SQL_DB  # noqa: E402

log = logging.getLogger("copilot.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

STATIC = Path(__file__).resolve().parent / "static"
CHUNK_ID = r"^[A-Z]{2,5}_\d{4}_[0-9A-Z]{1,3}_\d{4}$"
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SOURCE_LABELS = {
    "SQL": "Financial data from SEC XBRL filings (annual_financials)",
    "GRAPH": "Knowledge graph built from the 10-K reports and their Exhibit 21 subsidiary lists",
}
_model_lock = threading.Lock()     # one question at a time: one local GPU serves one request well


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500,
                          examples=["Which companies name AMD as a competitor?"])
    include_details: bool = Field(False, description="also return sub-questions, queries and result sizes")

    @field_validator("question")
    @classmethod
    def clean(cls, v: str) -> str:
        v = CONTROL_CHARS.sub("", v).strip()
        if len(v) < 3:
            raise ValueError("question is too short")
        return v


class Citation(BaseModel):
    id: str
    label: str
    company: str | None = None
    fiscal_year: int | None = None
    section: str | None = None
    url: str | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    unverified_citations: list[str]
    route: list[str]
    timings: dict
    seconds: float
    details: dict | None = None


@lru_cache(maxsize=1)
def chunk_table():
    import pandas as pd
    return pd.read_parquet(INDEX / "chunks.parquet").set_index("chunk_id")


def resolve(citation_id: str) -> Citation:
    if citation_id in SOURCE_LABELS:
        return Citation(id=citation_id, label=SOURCE_LABELS[citation_id])
    chunks = chunk_table()
    if citation_id not in chunks.index:
        return Citation(id=citation_id, label="Unknown passage")
    row = chunks.loc[citation_id]
    url = row["url"] if str(row["url"]).startswith("https://www.sec.gov/") else None
    return Citation(id=citation_id, label=row["header"], company=row["company"],
                    fiscal_year=int(row["fiscal_year"]), section=row["section"], url=url)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from tools.vector_tool import store     # load the index once, not on the first question
    store()
    chunk_table()
    yield


app = FastAPI(title="GraphRAG Enterprise Copilot",
              description="Agentic question answering over SEC 10-K reports: vector search, "
                          "knowledge graph (Cypher) and financial data (SQL).",
              version="1.1", lifespan=lifespan,
              docs_url="/docs" if ENABLE_DOCS else None, redoc_url=None,
              openapi_url="/openapi.json" if ENABLE_DOCS else None)
app.add_middleware(SecurityHeaders)
if ALLOWED_ORIGINS:
    app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type", "X-API-Key"])
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.post("/ask", response_model=AskResponse,
          dependencies=[Depends(require_api_key), Depends(limit_questions)])
def ask(req: AskRequest) -> AskResponse:
    from agent.graph import ask as run_agent
    t0 = time.time()
    if not _model_lock.acquire(timeout=BUSY_WAIT_SECONDS):
        raise HTTPException(status_code=429, headers={"Retry-After": "30"},
                            detail="The model is busy with another question. Try again shortly")
    try:
        state = run_agent(req.question)
    except Exception:
        ref = uuid.uuid4().hex[:8]
        log.exception("agent failed (ref %s)", ref)
        raise HTTPException(status_code=503,
                            detail=f"The question could not be answered right now (reference {ref})")
    finally:
        _model_lock.release()
    details = None
    if req.include_details:
        details = {"sub_questions": state.get("sub_questions", {}),
                   "queries": {r.tool: r.query for r in state["results"]},
                   "rows_or_passages": {r.tool: len(r.rows) or len(r.passages) for r in state["results"]}}
    return AskResponse(question=req.question, answer=state["answer"],
                       citations=[resolve(c) for c in state.get("citations", [])],
                       unverified_citations=state.get("removed_citations", []),
                       route=state["tools"], timings=state.get("timings", {}),
                       seconds=round(time.time() - t0, 2), details=details)


@app.get("/sources/{chunk_id}", dependencies=[Depends(require_api_key)])
def source(chunk_id: str = PathParam(..., pattern=CHUNK_ID, max_length=24)) -> dict:
    chunks = chunk_table()
    if chunk_id not in chunks.index:
        raise HTTPException(status_code=404, detail="No passage with this id")
    return {"id": chunk_id, **resolve(chunk_id).model_dump(exclude={"id"}),
            "text": chunks.loc[chunk_id]["text"]}


@app.get("/health")
def health() -> dict:
    """Status of each dependency. Deliberately reports states only, never error details."""
    checks = {}
    try:
        tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).json()
        models = [m["name"] for m in tags.get("models", [])]
        checks["ollama"] = "ok" if any(m.startswith(LLM_MODEL) for m in models) else "model missing"
    except Exception:
        checks["ollama"] = "unreachable"
    try:
        from core.neo4j_db import driver
        with driver() as db:
            db.verify_connectivity()
        checks["neo4j"] = "ok"
    except BaseException:                    # driver() exits when the password is unset
        checks["neo4j"] = "unreachable"
    try:
        con = sqlite3.connect(f"file:{SQL_DB}?mode=ro", uri=True)
        n = con.execute("SELECT COUNT(*) FROM annual_financials").fetchone()[0]
        con.close()
        checks["sql"] = f"ok ({n} company-years)"
    except Exception:
        checks["sql"] = "unavailable"
    try:
        from tools.vector_tool import store
        checks["vector_index"] = f"ok ({store().index.ntotal:,} passages)"
    except Exception:
        checks["vector_index"] = "unavailable"
    status = "ok" if all(v.startswith("ok") for v in checks.values()) else "degraded"
    return {"status": status, "checks": checks}


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/favicon.svg", include_in_schema=False)
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(STATIC / "favicon.svg", media_type="image/svg+xml")
