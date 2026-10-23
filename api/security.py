"""Security controls for the API, configured with environment variables.

  API_KEY                 if set, /ask and /sources require header X-API-Key: <key>
  ASK_RATE_PER_MINUTE     questions per client per minute (default 6)
  MAX_BODY_BYTES          largest accepted request body (default 10 kB)
  ALLOWED_ORIGINS         comma-separated origins allowed to call the API from a browser
                          (default: none, so only the page served by this API can call it)
  TRUST_PROXY             "1" behind a reverse proxy (e.g. Azure): read the client address
                          from X-Forwarded-For
  ENABLE_DOCS             "0" hides /docs and /openapi.json
  BUSY_WAIT_SECONDS       how long a question waits for the model before "busy" (default 30)
"""
import hmac
import os
import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

API_KEY = os.environ.get("API_KEY", "").strip()
ASK_RATE_PER_MINUTE = int(os.environ.get("ASK_RATE_PER_MINUTE", "6"))
MAX_BODY_BYTES = int(os.environ.get("MAX_BODY_BYTES", "10240"))
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
TRUST_PROXY = os.environ.get("TRUST_PROXY", "0") == "1"
ENABLE_DOCS = os.environ.get("ENABLE_DOCS", "1") == "1"
BUSY_WAIT_SECONDS = float(os.environ.get("BUSY_WAIT_SECONDS", "30"))

# Scripts and styles only from this server (no inline code), fonts from Google Fonts,
# no framing, no plugins, requests only back to this server.
CSP = ("default-src 'self'; script-src 'self'; style-src 'self' https://fonts.googleapis.com; "
       "font-src https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; "
       "object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'")
DOCS_PATHS = ("/docs", "/redoc", "/openapi.json")   # FastAPI's docs page loads its own scripts


def client_ip(request: Request) -> str:
    if TRUST_PROXY:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class SecurityHeaders(BaseHTTPMiddleware):
    """Security headers on every response, and a request-size limit before the body is read."""

    async def dispatch(self, request: Request, call_next):
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > MAX_BODY_BYTES:
            return JSONResponse({"detail": "Request too large"}, status_code=413)
        response = await call_next(request)
        h = response.headers
        h["X-Content-Type-Options"] = "nosniff"
        h["X-Frame-Options"] = "DENY"
        h["Referrer-Policy"] = "strict-origin-when-cross-origin"
        h["Permissions-Policy"] = "camera=(), microphone=(), geolocation=(), payment=()"
        h["Cross-Origin-Opener-Policy"] = "same-origin"
        if not request.url.path.startswith(DOCS_PATHS):
            h["Content-Security-Policy"] = CSP
        if request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https":
            h["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


class RateLimiter:
    """Sliding-window limit per client address, kept in memory (one server process)."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.hits: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def check(self, key: str):
        now = time.monotonic()
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                retry = int(60 - (now - q[0])) + 1
                raise HTTPException(status_code=429, headers={"Retry-After": str(retry)},
                                    detail=f"Too many questions. Try again in {retry} seconds")
            q.append(now)
            if len(self.hits) > 10_000:                 # keep memory bounded
                for k in [k for k, v in self.hits.items() if not v]:
                    del self.hits[k]


ask_limiter = RateLimiter(ASK_RATE_PER_MINUTE)


def require_api_key(request: Request):
    """If API_KEY is configured, require it (constant-time comparison)."""
    if not API_KEY:
        return
    supplied = request.headers.get("x-api-key", "")
    if not hmac.compare_digest(supplied.encode(), API_KEY.encode()):
        raise HTTPException(status_code=401, detail="A valid access key is required",
                            headers={"WWW-Authenticate": "API-Key"})


def limit_questions(request: Request):
    ask_limiter.check(client_ip(request))
