"""Route the RAGAS judge's OpenAI-style requests to Ollama's native API with the output
schema ENFORCED.

RAGAS 0.4 talks to the judge through instructor and an OpenAI client. This instructor
version sends {"type": "json_object", "schema": ...}; Ollama's OpenAI-compatible endpoint
ignores the non-standard "schema" field and only enforces "some JSON", so the local
model sometimes answered off-schema (e.g. {"error": "..."}) and those judgements failed.
This httpx transport rewrites /chat/completions calls into native /api/chat calls with
format=<schema> (which Ollama does enforce), and converts the reply back. Everything else
(embeddings) passes through unchanged.
"""
import json

import httpx

DEFAULT_CTX = 16384


def to_text(content) -> str:
    """OpenAI messages may carry a list of parts; Ollama wants plain text."""
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


class OllamaNativeTransport(httpx.AsyncBaseTransport):
    def __init__(self, ollama_url: str, timeout: float = 900, post=None):
        self.ollama_url = ollama_url.rstrip("/")
        self.timeout = timeout
        self.passthrough = httpx.AsyncHTTPTransport()
        self._post = post                   # injectable for tests

    async def _native(self, payload: dict) -> dict:
        if self._post:
            return await self._post(payload)
        async with httpx.AsyncClient(timeout=self.timeout) as c:
            r = await c.post(f"{self.ollama_url}/api/chat", json=payload)
            r.raise_for_status()
            return r.json()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if not request.url.path.endswith("/chat/completions"):
            return await self.passthrough.handle_async_request(request)
        body = json.loads(request.content)
        rf = body.get("response_format") or {}
        schema = rf.get("schema") or (rf.get("json_schema") or {}).get("schema") or ("json" if rf else None)
        max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") or 4096
        data, content = None, ""
        for attempt in range(3):            # retry output cut off by runaway whitespace
            payload = {"model": body["model"], "stream": False, "think": False,
                       "messages": [{"role": m["role"], "content": to_text(m.get("content"))}
                                    for m in body["messages"]],
                       "options": {"temperature": body.get("temperature", 0) if attempt == 0 else 0.3,
                                   "num_predict": max_tokens * (attempt + 1), "num_ctx": DEFAULT_CTX}}
            if schema:
                payload["format"] = schema
            data = await self._native(payload)
            content = data["message"]["content"]
            if not schema:
                break
            try:
                json.loads(content)
                break
            except json.JSONDecodeError:
                continue
        prompt, completion = data.get("prompt_eval_count", 0), data.get("eval_count", 0)
        reply = {"id": "ollama-native", "object": "chat.completion", "created": 0,
                 "model": body["model"],
                 "choices": [{"index": 0, "finish_reason": "stop",
                              "message": {"role": "assistant", "content": content}}],
                 "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                           "total_tokens": prompt + completion}}
        return httpx.Response(200, json=reply, request=request)
