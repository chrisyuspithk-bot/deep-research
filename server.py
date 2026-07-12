"""
OpenAI-compatible Deep Research API Server.

Endpoints:
  GET  /health                — liveness check
  GET  /v1/models             — model list
  POST /v1/chat/completions   — deep research (streaming + reasoning_content)

OpenWebUI integration:
  - reasoning_content delta  → collapsible "Thinking" block (live progress)
  - content delta            → final research report
  - sources appended         → rendered at bottom of response

Usage:
  uv run uvicorn server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from engine import DeepResearchEngine, LLM, create_engine

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("server")

# ── Configuration ───────────────────────────────────────────────────────────

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8080/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "not-needed")
LLM_MODEL = os.getenv("LLM_MODEL", "nvidia/nemotron-super-3-nano")
SERVER_MODEL_NAME = os.getenv("SERVER_MODEL_NAME", "deep-research")
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT_REQUESTS", "3"))
N_INITIAL_QUERIES = int(os.getenv("N_INITIAL_QUERIES", "5"))
N_FOLLOW_UP_QUERIES = int(os.getenv("N_FOLLOW_UP_QUERIES", "3"))

# ── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Deep Research API",
    version="1.0.0",
    description="OpenAI-compatible deep research endpoint with iterative search, fact-checking, and citations.",
)

# Semaphore to limit concurrent research tasks
import asyncio

_semaphore = asyncio.Semaphore(MAX_CONCURRENT)

# Lazy-initialized engine
_engine: DeepResearchEngine | None = None


def get_engine() -> DeepResearchEngine:
    global _engine
    if _engine is None:
        _engine = create_engine(
            llm_base_url=LLM_BASE_URL,
            llm_api_key=LLM_API_KEY,
            llm_model=LLM_MODEL,
        )
    return _engine


# ── Endpoints ───────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok", "model": LLM_MODEL}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": SERVER_MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "deep-research",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages: list[dict] = body.get("messages", [])
    stream: bool = body.get("stream", False)

    # Extract the last user message as the research question
    user_messages = [m for m in messages if m.get("role") == "user"]
    if not user_messages:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "No user message found", "type": "invalid_request"}},
        )
    question = user_messages[-1]["content"]

    if isinstance(question, list):
        # Multimodal — extract text parts
        question = " ".join(p.get("text", "") for p in question if p.get("type") == "text")

    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    logger.info("[%s] Research question: %s", request_id, question[:100])

    if not stream:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "Deep research only supports streaming (stream=true). "
                    "Research takes time — streaming provides live progress.",
                    "type": "invalid_request",
                }
            },
        )

    engine = get_engine()

    async def event_stream():
        created = int(time.time())

        async with _semaphore:
            try:
                async for delta in engine.research_and_stream(
                    question,
                    n_initial_queries=N_INITIAL_QUERIES,
                    n_follow_up_queries=N_FOLLOW_UP_QUERIES,
                ):
                    choice = {
                        "index": 0,
                        "delta": delta,
                    }
                    # Check for finish
                    is_final = "content" in delta and "[DONE]" in delta.get("content", "")
                    if is_final:
                        choice["finish_reason"] = "stop"

                    chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": SERVER_MODEL_NAME,
                        "choices": [choice],
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

                # Final chunk with finish_reason
                final = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": SERVER_MODEL_NAME,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(final)}\n\n"
                yield "data: [DONE]\n\n"

            except Exception as exc:
                logger.exception("[%s] Research failed", request_id)
                error_chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": SERVER_MODEL_NAME,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": f"\n\n❌ Research failed: {exc}\n\nPlease try again with a different query."
                            },
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
                yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
        log_level="info",
    )
