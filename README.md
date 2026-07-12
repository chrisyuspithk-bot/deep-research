# Deep Research API

OpenAI-compatible deep research endpoint. Drop it into **OpenWebUI** (or any OpenAI client) and get iterative, cited research reports powered by DDGS search and your LLM of choice.

## Features

- **Iterative search** — decomposes questions into targeted queries, searches, finds gaps, searches again
- **Fact-checking** — cross-references claims across sources with confidence scoring
- **Inline citations** — every claim tagged with `[n]` linking back to source URLs
- **Live progress** — streams `reasoning_content` for OpenWebUI's collapsible "Thinking" block
- **LLM-agnostic** — works with any OpenAI-compatible endpoint (Nemotron, vLLM, Ollama, etc.)

## Quick Start

```bash
# 1. Clone
git clone https://github.com/chrisyuspithk-bot/deep-research.git
cd deep-research

# 2. Install
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env → point LLM_BASE_URL at your LLM

# 4. Run
python server.py
# → http://localhost:8000
```

## OpenWebUI Setup

1. Admin Panel → Settings → Connections → OpenAI API
2. URL: `http://your-host:8000/v1`
3. Key: whatever you set in `LLM_API_KEY` (or `not-needed`)
4. Save — `deep-research` model will appear in the model picker

## API

| Endpoint | Description |
|---|---|
| `GET /health` | Liveness check |
| `GET /v1/models` | Model list (OpenAI format) |
| `POST /v1/chat/completions` | Deep research (streaming only) |

### Chat Completions

```bash
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deep-research",
    "messages": [{"role": "user", "content": "What is the current state of fusion energy?"}],
    "stream": true
  }'
```

## Pipeline

```
User Question
    │
    ▼
① Query Decomposition   → LLM generates 5 targeted search queries
    │
    ▼
② Round-1 Search        → Parallel DDGS searches, deduplicate
    │
    ▼
③ Content Extraction    → Fetch pages, trafilatura clean text
    │
    ▼
④ Gap Analysis          → LLM finds missing angles → follow-up queries
    │
    ▼
⑤ Round-2 Search        → New sources for gaps
    │
    ▼
⑥ Cross-Reference       → HIGH/MEDIUM/DISPUTED/LOW confidence scoring
    │
    ▼
⑦ Synthesis             → Stream final report with [n] citations
```

## Configuration

All via `.env`:

| Variable | Default | Description |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8080/v1` | Your OpenAI-compatible LLM endpoint |
| `LLM_API_KEY` | `not-needed` | API key for LLM |
| `LLM_MODEL` | `nvidia/nemotron-super-3-nano` | Model name passed to LLM |
| `SERVER_MODEL_NAME` | `deep-research` | Model name shown to clients |
| `N_INITIAL_QUERIES` | `5` | Search queries for round 1 |
| `N_FOLLOW_UP_QUERIES` | `3` | Follow-up queries after gap analysis |
| `MAX_CONCURRENT_REQUESTS` | `3` | Max simultaneous research tasks |
| `PORT` | `8000` | Server port |

## Requirements

- Python 3.11+
- An OpenAI-compatible LLM endpoint (local or remote)
- Internet access for DDGS search and page fetching
