#!/usr/bin/env bash
# Test the deep-research endpoint with a real LLM backend.
#
# Usage:
#   export DEEPSEEK_API_KEY="sk-..."
#   bash test_research.sh
#
# Or inline:
#   DEEPSEEK_API_KEY="sk-..." bash test_research.sh

set -euo pipefail

# Auto-detect python command (macOS uses python3)
PYTHON=$(command -v python3 2>/dev/null || command -v python 2>/dev/null || echo "python3")

API_KEY="${DEEPSEEK_API_KEY:-}"
if [ -z "$API_KEY" ]; then
    echo "❌ Set DEEPSEEK_API_KEY environment variable"
    exit 1
fi

PORT="${PORT:-8000}"
BASE="http://localhost:${PORT}"

# ── Kill any existing server on our port ──────────────────────────────
EXISTING_PID=$(lsof -ti "tcp:${PORT}" 2>/dev/null || true)
if [ -n "$EXISTING_PID" ]; then
    echo "🔪 Killing existing process on port ${PORT} (PID ${EXISTING_PID})"
    kill "$EXISTING_PID" 2>/dev/null || true
    sleep 1
fi

# ── Start server ──────────────────────────────────────────────────────
echo "🚀 Starting server on port ${PORT}..."
LLM_BASE_URL="https://api.deepseek.com/v1" \
LLM_API_KEY="$API_KEY" \
LLM_MODEL="deepseek-chat" \
SERVER_MODEL_NAME="deep-research" \
PORT="$PORT" \
    nohup "$PYTHON" server.py > /tmp/deep-research-server.log 2>&1 &

SERVER_PID=$!
echo "   PID: ${SERVER_PID}"

# Wait for readiness
for i in $(seq 1 20); do
    if curl -s "${BASE}/health" > /dev/null 2>&1; then
        echo "✅ Server ready"
        break
    fi
    if [ "$i" -eq 20 ]; then
        echo "❌ Server failed to start. Log:"
        cat /tmp/deep-research-server.log
        kill "$SERVER_PID" 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

# ── Test 1: Health ────────────────────────────────────────────────────
echo ""
echo "━━━ Test 1: Health ━━━"
curl -s "${BASE}/health" | "$PYTHON" -m json.tool

# ── Test 2: Models list ───────────────────────────────────────────────
echo ""
echo "━━━ Test 2: Models ━━━"
curl -s "${BASE}/v1/models" | "$PYTHON" -m json.tool

# ── Test 3: Non-streaming rejection ───────────────────────────────────
echo ""
echo "━━━ Test 3: Non-streaming rejection ━━━"
curl -s -X POST "${BASE}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"deep-research","messages":[{"role":"user","content":"test"}],"stream":false}' \
    | "$PYTHON" -m json.tool

# ── Test 4: Live research (streaming) ─────────────────────────────────
echo ""
echo "━━━ Test 4: Live research ━━━"
echo "   Question: What are the latest breakthroughs in quantum computing?"
echo "   ─────────────────────────────────────────────────"
echo ""

curl -s -N --max-time 120 -X POST "${BASE}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deep-research",
        "messages": [
            {"role": "user", "content": "What are the latest breakthroughs in quantum computing as of 2025-2026?"}
        ],
        "stream": true
    }' | while IFS= read -r line; do
    if [[ "$line" == data:* ]]; then
        data="${line#data: }"
        if [ "$data" = "[DONE]" ]; then
            echo ""
            echo "   ─────────────────────────────────────────────────"
            echo "   ✅ Research complete"
            continue
        fi
        # Extract reasoning or content
        reasoning=$(echo "$data" | "$PYTHON" -c "
import sys, json
try:
    d = json.load(sys.stdin)
    delta = d['choices'][0].get('delta', {})
    rc = delta.get('reasoning_content', '')
    c = delta.get('content', '')
    if rc: print(f'🧠 {rc}', end='')
    if c: print(c, end='')
except: pass
" 2>/dev/null || true)
    fi
done

echo ""

# ── Cleanup ───────────────────────────────────────────────────────────
echo "🧹 Stopping server (PID ${SERVER_PID})..."
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
echo "✅ Done"
