#!/usr/bin/env python3
"""Parse OpenAI SSE stream from stdin, print human-readable output."""
import sys, json

for line in sys.stdin:
    line = line.strip()
    if not line.startswith("data: "):
        continue
    data = line[6:]
    if data == "[DONE]":
        print("\n✅ Research complete")
        break
    try:
        d = json.loads(data)
        delta = d["choices"][0].get("delta", {})
        rc = delta.get("reasoning_content", "")
        content = delta.get("content", "")
        if rc:
            print(f"🧠 {rc}", end="", flush=True)
        if content:
            print(content, end="", flush=True)
    except (json.JSONDecodeError, KeyError, IndexError):
        pass

print()
