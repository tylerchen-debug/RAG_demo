#!/usr/bin/env bash
# Demo 2 -- the four calls to run in class, in order.
# Usage:  ./demo_requests.sh          (server must already be running)
set -u
BASE="${BASE:-http://localhost:8000}"

ask() {
  echo
  echo "=============================================================="
  echo "ASK: $1"
  echo "=============================================================="
  curl -s -X POST "$BASE/knowledge/ask" \
    -H 'Content-Type: application/json' \
    -d "{\"question\": \"$1\"}" | python3 -m json.tool
}

echo "### Health check"
curl -s "$BASE/health" | python3 -m json.tool

# 1) In scope + well covered -> answered with sources.
ask "Can I get a refund if the shirt arrived with a printing defect?"

# 2) In scope + well covered -> answered with sources.
ask "How long does shipping usually take?"

# 3) Out of scope -> rejected before any embedding or LLM call.
ask "What is the weather in Boston tomorrow?"

# 4) Plausible-sounding but not in the knowledge base -> retrieval gate rejects.
ask "Can you book me a hotel in Tokyo for next week?"
