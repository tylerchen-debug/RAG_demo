# Demo 2 — End-to-End `/knowledge/ask`

Demo 1 stopped at retrieval. This demo closes the loop: a real HTTP endpoint
that takes a user question and returns an **answer plus its sources** — but only
when the question is both *in scope* and *actually covered* by the knowledge
base.

The point students should take away: a production RAG assistant is not "call
retrieval, then call the LLM." It is a **controlled pipeline with guards**, and
most of the engineering value lives in the guards.

## The request pipeline

```
POST /knowledge/ask   {"question": "..."}
        │
        │  ┌──────────────────────────────────────────────┐
        ├─►│ (1) SCOPE GUARD          scope.py            │
        │  │     keyword pre-filter, no DB / model at all  │
        │  └──────────────────────────────────────────────┘
        │        └─► out of scope ──► rejected_out_of_scope   (llm_called: false)
        │
        │  ┌──────────────────────────────────────────────┐
        ├─►│ (2) RETRIEVAL            retrieval.py        │
        │  │     embed question ─► pgvector top-K          │
        │  └──────────────────────────────────────────────┘
        │        └─► best score < MIN_SIMILARITY
        │                     ──► rejected_low_relevance     (llm_called: false)
        │
        │  ┌──────────────────────────────────────────────┐
        └─►│ (3) ANSWER               llm.py              │
           │     the ONLY paid LLM call, grounded in (2)   │
           └──────────────────────────────────────────────┘
                  └─► answered  { answer, sources[] }        (llm_called: true)
```

Two different rejections, two different reasons:

| Guard | Question it stops | Why it exists | Cost of a miss |
| --- | --- | --- | --- |
| **Scope guard** | "What's the weather in Boston?" | The question is *not our business* at all | Wasted embedding + LLM spend, assistant used as a free ChatGPT |
| **Retrieval gate** | "Book me a hotel in Tokyo" | Plausible request, but *nothing in the knowledge base covers it* | The LLM gets irrelevant context and hallucinates a confident wrong answer |

## Prerequisites

- Demo 1 already ingested the knowledge base (the `document_chunks` table must
  have rows). This demo **reads** that table; it never writes to it.
- The Demo 1 database container running on port 5433.

```bash
cd ../demo1_offline_ingestion && docker compose up -d
```

## Setup

```bash
cd demo2_knowledge_ask

pyenv virtualenv 3.11 rag-demo2      # or reuse the Demo 1 environment
pyenv local rag-demo2
pip install -r requirements.txt

cp .env.example .env
```

By default `LLM_PROVIDER=stub`, so **the demo runs with no API key**: stage (3)
quotes the retrieved passages instead of generating prose. The pipeline, the
guards and the `sources` payload are all identical. To show a real generated
answer, set in `.env`:

```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

> `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` must match what Demo 1 used at
> ingestion time. Different model = different vector space = meaningless scores.

## Run the API

```bash
uvicorn app:app --reload --port 8000
```

Sanity check — it reports how many chunks it can see and the active threshold:

```bash
curl -s localhost:8000/health | python3 -m json.tool
```

```json
{
  "status": "ok",
  "chunks_indexed": 73,
  "embedding_model": "BAAI/bge-small-en-v1.5",
  "llm_provider": "stub",
  "min_similarity": 0.72
}
```

Interactive docs (handy instead of Postman): <http://localhost:8000/docs>

## The four calls to make

Run them one at a time, or all at once with `./demo_requests.sh`.

### 1. A refund question → answered, with sources

```bash
curl -X POST localhost:8000/knowledge/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "Can I get a refund if the shirt arrived with a printing defect?"}'
```

```json
{
  "status": "answered",
  "answer": "...We'll provide a free replacement or a full refund if your order arrives with...",
  "sources": [
    { "ref": 1, "source": "refund_policy.md",
      "heading": "When you're entitled to a refund or reprint",
      "chunk_index": 3, "score": 0.864, "excerpt": "..." },
    { "ref": 2, "source": "refund_policy.md",
      "heading": "When we generally can't offer a refund",
      "chunk_index": 4, "score": 0.8136, "excerpt": "..." }
  ],
  "llm_called": true,
  "best_score": 0.864
}
```

**Key point:** every claim is traceable. `sources` is what makes the answer
auditable — a support agent can click through and verify.

> The `answer` text above is what `LLM_PROVIDER=openai` produces. With the
> default `stub` provider the passages are quoted verbatim instead, and
> `llm_called` is `false` — everything else is identical.

### 2. A shipping question → answered

```bash
curl -X POST localhost:8000/knowledge/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "How long does shipping usually take?"}'
```

Best score ≈ `0.787`, pulled from `shipping_policy.md`. Note it retrieves from a
*different* document than call 1 — nobody wrote routing rules, the vectors did it.

### 3. A weather question → scope rejection

```bash
curl -X POST localhost:8000/knowledge/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is the weather in Boston tomorrow?"}'
```

```json
{
  "status": "rejected_out_of_scope",
  "answer": "I'm the AI Pack Support Assistant and I can only help with orders, shipping, refunds, ... I can't help with weather questions.",
  "sources": [],
  "llm_called": false,
  "reason": "matched out-of-scope topic 'weather' via term 'weather'"
}
```

**Key point:** `sources` is empty and `best_score` is `null` — we never even
embedded the question. This rejection costs approximately nothing. This is how
you stop your support bot from becoming a free general-purpose chatbot.

### 4. A hotel-booking question → retrieval gate rejection

```bash
curl -X POST localhost:8000/knowledge/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "Can you book me a hotel in Tokyo for next week?"}'
```

```json
{
  "status": "rejected_low_relevance",
  "answer": "I couldn't find anything in our knowledge base that answers that...",
  "sources": [
    { "ref": 1, "source": "shipping_policy.md",
      "heading": "Questions about your shipment", "score": 0.563, "excerpt": "..." }
  ],
  "llm_called": false,
  "reason": "best similarity 0.563 < threshold 0.72",
  "best_score": 0.563
}
```

**Key point:** The scope guard let this
one through (nothing in it looks like "weather" or "stock tips"). Retrieval
happily returned chunks, because **vector search always returns its top K, no
matter how bad the matches are**. Look at the score: `0.563` versus `0.864` for
the refund question. Those chunks are about contacting support, not hotels.

If we had handed them to an LLM and asked "can you book a hotel?", a helpful
model would have tried to answer anyway. The gate is what turns "always
answers" into "answers or admits it doesn't know."

## Score separation (why the threshold works)

Measured on this knowledge base:

| Question | Best score | Outcome |
| --- | --- | --- |
| refund + printing defect | **0.864** | answered |
| how long does shipping take | **0.787** | answered |
| book a hotel in Tokyo | **0.563** | rejected by gate |

`MIN_SIMILARITY=0.72` sits in the empty band between the two clusters.

The threshold is **data- and model-specific** — it is not a universal constant.
To tune it for a different corpus, run a batch of known-good and known-bad
questions, look at where the two score clusters separate, and put the threshold
in the gap. Demo 1's `vector_search.py` is a convenient tool for this.

Try moving it in `.env` and re-running call 4:
- `MIN_SIMILARITY=0.4` → the hotel question now gets "answered" (watch the model
  try to help, from irrelevant context — a live hallucination demo).
- `MIN_SIMILARITY=0.9` → even the refund question gets refused (too strict; the
  assistant becomes useless).

## Key take away

- A RAG endpoint is a **pipeline with guards**, not a single LLM call.
- **Scope rejection** is pre-retrieval and nearly free — it protects budget and
  keeps the assistant on-brand.
- **The retrieval gate** is post-retrieval and protects *truthfulness* —
  vector search never says "no results," so you must decide what "too far" means.
- The LLM is the **last and most expensive** step, and it only ever sees
  context that already passed both guards.
- Returning `sources` makes answers auditable; `llm_called` makes cost visible.

## Files

| File | Purpose |
| --- | --- |
| `app.py` | FastAPI app — wires the three stages, defines `/knowledge/ask` |
| `scope.py` | **Guard 1** — pre-retrieval scope check |
| `retrieval.py` | Vector search + **Guard 2** (the retrieval gate) |
| `llm.py` | **Stage 3** — grounded answer generation (openai / stub) |
| `db.py` | Read-only connection to the Demo 1 database |
| `embeddings.py` | Same embedder as Demo 1 (must match!) |
| `config.py` | Settings from `.env` |
| `demo_requests.sh` | The four calls above, in demo order |
