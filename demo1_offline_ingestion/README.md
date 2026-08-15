# Demo 1 — Offline Ingestion + Vector Search

This demo shows **how documents enter a RAG knowledge base** and how retrieval
actually works. There is **no ChatGPT / LLM call anywhere in this demo**.
Students should walk away understanding that RAG starts
with an *embedding pipeline* and a *vector search*, not a chat completion.

This demo covers the "offline" half of RAG: how documents are chunked,
embedded, and written to the database, and how a query uses vector search to
retrieve the most relevant chunks — all without ever calling an LLM.

## The pipeline

```
knowledge_base/*.md
      │  (1) chunk        chunking.py
      ▼
   text chunks
      │  (2) embed        embeddings.py  (local fastembed model)
      ▼
   vectors + metadata
      │  (3) store        db.py  ->  PostgreSQL table: document_chunks
      ▼
 ┌─────────────────────────────────────────────┐
 │  query text ─embed─► vector ─►  ORDER BY      │   vector_search.py
 │  embedding <=> query  LIMIT k  ► top-K chunks │
 └─────────────────────────────────────────────┘
```

## Prerequisites

- Docker (for PostgreSQL + pgvector)
- Python 3.9+

## Setup

```bash
cd demo1_offline_ingestion

# 1) Start PostgreSQL with the pgvector extension
docker compose up -d

# 2) Create a virtual env and install dependencies
pyenv virtualenv 3.11 rag-demo1
pyenv local rag-demo1
pip install -r requirements.txt

# 3) Configure (optional — defaults already match docker-compose.yml)
cp .env.example .env
```

By default we use **fastembed** with the local model `BAAI/bge-small-en-v1.5`
(384-dim). It runs offline and needs no API key; the model downloads once on
first use. To use OpenAI instead, set `EMBEDDING_PROVIDER=openai` and
`OPENAI_API_KEY` in `.env`.

## Run the demo

### Step 1 — Ingest the knowledge base

```bash
python ingest_knowledge_base.py
```

You'll see how many chunks each document produced and a confirmation that the
rows landed in `document_chunks`. Re-running recreates the table from scratch
(use `--append` to keep existing rows).

### Step 2 — Look inside the `document_chunks` table

From Python:

```bash
python inspect_db.py
```

Or directly with `psql`:

```bash
docker compose exec db psql -U rag -d rag_demo

-- inside psql:
\d document_chunks
SELECT id, source, heading, chunk_index, char_count FROM document_chunks LIMIT 10;
SELECT source, COUNT(*) FROM document_chunks GROUP BY source;
```

Each row = one chunk of text + its embedding `vector(384)`.

### Step 3 & 4 — Run a vector search and see top-K chunks with scores

```bash
python vector_search.py "how long does shipping take?"
python vector_search.py "can I get a refund if I change my mind" --k 3
python vector_search.py "who owns the AI generated designs"
```

Example output:

```
Query: "how long does shipping take?"
Top 5 chunks by cosine similarity:

[1] score=0.8123  shipping_policy.md | Shipping options and timing (chunk #3)
    Once your item is produced, it ships via one of our carrier partners...
```

`score` is cosine similarity (`1 - cosine_distance`); higher = more relevant.

## What students should take away

- A document does **not** go into a RAG system whole — it is **chunked**.
- Every chunk is turned into a **vector (embedding)** by a model.
- Retrieval is **nearest-neighbor search** in vector space, done by the database.
- The LLM only enters *later* (a future demo), fed with these retrieved chunks.
  RAG ≠ "just calling ChatGPT."

## Teardown

```bash
docker compose down        # stop the database
docker compose down -v     # also delete the stored data
```

## Files

| File | Purpose |
| --- | --- |
| `docker-compose.yml` | PostgreSQL + pgvector database |
| `config.py` | Settings from `.env` |
| `chunking.py` | Markdown → chunks |
| `embeddings.py` | Embedding providers (fastembed / OpenAI) |
| `db.py` | Connection, schema, inserts, index |
| `ingest_knowledge_base.py` | **Step 1** — build the knowledge base |
| `inspect_db.py` | **Step 2** — peek at `document_chunks` |
| `vector_search.py` | **Steps 3–4** — query + top-K results |
