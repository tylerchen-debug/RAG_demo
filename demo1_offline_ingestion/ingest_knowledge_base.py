"""Demo 1 -- Step 1: Offline ingestion.

Reads every markdown file in the knowledge base, splits each into chunks,
computes an embedding for every chunk, and stores everything in the
PostgreSQL `document_chunks` table.

    python ingest_knowledge_base.py

This is the "offline" (batch) half of RAG. It runs ahead of time, independent
of any user question. At query time we only search the vectors it produced --
we never re-read the raw documents.
"""
import argparse
import glob
import os

import numpy as np

import config
import db
from chunking import chunk_document
from embeddings import get_embedder


def load_documents(kb_dir: str):
    paths = sorted(glob.glob(os.path.join(kb_dir, "*.md")))
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            yield os.path.basename(path), f.read()


def main():
    parser = argparse.ArgumentParser(description="Ingest the knowledge base into pgvector.")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Keep existing rows instead of recreating the table (default: recreate).",
    )
    args = parser.parse_args()

    kb_dir = config.KNOWLEDGE_BASE_DIR
    print(f"Knowledge base : {kb_dir}")
    print(f"Provider       : {config.EMBEDDING_PROVIDER}")

    print("\nLoading embedding model (first run may download it)...")
    embedder = get_embedder()
    print(f"Embedding model: {embedder.model_name}  (dim={embedder.dim})")

    # 1) Read + chunk every document.
    all_chunks = []
    per_file = {}
    for source, text in load_documents(kb_dir):
        chunks = chunk_document(
            source, text, config.CHUNK_MAX_CHARS, config.CHUNK_OVERLAP_CHARS
        )
        per_file[source] = len(chunks)
        all_chunks.extend(chunks)

    if not all_chunks:
        print(f"\nNo markdown files found in {kb_dir}. Nothing to ingest.")
        return

    print("\nChunking summary:")
    for source, n in per_file.items():
        print(f"  {source:<28} -> {n:>3} chunks")
    print(f"  {'TOTAL':<28} -> {len(all_chunks):>3} chunks")

    # 2) Embed all chunks.
    print("\nEmbedding chunks...")
    vectors = embedder.encode([c.content for c in all_chunks])

    # 3) Store in PostgreSQL.
    print("Writing to PostgreSQL...")
    conn = db.connect()
    db.ensure_schema(conn, embedder.dim, recreate=not args.append)

    rows = [
        (
            c.source,
            c.heading,
            c.chunk_index,
            c.content,
            len(c.content),
            np.array(vec, dtype=np.float32),
        )
        for c, vec in zip(all_chunks, vectors)
    ]
    db.insert_chunks(conn, rows)
    db.create_index(conn)

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM document_chunks")
        total = cur.fetchone()[0]
    conn.close()

    print(f"\nDone. document_chunks now holds {total} rows.")
    print("Next: inspect the table, then run a vector search:")
    print('  python inspect_db.py')
    print('  python vector_search.py "how long does shipping take?"')


if __name__ == "__main__":
    main()
