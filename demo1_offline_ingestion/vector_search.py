"""Demo 1 -- Step 3 & 4: Vector search.

Embeds a natural-language query with the SAME model used at ingestion time,
then asks PostgreSQL for the nearest chunks by cosine distance and prints the
top-K results with their similarity scores.

    python vector_search.py "how do I get a refund?"
    python vector_search.py "who owns AI generated designs" --k 3

The key idea: retrieval is just nearest-neighbor search in embedding space.
No LLM is involved here -- this is the "R" (retrieval) in RAG.
"""
import argparse

import numpy as np

import db
from embeddings import get_embedder


def search(conn, embedder, query: str, k: int):
    query_vec = np.array(embedder.encode([query])[0], dtype=np.float32)
    with conn.cursor() as cur:
        # `<=>` is pgvector's cosine-distance operator (0 = identical).
        # cosine_similarity = 1 - cosine_distance, so higher is better.
        cur.execute(
            """
            SELECT source, heading, chunk_index, content,
                   1 - (embedding <=> %s) AS cosine_similarity
            FROM document_chunks
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (query_vec, query_vec, k),
        )
        return cur.fetchall()


def main():
    parser = argparse.ArgumentParser(description="Vector search over document_chunks.")
    parser.add_argument("query", help="Natural-language query.")
    parser.add_argument("--k", type=int, default=5, help="Number of results (default 5).")
    args = parser.parse_args()

    embedder = get_embedder()
    conn = db.connect()
    results = search(conn, embedder, args.query, args.k)
    conn.close()

    print(f'\nQuery: "{args.query}"')
    print(f"Model: {embedder.model_name} (dim={embedder.dim})")
    print(f"Top {len(results)} chunks by cosine similarity:\n")

    for rank, (source, heading, idx, content, score) in enumerate(results, start=1):
        snippet = " ".join(content.split())
        if len(snippet) > 220:
            snippet = snippet[:220] + "..."
        print(f"[{rank}] score={score:.4f}  {source} | {heading} (chunk #{idx})")
        print(f"    {snippet}\n")


if __name__ == "__main__":
    main()
