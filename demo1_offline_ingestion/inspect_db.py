"""Demo 1 -- Step 2: Inspect the document_chunks table.

A quick way to "look inside" the knowledge base from Python instead of psql.

    python inspect_db.py
"""
import db


def main():
    conn = db.connect()
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM document_chunks")
        total = cur.fetchone()[0]

        cur.execute("SELECT source, COUNT(*) FROM document_chunks GROUP BY source ORDER BY source")
        by_source = cur.fetchall()

        cur.execute(
            """
            SELECT id, source, heading, chunk_index, char_count,
                   vector_dims(embedding) AS dims
            FROM document_chunks
            ORDER BY id
            LIMIT 5
            """
        )
        sample = cur.fetchall()

        cur.execute("SELECT content, embedding FROM document_chunks ORDER BY id LIMIT 1")
        first = cur.fetchone()
    conn.close()

    print(f"document_chunks total rows: {total}\n")

    print("Rows per source document:")
    for source, n in by_source:
        print(f"  {source:<28} {n:>3}")

    print("\nFirst 5 rows (metadata only):")
    print(f"  {'id':>3}  {'source':<28} {'chunk':>5} {'chars':>5} {'dims':>5}  heading")
    for row_id, source, heading, idx, chars, dims in sample:
        print(f"  {row_id:>3}  {source:<28} {idx:>5} {chars:>5} {dims:>5}  {heading}")

    if first:
        content, embedding = first
        preview = " ".join(content.split())[:160]
        # pgvector may return a Vector wrapper (.to_list) or a numpy array (iterable).
        values = embedding.to_list() if hasattr(embedding, "to_list") else list(embedding)
        vec_preview = ", ".join(f"{v:.4f}" for v in values[:6])
        print("\nExample chunk #0:")
        print(f"  content : {preview}...")
        print(f"  embedding (first 6 dims): [{vec_preview}, ...]")


if __name__ == "__main__":
    main()
