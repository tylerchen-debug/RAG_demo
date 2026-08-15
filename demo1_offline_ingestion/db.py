"""PostgreSQL + pgvector helpers."""
import psycopg2
from pgvector.psycopg2 import register_vector

import config


def connect():
    """Open a connection, ensure the vector extension exists, and register the
    pgvector adapter so numpy arrays / lists convert to `vector` automatically."""
    conn = psycopg2.connect(config.db_dsn())
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    register_vector(conn)
    return conn


def ensure_schema(conn, dim: int, recreate: bool = True):
    """Create the document_chunks table sized to the embedding dimension."""
    with conn.cursor() as cur:
        if recreate:
            cur.execute("DROP TABLE IF EXISTS document_chunks")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id          SERIAL PRIMARY KEY,
                source      TEXT    NOT NULL,
                heading     TEXT,
                chunk_index INT     NOT NULL,
                content     TEXT    NOT NULL,
                char_count  INT     NOT NULL,
                embedding   vector({dim}) NOT NULL
            )
            """
        )
    conn.commit()


def create_index(conn):
    """Approximate-nearest-neighbor index for cosine distance (HNSW)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
            ON document_chunks
            USING hnsw (embedding vector_cosine_ops)
            """
        )
    conn.commit()


def insert_chunks(conn, rows):
    """rows: iterable of (source, heading, chunk_index, content, char_count, embedding)."""
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO document_chunks
                (source, heading, chunk_index, content, char_count, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            rows,
        )
    conn.commit()
