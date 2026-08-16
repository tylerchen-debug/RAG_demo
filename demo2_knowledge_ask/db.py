"""PostgreSQL + pgvector helpers (read-only for this demo)."""
import psycopg2
from pgvector.psycopg2 import register_vector

import config


def connect():
    """Open a connection and register the pgvector adapter."""
    conn = psycopg2.connect(config.db_dsn())
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    register_vector(conn)
    return conn


def chunk_count(conn) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM document_chunks")
        return cur.fetchone()[0]
