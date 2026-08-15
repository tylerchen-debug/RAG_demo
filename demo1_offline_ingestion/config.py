"""Central configuration for Demo 1, loaded from environment / .env file."""
import os

from dotenv import load_dotenv

load_dotenv()

_HERE = os.path.dirname(os.path.abspath(__file__))

# --- PostgreSQL ---
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "rag_demo")
DB_USER = os.getenv("DB_USER", "rag")
DB_PASSWORD = os.getenv("DB_PASSWORD", "ragpassword")


def db_dsn() -> str:
    return (
        f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} "
        f"user={DB_USER} password={DB_PASSWORD}"
    )


# --- Embeddings ---
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "fastembed").lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "").strip()

# --- Knowledge base location ---
# Defaults to the sibling ../knowledge_base folder regardless of where you run from.
KNOWLEDGE_BASE_DIR = os.getenv(
    "KNOWLEDGE_BASE_DIR",
    os.path.abspath(os.path.join(_HERE, "..", "knowledge_base")),
)

# --- Chunking ---
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "800"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "150"))
