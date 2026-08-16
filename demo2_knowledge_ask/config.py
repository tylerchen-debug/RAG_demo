"""Central configuration for Demo 2, loaded from environment / .env file."""
import os

from dotenv import load_dotenv

load_dotenv()

# --- PostgreSQL (same database Demo 1 ingested into) ---
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5433")
DB_NAME = os.getenv("DB_NAME", "rag_demo")
DB_USER = os.getenv("DB_USER", "rag")
DB_PASSWORD = os.getenv("DB_PASSWORD", "ragpassword")


def db_dsn() -> str:
    return (
        f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} "
        f"user={DB_USER} password={DB_PASSWORD}"
    )


# --- Embeddings (MUST match what Demo 1 used at ingestion time) ---
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "fastembed").lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "").strip()

# --- Retrieval ---
TOP_K = int(os.getenv("TOP_K", "4"))

# Retrieval gate: if the best chunk scores below this, we refuse *without*
# calling the LLM. Tune it by watching real scores (see README).
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.72"))

# --- Answer generation ---
# "openai" = real LLM call; "stub" = offline templated answer (no API key needed).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "stub").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# --- Assistant identity (used in prompts and refusal messages) ---
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "AI Pack Support Assistant")
BUSINESS_SCOPE = os.getenv(
    "BUSINESS_SCOPE",
    "orders, shipping, refunds, product FAQ, design guidelines, "
    "and the AI-generated artwork policy for AI Pack LLC",
)
