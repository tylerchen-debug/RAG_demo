"""Retrieval + Guard #2 (the retrieval gate).

Retrieval is the same nearest-neighbour search as Demo 1. What is new here is
the *gate*: if even the best-matching chunk is too far away, the knowledge base
simply does not cover the question, so we refuse instead of handing the LLM
irrelevant context and inviting a hallucination.
"""
from dataclasses import dataclass
from typing import List

import numpy as np

import config


@dataclass
class Passage:
    source: str
    heading: str
    chunk_index: int
    content: str
    score: float


def retrieve(conn, embedder, question: str, k: int) -> List[Passage]:
    query_vec = np.array(embedder.encode([question])[0], dtype=np.float32)
    with conn.cursor() as cur:
        # `<=>` is pgvector's cosine-distance operator; similarity = 1 - distance.
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
        rows = cur.fetchall()
    return [Passage(s, h, i, c, float(score)) for s, h, i, c, score in rows]


def passes_gate(passages: List[Passage]) -> bool:
    """True when the best passage is close enough to be worth answering from."""
    return bool(passages) and passages[0].score >= config.MIN_SIMILARITY
