"""Demo 2 -- End-to-end /knowledge/ask.

    uvicorn app:app --reload --port 8000

The request walks through three stages, and each one can stop it early:

    question
       |
       |  (1) SCOPE GUARD      scope.py     -- no DB, no embedding, no LLM
       |      out of scope? --> 200 {"status": "rejected_out_of_scope"}
       v
       |  (2) RETRIEVAL        retrieval.py -- embed + pgvector nearest neighbours
       |      best score < MIN_SIMILARITY? --> 200 {"status": "rejected_low_relevance"}
       v
       |  (3) ANSWER           llm.py       -- the ONLY paid LLM call
       v
    {"status": "answered", "answer": ..., "sources": [...]}

Every response reports `llm_called`, so students can watch how often the guards
save a model invocation.
"""
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

import config
import db
import llm
import retrieval
import scope
from embeddings import get_embedder

# Loaded once at startup: the model and the DB connection are expensive to build.
state = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["embedder"] = get_embedder()
    state["conn"] = db.connect()
    yield
    state["conn"].close()


app = FastAPI(title="Demo 2 -- Controlled RAG Assistant", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    k: Optional[int] = Field(None, ge=1, le=10)


class Source(BaseModel):
    ref: int
    source: str
    heading: str
    chunk_index: int
    score: float
    excerpt: str


class AskResponse(BaseModel):
    status: str
    answer: str
    sources: List[Source] = []
    llm_called: bool
    reason: Optional[str] = None
    best_score: Optional[float] = None


def _to_sources(passages) -> List[Source]:
    out = []
    for i, p in enumerate(passages, start=1):
        excerpt = " ".join(p.content.split())
        if len(excerpt) > 240:
            excerpt = excerpt[:240] + "..."
        out.append(
            Source(
                ref=i,
                source=p.source,
                heading=p.heading,
                chunk_index=p.chunk_index,
                score=round(p.score, 4),
                excerpt=excerpt,
            )
        )
    return out


@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks_indexed": db.chunk_count(state["conn"]),
        "embedding_model": state["embedder"].model_name,
        "llm_provider": config.LLM_PROVIDER,
        "min_similarity": config.MIN_SIMILARITY,
    }


@app.post("/knowledge/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    question = req.question.strip()
    k = req.k or config.TOP_K

    # (1) Scope guard -- cheapest possible rejection.
    decision = scope.check_scope(question)
    if not decision.in_scope:
        return AskResponse(
            status="rejected_out_of_scope",
            answer=(
                f"I'm the {config.ASSISTANT_NAME} and I can only help with "
                f"{config.BUSINESS_SCOPE}. I can't help with "
                f"{decision.matched_topic} questions."
            ),
            llm_called=False,
            reason=f"matched out-of-scope topic '{decision.matched_topic}' "
                   f"via term '{decision.matched_term}'",
        )

    # (2) Retrieval.
    passages = retrieval.retrieve(state["conn"], state["embedder"], question, k)
    best_score = round(passages[0].score, 4) if passages else None

    # (2b) Retrieval gate -- in scope, but the knowledge base has no answer.
    if not retrieval.passes_gate(passages):
        return AskResponse(
            status="rejected_low_relevance",
            answer=(
                "I couldn't find anything in our knowledge base that answers "
                "that. If this is about your order, please contact "
                "support@aipack.com and a human will help you."
            ),
            sources=_to_sources(passages),
            llm_called=False,
            reason=f"best similarity {best_score} < threshold {config.MIN_SIMILARITY}",
            best_score=best_score,
        )

    # (3) Grounded answer -- the only path that spends an LLM call.
    answer = llm.generate_answer(question, passages)
    return AskResponse(
        status="answered",
        answer=answer,
        sources=_to_sources(passages),
        llm_called=config.LLM_PROVIDER != "stub",
        best_score=best_score,
    )
