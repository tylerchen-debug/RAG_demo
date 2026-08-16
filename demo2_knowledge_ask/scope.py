"""Guard #1 -- Scope check (runs BEFORE retrieval).

The cheapest way to stop abuse is to never start the pipeline. This module
rejects questions that are obviously outside the business domain (weather,
stock tips, medical advice, "write me some code", ...) using a deterministic
keyword check -- no database hit, no embedding, no LLM call, no cost.

Deliberately simple and readable so students can see *exactly* why a question
was rejected. Production systems usually upgrade this to a small classifier,
but the architectural role stays identical: a fast, cheap pre-filter.
"""
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

# Topic -> trigger words. If a question matches one of these and shows no sign
# of being about our business, it is refused up front.
OUT_OF_SCOPE_TOPICS: Dict[str, List[str]] = {
    "weather": ["weather", "temperature", "forecast", "raining", "snow", "humidity"],
    "financial advice": ["stock", "stocks", "crypto", "bitcoin", "invest", "portfolio"],
    "medical advice": ["symptom", "diagnose", "medication", "doctor", "illness"],
    "legal advice": ["sue", "lawsuit", "legal advice", "attorney"],
    "politics": ["election", "president", "political party", "vote for"],
    "general coding help": ["write code", "python script", "sql query", "debug my"],
    "general knowledge": ["capital of", "who won", "translate this", "write a poem"],
}

# Business vocabulary. Its presence means the question is plausibly ours, so we
# let it through to retrieval even if it also brushes a keyword above.
IN_SCOPE_HINTS: List[str] = [
    "order", "orders", "shipping", "ship", "delivery", "deliver", "tracking",
    "refund", "return", "cancel", "exchange", "reprint",
    "product", "size", "sizing", "fabric", "print", "printing", "quality",
    "design", "artwork", "upload", "resolution", "file",
    "ai", "generated", "copyright", "ownership", "license",
    "payment", "invoice", "discount", "aipack", "ai pack",
]


@dataclass
class ScopeDecision:
    in_scope: bool
    matched_topic: Optional[str] = None
    matched_term: Optional[str] = None


def _contains_term(text: str, term: str) -> bool:
    """Word-boundary match so "invest" doesn't fire on "investigation"."""
    return re.search(rf"\b{re.escape(term)}\b", text) is not None


def check_scope(question: str) -> ScopeDecision:
    text = question.lower()

    if any(_contains_term(text, hint) for hint in IN_SCOPE_HINTS):
        return ScopeDecision(in_scope=True)

    for topic, terms in OUT_OF_SCOPE_TOPICS.items():
        for term in terms:
            if _contains_term(text, term):
                return ScopeDecision(in_scope=False, matched_topic=topic, matched_term=term)

    # Unknown territory: let retrieval decide (Guard #2 catches it).
    return ScopeDecision(in_scope=True)
