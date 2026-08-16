"""Answer generation -- the only place an LLM is ever called.

Two providers:
  - "openai" : a real chat completion, tightly constrained by the system prompt.
  - "stub"   : no network, no key; stitches the retrieved chunks together so the
               end-to-end flow can be demonstrated offline in a classroom.

Note that both providers receive the SAME retrieved context. The guards in
scope.py and retrieval.py decide *whether* we get here at all.
"""
from typing import List

from openai import OpenAI

import config
from retrieval import Passage

SYSTEM_PROMPT = """You are {name}, a support assistant for AI Pack LLC.

Rules you must follow:
1. Answer ONLY using the numbered context passages provided by the user message.
2. If the passages do not contain the answer, say you don't have that information
   and suggest contacting support@aipack.com. Never guess.
3. Cite the passages you used like [1], [2].
4. Never discuss topics outside {scope}.
5. Be concise and friendly. Plain prose, no marketing language.
"""


def _format_context(passages: List[Passage]) -> str:
    blocks = []
    for i, p in enumerate(passages, start=1):
        text = " ".join(p.content.split())
        blocks.append(f"[{i}] (source: {p.source} | section: {p.heading})\n{text}")
    return "\n\n".join(blocks)


def _answer_openai(question: str, passages: List[Passage]) -> str:
    client = OpenAI()
    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(
                    name=config.ASSISTANT_NAME, scope=config.BUSINESS_SCOPE
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context passages:\n\n{_format_context(passages)}\n\n"
                    f"Question: {question}"
                ),
            },
        ],
    )
    return resp.choices[0].message.content.strip()


def _answer_stub(question: str, passages: List[Passage]) -> str:
    """Offline stand-in: quotes the best passages instead of generating text."""
    best = passages[0]
    excerpt = " ".join(best.content.split())
    if len(excerpt) > 600:
        excerpt = excerpt[:600] + "..."
    cited = ", ".join(f"[{i}]" for i in range(1, len(passages) + 1))
    return (
        f"(LLM_PROVIDER=stub -- no model was called; showing retrieved context.)\n\n"
        f"Based on our {best.source} under \"{best.heading}\":\n\n{excerpt}\n\n"
        f"Sources used: {cited}"
    )


def generate_answer(question: str, passages: List[Passage]) -> str:
    provider = config.LLM_PROVIDER
    if provider == "openai":
        return _answer_openai(question, passages)
    if provider == "stub":
        return _answer_stub(question, passages)
    raise ValueError(f"Unknown LLM_PROVIDER={provider!r}. Use 'openai' or 'stub'.")
