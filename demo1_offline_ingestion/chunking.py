"""Markdown-aware chunking.

A RAG knowledge base is not stored as whole documents -- it is split into small,
self-contained "chunks" that each get their own embedding. This module turns a
markdown file into a list of chunks while keeping the nearest heading as context.

Strategy (kept deliberately simple so it is easy to teach):
  1. Split the document into sections at markdown headings (#, ##, ...).
  2. Within each section, greedily pack paragraphs into chunks up to
     CHUNK_MAX_CHARS characters.
  3. Carry the last paragraph of a chunk into the next one to create overlap,
     so a sentence split across a boundary is still retrievable from both sides.
"""
import re
from dataclasses import dataclass
from typing import List, Tuple

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    source: str
    heading: str
    chunk_index: int
    content: str


def _split_sections(text: str) -> List[Tuple[str, str]]:
    """Return [(heading, body), ...]. Heading is the nearest preceding # line."""
    sections: List[Tuple[str, str]] = []
    current_heading = ""
    buffer: List[str] = []

    for line in text.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            if buffer:
                sections.append((current_heading, "\n".join(buffer).strip()))
                buffer = []
            current_heading = m.group(2).strip()
        else:
            buffer.append(line)

    if buffer:
        sections.append((current_heading, "\n".join(buffer).strip()))

    return [(h, b) for h, b in sections if b]


def _hard_split(paragraph: str, max_chars: int, overlap: int) -> List[str]:
    """Sliding-window split for a single oversized paragraph."""
    text = re.sub(r"\s+", " ", paragraph).strip()
    if len(text) <= max_chars:
        return [text] if text else []

    pieces: List[str] = []
    start = 0
    step = max(1, max_chars - overlap)
    while start < len(text):
        pieces.append(text[start : start + max_chars].strip())
        start += step
    return pieces


def _pack_paragraphs(paragraphs: List[str], max_chars: int, overlap: int) -> List[str]:
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for para in paragraphs:
        if len(para) > max_chars:
            if current:
                chunks.append("\n\n".join(current))
                current, current_len = [], 0
            chunks.extend(_hard_split(para, max_chars, overlap))
            continue

        if current_len and current_len + len(para) + 2 > max_chars:
            chunks.append("\n\n".join(current))
            # Start the next chunk with the previous paragraph for overlap.
            if overlap and current:
                tail = current[-1]
                current, current_len = [tail], len(tail)
            else:
                current, current_len = [], 0

        current.append(para)
        current_len += len(para) + 2

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def chunk_document(source: str, text: str, max_chars: int, overlap: int) -> List[Chunk]:
    chunks: List[Chunk] = []
    index = 0
    for heading, body in _split_sections(text):
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        for content in _pack_paragraphs(paragraphs, max_chars, overlap):
            chunks.append(Chunk(source, heading, index, content))
            index += 1
    return chunks
