"""
NyayaSathi AI — Clause Segmenter
==================================
Splits a legal document into individual clauses by detecting:
  • Numbered / lettered headings  (1., 2.1, (a), A., etc.)
  • Section headers               (ARTICLE I, SECTION 2, SCHEDULE A)
  • Paragraph breaks when no numbering is found

Each clause is returned as a dict with ``id``, ``title``, and ``text``
so that downstream components (risk analyser, simplifier) can work on
individual units.
"""

from __future__ import annotations

import re
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# ──────────────────────────── Patterns ────────────────────────────────

# Matches lines that look like clause headers:
#   "1.", "1.1", "1.1.1", "(a)", "(i)", "A.", "ARTICLE 5", "SECTION 3", "SCHEDULE B"
_CLAUSE_HEADER_RE = re.compile(
    r"^"
    r"(?:"
    r"(?P<numbered>\d+(?:\.\d+)*)\.\s"           # 1. / 2.1. / 3.1.2.
    r"|(?P<lettered>\([a-zA-Z0-9]+\))\s"          # (a) / (ii)
    r"|(?P<alpha>[A-Z])\.\s"                       # A. / B.
    r"|(?P<section>(?:ARTICLE|SECTION|SCHEDULE|CLAUSE|PART)"
    r"\s+[IVXLCDM\d]+[A-Z]?)\b"                   # ARTICLE IV, SECTION 3
    r")",
    re.MULTILINE,
)


def segment_clauses(text: str) -> list[dict]:
    """Split document text into structured clauses.

    Returns
    -------
    list[dict]
        Each dict has keys:
        - ``id``    : int (1-indexed)
        - ``title`` : str | None (the header line, if detected)
        - ``text``  : str (clause body)
    """
    # Try numbered/header-based splitting first
    clauses = _split_by_headers(text)

    # Fallback: paragraph-based splitting if very few clauses detected
    if len(clauses) < 3:
        logger.info("Header-based split found only %d clause(s) — falling back to paragraph split", len(clauses))
        clauses = _split_by_paragraphs(text)

    logger.info("Segmented document into %d clause(s)", len(clauses))
    return clauses


# ──────────────────────────── Strategies ──────────────────────────────


def _split_by_headers(text: str) -> list[dict]:
    """Split on recognised clause header patterns."""
    matches = list(_CLAUSE_HEADER_RE.finditer(text))

    if not matches:
        return []

    clauses: list[dict] = []

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()

        # First line is the title; rest is the body
        title, body = _split_title_body(chunk)

        clauses.append({
            "id": i + 1,
            "title": title,
            "text": body if body else chunk,
        })

    # If there's text before the first header, capture it as a preamble
    preamble = text[: matches[0].start()].strip()
    if preamble and len(preamble) > 50:
        clauses.insert(0, {"id": 0, "title": "Preamble", "text": preamble})
        # Re-number
        for idx, cl in enumerate(clauses):
            cl["id"] = idx + 1

    return clauses


def _split_by_paragraphs(text: str) -> list[dict]:
    """Fallback: split on double-newlines (paragraph boundaries)."""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    # Merge very short paragraphs (< 60 chars) with the next one
    merged: list[str] = []
    buffer = ""
    for para in paragraphs:
        if buffer:
            buffer += "\n\n" + para
            if len(buffer) >= 60:
                merged.append(buffer)
                buffer = ""
        elif len(para) < 60:
            buffer = para
        else:
            merged.append(para)
    if buffer:
        merged.append(buffer)

    clauses = []
    for i, para in enumerate(merged, start=1):
        title, body = _split_title_body(para)
        clauses.append({
            "id": i,
            "title": title,
            "text": body if body else para,
        })
    return clauses


# ──────────────────────────── Helpers ─────────────────────────────────


def _split_title_body(chunk: str) -> tuple[Optional[str], str]:
    """Split text into an optional title (first line) and body."""
    lines = chunk.split("\n", 1)
    if len(lines) == 2:
        title = lines[0].strip()
        body = lines[1].strip()
        return title, body
    return None, chunk
