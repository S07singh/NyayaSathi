"""
NyayaSathi AI — Text Cleaning Utilities
========================================
Functions for normalising raw text extracted from PDF / DOCX files.
"""

import re
import unicodedata


def clean_text(raw: str) -> str:
    """Master cleaning pipeline — composes all sub-cleaners.

    1. Fix Unicode artefacts (smart quotes, em-dashes, etc.)
    2. Remove page numbers / headers / footers
    3. Normalise whitespace
    """
    text = _fix_unicode(raw)
    text = remove_page_numbers(text)
    text = _normalise_whitespace(text)
    return text.strip()


def remove_page_numbers(text: str) -> str:
    """Remove standalone page number lines like 'Page 3', '- 12 -', '3 of 10'."""
    patterns = [
        r"(?m)^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$",         # bare numbers
        r"(?mi)^\s*page\s+\d{1,4}\s*(of\s+\d{1,4})?\s*$",  # Page N (of M)
    ]
    for pat in patterns:
        text = re.sub(pat, "", text)
    return text


# ──────────────────── Private helpers ────────────────────────────────

def _fix_unicode(text: str) -> str:
    """Normalise to NFKC and replace common non-ASCII punctuation."""
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "\u2018": "'", "\u2019": "'",   # smart single quotes
        "\u201c": '"', "\u201d": '"',   # smart double quotes
        "\u2013": "-", "\u2014": "-",   # en-dash / em-dash
        "\u2026": "...",                 # ellipsis
        "\u00a0": " ",                   # non-breaking space
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _normalise_whitespace(text: str) -> str:
    """Collapse runs of whitespace; preserve paragraph breaks."""
    # Collapse 3+ newlines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse runs of spaces/tabs within a line
    text = re.sub(r"[^\S\n]+", " ", text)
    return text
