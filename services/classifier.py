"""
NyayaSathi AI — Document Classifier
=====================================
Classifies a legal document into one of the predefined categories
using a hybrid approach:
  1. **Keyword scoring** — fast first pass
  2. **Embedding similarity** — tie-breaker using cosine similarity
     between the document and category reference descriptions

This avoids an LLM call for classification, keeping it fast and offline.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np

from embeddings.embedding_model import EmbeddingModel
from utils.logger import get_logger
import config

logger = get_logger(__name__)

# ──────────────────────────── Keyword Banks ──────────────────────────

_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Rental Agreement": [
        "tenant", "landlord", "lease", "rent", "premises", "occupancy",
        "security deposit", "eviction", "subletting", "maintenance",
        "dwelling", "habitation", "lessor", "lessee", "monthly rent",
        "tenancy", "rental period", "move-in", "move-out",
    ],
    "Employment Contract": [
        "employee", "employer", "salary", "compensation", "probation",
        "termination", "notice period", "non-compete", "confidentiality",
        "benefits", "working hours", "leave", "designation", "appointment",
        "human resources", "payroll", "employment", "job title",
    ],
    "Small Business Contract": [
        "vendor", "supplier", "purchase order", "invoice", "delivery",
        "payment terms", "scope of work", "indemnification", "warranty",
        "service agreement", "contractor", "subcontractor", "milestone",
        "business", "partnership", "agreement between parties",
    ],
    "Legal Notice": [
        "notice", "hereby", "demand", "legal action", "court",
        "advocate", "respondent", "petitioner", "cease and desist",
        "violation", "breach", "summon", "tribunal", "jurisdiction",
        "complainant", "accused", "proceeding",
    ],
}

# Category descriptions used for embedding-based fallback
_CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "Rental Agreement": (
        "A rental or lease agreement between a landlord and tenant "
        "governing the terms of property occupancy, rent payment, "
        "security deposit, maintenance, and eviction conditions."
    ),
    "Employment Contract": (
        "An employment contract between an employer and employee "
        "specifying salary, designation, working hours, leave, "
        "probation, termination terms, and confidentiality obligations."
    ),
    "Small Business Contract": (
        "A business contract between vendors, suppliers, or service "
        "providers covering scope of work, payment terms, delivery, "
        "warranties, and indemnification clauses."
    ),
    "Legal Notice": (
        "A formal legal notice issued by an advocate or party demanding "
        "compliance, threatening legal proceedings, or notifying of "
        "breach, violation, or court summons."
    ),
}


def classify_document(
    text: str,
    embedding_model: Optional[EmbeddingModel] = None,
) -> str:
    """Return the most likely document category.

    Parameters
    ----------
    text : str
        Cleaned document text.
    embedding_model : EmbeddingModel | None
        If provided, used for embedding-based fallback when keyword
        scores are tied.

    Returns
    -------
    str
        One of the values in ``config.DOCUMENT_CATEGORIES``.
    """
    # Step 1: keyword scoring
    scores = _keyword_scores(text)
    logger.debug("Keyword scores: %s", scores)

    top_score = max(scores.values())
    top_cats = [cat for cat, s in scores.items() if s == top_score]

    if len(top_cats) == 1 and top_score > 0:
        result = top_cats[0]
        logger.info("Classified via keywords → %s (score=%d)", result, top_score)
        return result

    # Step 2: embedding similarity fallback
    logger.info("Keyword tie / zero scores — falling back to embeddings")
    model = embedding_model or EmbeddingModel.get_instance()
    result = _embedding_classify(text, model)
    logger.info("Classified via embeddings → %s", result)
    return result


# ──────────────────────────── Private Helpers ─────────────────────────


def _keyword_scores(text: str) -> dict[str, int]:
    """Count keyword hits per category (case-insensitive)."""
    text_lower = text.lower()
    scores: dict[str, int] = {}
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        scores[cat] = sum(
            len(re.findall(r"\b" + re.escape(kw) + r"\b", text_lower))
            for kw in keywords
        )
    return scores


def _embedding_classify(text: str, model: EmbeddingModel) -> str:
    """Classify by cosine similarity between document and category descriptions."""
    # Truncate document to first ~2000 chars to keep embedding representative
    snippet = text[:2000]
    doc_emb = model.encode_single(snippet)  # (D,)

    categories = list(_CATEGORY_DESCRIPTIONS.keys())
    desc_texts = [_CATEGORY_DESCRIPTIONS[c] for c in categories]
    desc_embs = model.encode(desc_texts)    # (C, D)

    # Cosine similarity
    doc_norm = doc_emb / (np.linalg.norm(doc_emb) + 1e-9)
    desc_norms = desc_embs / (np.linalg.norm(desc_embs, axis=1, keepdims=True) + 1e-9)
    sims = desc_norms @ doc_norm  # (C,)

    best_idx = int(np.argmax(sims))
    return categories[best_idx]
