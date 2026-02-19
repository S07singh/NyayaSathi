"""
NyayaSathi AI — Risk Analyzer
===============================
Analyses each clause for potential risks using an LLM, then computes a
weighted composite risk score.

Risk Dimensions (weights from config.RISK_WEIGHTS):
  • penalty_severity       — 30 %
  • clause_imbalance       — 25 %
  • termination_unfairness — 20 %
  • liability_excess       — 15 %
  • missing_rights         — 10 %

Each dimension is scored 0-100 by the LLM.  The overall score is a
weighted average.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from utils.logger import get_logger
import config

logger = get_logger(__name__)

# System prompt sent to the LLM for risk evaluation
_RISK_SYSTEM_PROMPT = """You are a legal risk analyst AI. Given a clause from a legal document, evaluate it across five risk dimensions. For each dimension, provide:
1. A score from 0 (no risk) to 100 (extreme risk)
2. A brief explanation (1-2 sentences)

Risk Dimensions:
- penalty_severity: Does this clause impose excessive penalties, fines, or liquidated damages?
- clause_imbalance: Is this clause one-sided, giving disproportionate rights to one party?
- termination_unfairness: Does this clause create unfair termination conditions, auto-renewal traps, or unreasonable lock-in periods?
- liability_excess: Does this clause impose excessive liability or broad indemnification obligations?
- missing_rights: Does this clause fail to provide standard protections, notice periods, or dispute resolution mechanisms?

Respond ONLY with valid JSON in this exact format:
{
    "penalty_severity": {"score": <0-100>, "explanation": "<text>"},
    "clause_imbalance": {"score": <0-100>, "explanation": "<text>"},
    "termination_unfairness": {"score": <0-100>, "explanation": "<text>"},
    "liability_excess": {"score": <0-100>, "explanation": "<text>"},
    "missing_rights": {"score": <0-100>, "explanation": "<text>"}
}"""


class RiskAnalyzer:
    """LLM-assisted risk scoring for legal clauses."""

    def __init__(self) -> None:
        self.weights = config.RISK_WEIGHTS

    # ── Public API ────────────────────────────────────────────────────

    def analyze_clause(
        self,
        clause: str,
        llm_router,
        mode: str = "fast",
    ) -> dict:
        """Score a single clause across all risk dimensions.

        Returns
        -------
        dict
            ``dimensions`` (per-dimension score + explanation),
            ``overall_score`` (weighted 0-100),
            ``risk_level`` ("Low" / "Medium" / "High" / "Critical").
        """
        prompt = (
            f"Analyze the following legal clause for risks:\n\n"
            f"---\n{clause}\n---\n\n"
            f"Provide your analysis as JSON."
        )

        try:
            raw = llm_router.generate_response(
                prompt=prompt,
                mode=mode,
                system_prompt=_RISK_SYSTEM_PROMPT,
                temperature=0.1,       # low temp for consistent scoring
                max_tokens=1024,
            )
            dimensions = self._parse_risk_json(raw)
        except Exception as exc:
            logger.warning("LLM risk analysis failed: %s — using fallback", exc)
            dimensions = self._fallback_dimensions()

        overall = self.compute_overall_score(dimensions)
        risk_level = self._score_to_level(overall)

        return {
            "dimensions": dimensions,
            "overall_score": round(overall, 1),
            "risk_level": risk_level,
        }

    def analyze_document(
        self,
        clauses: list[dict],
        llm_router,
        mode: str = "fast",
    ) -> dict:
        """Analyze all clauses and produce a document-level report.

        Parameters
        ----------
        clauses : list[dict]
            Output of ``clause_segmenter.segment_clauses()``.

        Returns
        -------
        dict
            ``clause_risks`` (list), ``overall_score``, ``risk_level``,
            ``high_risk_clauses`` (ids of clauses scoring ≥ 60).
        """
        clause_risks: list[dict] = []
        for cl in clauses:
            result = self.analyze_clause(cl["text"], llm_router, mode)
            result["clause_id"] = cl["id"]
            result["clause_title"] = cl.get("title")
            clause_risks.append(result)

        # Document-level score = average of clause scores
        scores = [cr["overall_score"] for cr in clause_risks]
        doc_score = sum(scores) / len(scores) if scores else 0.0
        doc_level = self._score_to_level(doc_score)

        high_risk = [cr for cr in clause_risks if cr["overall_score"] >= 60]

        logger.info(
            "Document risk: %.1f (%s) — %d high-risk clause(s)",
            doc_score, doc_level, len(high_risk),
        )

        return {
            "clause_risks": clause_risks,
            "overall_score": round(doc_score, 1),
            "risk_level": doc_level,
            "high_risk_clauses": high_risk,
        }

    # ── Scoring ───────────────────────────────────────────────────────

    def compute_overall_score(self, dimensions: dict) -> float:
        """Weighted average of dimension scores."""
        total = 0.0
        for dim, weight in self.weights.items():
            dim_data = dimensions.get(dim, {})
            score = dim_data.get("score", 0)
            total += score * weight
        return total

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _parse_risk_json(raw: str) -> dict:
        """Extract and parse the JSON object from the LLM response."""
        # Try to find JSON block in the response
        json_match = re.search(r"\{[\s\S]*\}", raw)
        if json_match:
            data = json.loads(json_match.group())
            # Validate structure
            for dim in config.RISK_WEIGHTS:
                if dim not in data:
                    data[dim] = {"score": 0, "explanation": "Not evaluated."}
                else:
                    data[dim]["score"] = max(0, min(100, int(data[dim].get("score", 0))))
            return data
        raise ValueError("No valid JSON found in LLM response.")

    @staticmethod
    def _fallback_dimensions() -> dict:
        """Return neutral scores when LLM is unavailable."""
        return {
            dim: {"score": 0, "explanation": "Analysis unavailable."}
            for dim in config.RISK_WEIGHTS
        }

    @staticmethod
    def _score_to_level(score: float) -> str:
        if score >= 75:
            return "Critical"
        elif score >= 50:
            return "High"
        elif score >= 25:
            return "Medium"
        return "Low"
