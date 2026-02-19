"""
NyayaSathi AI — Report Generator
===================================
Produces a structured Markdown report summarising the analysis of a
legal document.  The report includes:
  • Document summary and classification
  • Key obligations and deadlines
  • Risk breakdown (per-clause and overall)
  • Suggested negotiation points

The Markdown output can be displayed in Streamlit and also exported
as a downloadable file.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from utils.logger import get_logger
import config

logger = get_logger(__name__)


class ReportGenerator:
    """Generates a comprehensive legal analysis report in Markdown."""

    def generate_report(
        self,
        doc_type: str,
        clauses: list[dict],
        risk_results: dict,
        simplified_texts: Optional[dict[int, str]] = None,
        obligations: Optional[list[str]] = None,
        deadlines: Optional[list[str]] = None,
        negotiation_points: Optional[list[str]] = None,
    ) -> str:
        """Build the full Markdown report.

        Parameters
        ----------
        doc_type : str
            Classification result (e.g. "Rental Agreement").
        clauses : list[dict]
            Output of ``clause_segmenter.segment_clauses()``.
        risk_results : dict
            Output of ``RiskAnalyzer.analyze_document()``.
        simplified_texts : dict | None
            Mapping of clause_id → simplified plain-English text.
        obligations : list[str] | None
            Key obligations extracted from the document.
        deadlines : list[str] | None
            Important deadlines extracted from the document.
        negotiation_points : list[str] | None
            Suggested points for negotiation.

        Returns
        -------
        str
            Complete Markdown report.
        """
        sections: list[str] = []

        # ───────── Header ─────────
        sections.append(f"# {config.APP_TITLE} — Analysis Report")
        sections.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        sections.append(f"**Document Type:** {doc_type}")
        sections.append(f"**Total Clauses:** {len(clauses)}")
        sections.append(
            f"**Overall Risk Score:** {risk_results.get('overall_score', 'N/A')} / 100 "
            f"({risk_results.get('risk_level', 'N/A')})"
        )
        sections.append("")

        # ───────── Summary ─────────
        sections.append("## 📋 Document Summary")
        sections.append(
            f"This **{doc_type}** contains **{len(clauses)} clauses**. "
            f"The overall risk assessment is "
            f"**{risk_results.get('risk_level', 'N/A')}** "
            f"(score: {risk_results.get('overall_score', 0)}/100)."
        )
        sections.append("")

        # ───────── Key Obligations ─────────
        if obligations:
            sections.append("## 📌 Key Obligations")
            for ob in obligations:
                sections.append(f"- {ob}")
            sections.append("")

        # ───────── Deadlines ─────────
        if deadlines:
            sections.append("## ⏰ Important Deadlines")
            for dl in deadlines:
                sections.append(f"- {dl}")
            sections.append("")

        # ───────── Risk Breakdown ─────────
        sections.append("## ⚠️ Risk Breakdown")
        sections.append("")

        # Per-clause table
        sections.append("| Clause | Title | Risk Score | Level |")
        sections.append("|--------|-------|-----------|-------|")
        for cr in risk_results.get("clause_risks", []):
            title = cr.get("clause_title") or f"Clause {cr['clause_id']}"
            sections.append(
                f"| {cr['clause_id']} | {title} | "
                f"{cr['overall_score']}/100 | {cr['risk_level']} |"
            )
        sections.append("")

        # High-risk details
        high_risk = risk_results.get("high_risk_clauses", [])
        if high_risk:
            sections.append("### 🔴 High-Risk Clauses (Score ≥ 60)")
            sections.append("")
            for cr in high_risk:
                title = cr.get("clause_title") or f"Clause {cr['clause_id']}"
                sections.append(f"#### {title} — Score: {cr['overall_score']}/100")
                for dim, data in cr.get("dimensions", {}).items():
                    label = dim.replace("_", " ").title()
                    sections.append(f"- **{label}** ({data['score']}/100): {data['explanation']}")
                sections.append("")

        # ───────── Simplified Explanations ─────────
        if simplified_texts:
            sections.append("## 📖 Simplified Explanations")
            sections.append("")
            for cl in clauses:
                cid = cl["id"]
                if cid in simplified_texts:
                    title = cl.get("title") or f"Clause {cid}"
                    sections.append(f"### {title}")
                    sections.append(simplified_texts[cid])
                    sections.append("")

        # ───────── Negotiation Points ─────────
        if negotiation_points:
            sections.append("## 💡 Suggested Negotiation Points")
            for pt in negotiation_points:
                sections.append(f"- {pt}")
            sections.append("")

        # ───────── Footer ─────────
        sections.append("---")
        sections.append(
            f"*Report generated by {config.APP_TITLE} v{config.APP_VERSION}. "
            f"This is an AI-assisted analysis and should not be considered legal advice.*"
        )

        report = "\n".join(sections)
        logger.info("Report generated (%d characters)", len(report))
        return report

    @staticmethod
    def export_to_file(report_text: str, filename: str = "analysis_report.md") -> str:
        """Save report to the reports/ directory and return the path."""
        reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports", "output")
        os.makedirs(reports_dir, exist_ok=True)
        path = os.path.join(reports_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(report_text)
        logger.info("Report exported to %s", path)
        return path
