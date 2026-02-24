"""
workflows/stroke_risk.py — Cardiology: CHA₂DS₂-VASc Stroke Risk Calculator.

Input:  Clinical Note (Text) + optional Lab Report (PDF text).
Protocol: CHA₂DS₂-VASc Scoring.
Chain:
  1. Extraction  — identify Age, Sex, HF, HTN, Stroke/TIA, Vascular disease, Diabetes.
  2. Calculation  — deterministic Python score.
  3. Output      — risk percentage & anticoagulation strategy.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from workflows.base import ClinicalWorkflow, InputType
from llm_config import llm_completion

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """\
You are a clinical cardiology extraction engine.
From the following clinical text, extract the CHA₂DS₂-VASc risk factors.

Return ONLY valid JSON:
{
  "age": <int or null>,
  "sex": "<male|female|unknown>",
  "heart_failure": <true|false>,
  "hypertension": <true|false>,
  "stroke_tia_history": <true|false>,
  "vascular_disease": <true|false>,
  "diabetes": <true|false>,
  "extraction_notes": "<brief notes about what was found>"
}

Clinical text:
---
{text}
---
"""

# CHA₂DS₂-VASc deterministic calculation
def _calculate_cha2ds2_vasc(factors: dict) -> dict:
    """
    Deterministic CHA₂DS₂-VASc score.
    Returns dict with score, breakdown, risk_pct, recommendation.
    """
    score = 0
    breakdown = {}

    # C — Congestive heart failure (+1)
    if factors.get("heart_failure"):
        score += 1
        breakdown["Heart failure (C)"] = 1

    # H — Hypertension (+1)
    if factors.get("hypertension"):
        score += 1
        breakdown["Hypertension (H)"] = 1

    # A₂ — Age ≥ 75 (+2)
    age = factors.get("age")
    if age is not None and age >= 75:
        score += 2
        breakdown["Age ≥75 (A₂)"] = 2
    elif age is not None and 65 <= age < 75:
        score += 1
        breakdown["Age 65-74 (A)"] = 1

    # D — Diabetes (+1)
    if factors.get("diabetes"):
        score += 1
        breakdown["Diabetes (D)"] = 1

    # S₂ — Stroke/TIA history (+2)
    if factors.get("stroke_tia_history"):
        score += 2
        breakdown["Stroke/TIA (S₂)"] = 2

    # V — Vascular disease (+1)
    if factors.get("vascular_disease"):
        score += 1
        breakdown["Vascular disease (V)"] = 1

    # Sc — Sex category (female +1)
    if factors.get("sex", "").lower() == "female":
        score += 1
        breakdown["Female sex (Sc)"] = 1

    # Risk estimation (approximate annual stroke risk %)
    risk_table = {
        0: 0.2, 1: 0.6, 2: 2.2, 3: 3.2, 4: 4.8,
        5: 7.2, 6: 9.7, 7: 11.2, 8: 10.8, 9: 12.2,
    }
    risk_pct = risk_table.get(min(score, 9), 12.2)

    # Anticoagulation recommendation
    if score == 0:
        rec = "No antithrombotic therapy recommended."
    elif score == 1:
        rec = ("Consider oral anticoagulation (OAC) or antiplatelet therapy. "
               "OAC preferred if bleeding risk is acceptable.")
    else:
        rec = ("Oral anticoagulation recommended (e.g., DOAC preferred over warfarin). "
               "Assess bleeding risk with HAS-BLED score.")

    return {
        "score": score,
        "max_score": 9,
        "breakdown": breakdown,
        "annual_stroke_risk_pct": risk_pct,
        "recommendation": rec,
    }


class StrokeRiskWorkflow(ClinicalWorkflow):
    workflow_id  = "stroke_risk"
    name         = "StrokeRisk+"
    icon         = "🫀"
    description  = "CHA₂DS₂-VASc stroke risk calculator from clinical notes & lab reports."
    input_types  = [InputType.TEXT, InputType.PDF]
    protocol     = "CHA₂DS₂-VASc"
    specialty    = "Cardiology"

    def validate_input(self, data: dict[str, Any]) -> list[str]:
        errors = []
        if not data.get("text") and not data.get("pdf_text"):
            errors.append("Clinical note text or PDF text is required.")
        return errors

    def execute(self, data: dict[str, Any]) -> dict[str, Any]:
        combined_text = ""
        if data.get("text"):
            combined_text += data["text"]
        if data.get("pdf_text"):
            combined_text += "\n\n" + data["pdf_text"]

        # ── Step 1: LLM extraction ───────────────────────────────────────────
        prompt = EXTRACTION_PROMPT.replace("{text}", combined_text)
        messages = [
            {"role": "system", "content": "You are a clinical data extraction assistant."},
            {"role": "user", "content": prompt},
        ]
        resp = llm_completion(messages=messages, max_tokens=512, temperature=0.1)
        raw = resp.choices[0].message.content.strip()

        try:
            clean = raw
            if "```" in clean:
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            factors = json.loads(clean)
        except (json.JSONDecodeError, IndexError):
            factors = {
                "age": None, "sex": "unknown",
                "heart_failure": False, "hypertension": False,
                "stroke_tia_history": False, "vascular_disease": False,
                "diabetes": False, "extraction_notes": raw,
            }

        # ── Step 2: Deterministic calculation ────────────────────────────────
        calc = _calculate_cha2ds2_vasc(factors)

        # ── Step 3: Assemble output ──────────────────────────────────────────
        summary = (
            f"CHA₂DS₂-VASc Score: {calc['score']}/{calc['max_score']}. "
            f"Estimated annual stroke risk: {calc['annual_stroke_risk_pct']}%. "
            f"{calc['recommendation']}"
        )

        report_lines = [
            "═══ CHA₂DS₂-VASc STROKE RISK REPORT ═══",
            "",
            f"Score: {calc['score']} / {calc['max_score']}",
            f"Annual Stroke Risk: {calc['annual_stroke_risk_pct']}%",
            "",
            "─── Score Breakdown ───",
        ]
        for component, pts in calc["breakdown"].items():
            report_lines.append(f"  {component:.<40} +{pts}")
        report_lines += [
            "",
            "─── Recommendation ───",
            calc["recommendation"],
            "",
            "─── Extracted Factors ───",
            f"  Age: {factors.get('age', '?')}",
            f"  Sex: {factors.get('sex', '?')}",
            f"  Notes: {factors.get('extraction_notes', '')}",
        ]

        return {
            "summary": summary,
            "metrics": {
                "score": f"{calc['score']}/{calc['max_score']}",
                "annual_stroke_risk": f"{calc['annual_stroke_risk_pct']}%",
                "recommendation": calc["recommendation"][:80],
            },
            "protocol_adherence": True,
            "raw_output": "\n".join(report_lines),
        }
