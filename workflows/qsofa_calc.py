"""
workflows/qsofa_calc.py — Emergency Medicine: qSOFA Sepsis Screening.

Input:  Clinical Note (Text) + optional PDF.
Protocol: qSOFA (Quick Sequential Organ Failure Assessment).
Chain:
  1. Extraction  — LLM pulls systolic BP, respiratory rate, mental status.
  2. Calculation  — each criterion scores +1; total 0-3.
  3. Output      — risk stratification and recommendation.

Criteria (≥2 = high risk):
  • Systolic BP   ≤ 100 mmHg   (+1)
  • Respiratory rate ≥ 22 /min  (+1)
  • Altered mentation (GCS < 15) (+1)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from workflows.base import ClinicalWorkflow, InputType
from llm_config import llm_completion

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """\
You are a clinical extraction engine for sepsis screening.
From the following clinical text, extract the qSOFA parameters.

Return ONLY valid JSON:
{{
  "systolic_bp": <int or null>,
  "respiratory_rate": <int or null>,
  "gcs": <int 3-15 or null>,
  "altered_mentation": <true|false|null>,
  "temperature": <float or null>,
  "heart_rate": <int or null>,
  "suspected_infection": <true|false>,
  "extraction_notes": "<brief notes>"
}}

Clinical text:
---
{text}
---
"""


def _calculate_qsofa(params: dict) -> dict:
    """Deterministic qSOFA score (0-3)."""
    score = 0
    breakdown = {}

    # Criterion 1: systolic BP ≤ 100
    sbp = params.get("systolic_bp")
    if sbp is not None and sbp <= 100:
        score += 1
        breakdown["Systolic BP ≤100 mmHg"] = f"+1 (BP={sbp})"
    elif sbp is not None:
        breakdown["Systolic BP ≤100 mmHg"] = f"0 (BP={sbp})"

    # Criterion 2: RR ≥ 22
    rr = params.get("respiratory_rate")
    if rr is not None and rr >= 22:
        score += 1
        breakdown["Respiratory Rate ≥22/min"] = f"+1 (RR={rr})"
    elif rr is not None:
        breakdown["Respiratory Rate ≥22/min"] = f"0 (RR={rr})"

    # Criterion 3: altered mentation (GCS < 15)
    gcs = params.get("gcs")
    altered = params.get("altered_mentation")
    if gcs is not None and gcs < 15:
        score += 1
        breakdown["Altered Mentation (GCS<15)"] = f"+1 (GCS={gcs})"
    elif altered is True:
        score += 1
        breakdown["Altered Mentation (GCS<15)"] = "+1 (altered=true)"
    elif gcs is not None:
        breakdown["Altered Mentation (GCS<15)"] = f"0 (GCS={gcs})"

    # Risk stratification
    if score >= 2:
        risk = "HIGH"
        rec = ("qSOFA ≥2 — HIGH risk of poor outcome. "
               "Initiate sepsis bundle: obtain lactate, blood cultures, "
               "administer broad-spectrum antibiotics within 1 hour, "
               "start IV fluids 30 mL/kg. Escalate to ICU evaluation.")
    elif score == 1:
        risk = "INTERMEDIATE"
        rec = ("qSOFA = 1 — monitor closely. Reassess frequently. "
               "Consider full SOFA score calculation and sepsis workup "
               "if clinical suspicion remains.")
    else:
        risk = "LOW"
        rec = ("qSOFA = 0 — low risk by this screen. "
               "Continue clinical monitoring. qSOFA has limited sensitivity; "
               "do not use alone to rule out sepsis.")

    return {
        "score": score,
        "max_score": 3,
        "risk_level": risk,
        "breakdown": breakdown,
        "recommendation": rec,
    }


class QSofaCalcWorkflow(ClinicalWorkflow):
    workflow_id  = "qsofa_calc"
    name         = "qSOFA Calc"
    icon         = "🦠"
    description  = "Quick SOFA sepsis screening — counts 3 bedside criteria to flag high-risk patients."
    input_types  = [InputType.TEXT, InputType.PDF]
    protocol     = "qSOFA (Sepsis-3)"
    specialty    = "Emergency Medicine"

    def validate_input(self, data: dict[str, Any]) -> list[str]:
        errors = []
        if not data.get("text") and not data.get("pdf_text"):
            errors.append("Clinical note text or PDF text is required.")
        return errors

    def execute(self, data: dict[str, Any]) -> dict[str, Any]:
        combined = ""
        if data.get("text"):
            combined += data["text"]
        if data.get("pdf_text"):
            combined += "\n\n" + data["pdf_text"]

        # ── Step 1: LLM extraction ───────────────────────────────────────
        prompt = EXTRACTION_PROMPT.replace("{text}", combined)
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
            params = json.loads(clean)
        except (json.JSONDecodeError, IndexError):
            params = {
                "systolic_bp": None, "respiratory_rate": None,
                "gcs": None, "altered_mentation": None,
                "extraction_notes": raw,
            }

        # ── Step 2: Deterministic calculation ────────────────────────────
        calc = _calculate_qsofa(params)

        # ── Step 3: Assemble output ──────────────────────────────────────
        summary = (
            f"qSOFA Score: {calc['score']}/{calc['max_score']} — "
            f"{calc['risk_level']} risk. {calc['recommendation'][:120]}"
        )

        report_lines = [
            "═══ qSOFA SEPSIS SCREENING REPORT ═══",
            "",
            f"Score: {calc['score']} / {calc['max_score']}",
            f"Risk Level: {calc['risk_level']}",
            "",
            "─── Criteria ───",
        ]
        for criterion, detail in calc["breakdown"].items():
            report_lines.append(f"  {criterion:.<45} {detail}")
        report_lines += [
            "",
            "─── Recommendation ───",
            calc["recommendation"],
            "",
            "─── Extracted Parameters ───",
            f"  Systolic BP: {params.get('systolic_bp', '?')} mmHg",
            f"  Respiratory Rate: {params.get('respiratory_rate', '?')} /min",
            f"  GCS: {params.get('gcs', '?')}",
            f"  Temperature: {params.get('temperature', '?')} °C",
            f"  Heart Rate: {params.get('heart_rate', '?')} bpm",
            f"  Suspected Infection: {params.get('suspected_infection', '?')}",
            f"  Notes: {params.get('extraction_notes', '')}",
        ]

        return {
            "summary": summary,
            "metrics": {
                "qSOFA_score": f"{calc['score']}/{calc['max_score']}",
                "risk_level": calc["risk_level"],
                "recommendation": calc["recommendation"][:80],
            },
            "protocol_adherence": True,
            "raw_output": "\n".join(report_lines),
        }
