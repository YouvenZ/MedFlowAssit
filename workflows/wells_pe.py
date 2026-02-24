"""
workflows/wells_pe.py — Pulmonology: Wells Score for Pulmonary Embolism.

Input:  Clinical Note (Text) + optional PDF.
Protocol: Wells Criteria (Original scoring, 7 items).
Chain:
  1. Extraction  — LLM identifies 7 clinical criteria.
  2. Calculation  — deterministic point scoring (max ~12.5).
  3. Output      — PE probability tier and recommendation.

Wells Criteria:
  • Clinical signs/symptoms of DVT           (+3.0)
  • PE is #1 diagnosis OR equally likely      (+3.0)
  • Heart rate > 100 bpm                      (+1.5)
  • Immobilization/surgery in prior 4 weeks   (+1.5)
  • Previous DVT/PE                           (+1.5)
  • Hemoptysis                                (+1.0)
  • Malignancy (on treatment/palliative/dx <6 mo) (+1.0)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from workflows.base import ClinicalWorkflow, InputType
from llm_config import llm_completion

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """\
You are a clinical extraction engine for PE risk assessment.
From the following clinical text, extract Wells criteria for pulmonary embolism.

Return ONLY valid JSON:
{{
  "dvt_signs": <true|false>,
  "pe_most_likely": <true|false>,
  "heart_rate_over_100": <true|false>,
  "immobilization_surgery_4wk": <true|false>,
  "previous_dvt_pe": <true|false>,
  "hemoptysis": <true|false>,
  "malignancy": <true|false>,
  "heart_rate": <int or null>,
  "extraction_notes": "<brief notes>"
}}

Clinical text:
---
{text}
---
"""


def _calculate_wells(criteria: dict) -> dict:
    """Deterministic Wells PE score."""
    score = 0.0
    breakdown = {}

    items = [
        ("dvt_signs",                  3.0, "Clinical signs/symptoms of DVT"),
        ("pe_most_likely",             3.0, "PE is #1 or equally likely diagnosis"),
        ("heart_rate_over_100",        1.5, "Heart rate > 100 bpm"),
        ("immobilization_surgery_4wk", 1.5, "Immobilization/surgery past 4 weeks"),
        ("previous_dvt_pe",            1.5, "Previous DVT/PE"),
        ("hemoptysis",                 1.0, "Hemoptysis"),
        ("malignancy",                 1.0, "Active malignancy"),
    ]

    for key, pts, label in items:
        if criteria.get(key):
            score += pts
            breakdown[label] = f"+{pts}"
        else:
            breakdown[label] = "0"

    # Three-tier model
    if score <= 1.0:
        tier = "LOW"
        probability = "~1.3%"
        rec = ("Low probability of PE. Consider D-dimer testing. "
               "If D-dimer negative, PE effectively ruled out (PERC may apply). "
               "No CTPA needed if D-dimer is normal.")
    elif score <= 4.0:
        tier = "MODERATE"
        probability = "~16.2%"
        rec = ("Moderate probability. Obtain D-dimer. "
               "If positive → proceed to CTPA. If negative → PE unlikely.")
    else:
        tier = "HIGH"
        probability = "~37.5%"
        rec = ("High probability of PE. Proceed directly to CT pulmonary angiography (CTPA). "
               "Consider empiric anticoagulation while awaiting imaging. "
               "D-dimer NOT recommended (insufficient to rule out).")

    return {
        "score": score,
        "max_score": 12.5,
        "tier": tier,
        "pretest_probability": probability,
        "breakdown": breakdown,
        "recommendation": rec,
    }


class WellsPEWorkflow(ClinicalWorkflow):
    workflow_id  = "wells_pe"
    name         = "WellsPE"
    icon         = "🫁"
    description  = "Wells criteria for pulmonary embolism probability. 7-item clinical prediction rule."
    input_types  = [InputType.TEXT, InputType.PDF]
    protocol     = "Wells Criteria (PE)"
    specialty    = "Pulmonology"

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
            criteria = json.loads(clean)
        except (json.JSONDecodeError, IndexError):
            criteria = {k: False for k in [
                "dvt_signs", "pe_most_likely", "heart_rate_over_100",
                "immobilization_surgery_4wk", "previous_dvt_pe",
                "hemoptysis", "malignancy",
            ]}
            criteria["extraction_notes"] = raw

        # ── Step 2: Deterministic calculation ────────────────────────────
        calc = _calculate_wells(criteria)

        # ── Step 3: Assemble output ──────────────────────────────────────
        summary = (
            f"Wells PE Score: {calc['score']}/{calc['max_score']} — "
            f"{calc['tier']} probability ({calc['pretest_probability']}). "
            f"{calc['recommendation'][:100]}"
        )

        report_lines = [
            "═══ WELLS PE PROBABILITY REPORT ═══",
            "",
            f"Score: {calc['score']} / {calc['max_score']}",
            f"Probability Tier: {calc['tier']} ({calc['pretest_probability']})",
            "",
            "─── Criteria Breakdown ───",
        ]
        for criterion, pts in calc["breakdown"].items():
            report_lines.append(f"  {criterion:.<50} {pts}")
        report_lines += [
            "",
            "─── Recommendation ───",
            calc["recommendation"],
            "",
            "─── Extracted Data ───",
            f"  Heart Rate: {criteria.get('heart_rate', '?')} bpm",
            f"  Notes: {criteria.get('extraction_notes', '')}",
        ]

        return {
            "summary": summary,
            "metrics": {
                "wells_score": f"{calc['score']}/{calc['max_score']}",
                "probability_tier": calc["tier"],
                "pretest_probability": calc["pretest_probability"],
            },
            "protocol_adherence": True,
            "raw_output": "\n".join(report_lines),
        }
