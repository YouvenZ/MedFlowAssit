"""
workflows/meld_calc.py — Hepatology: MELD-Na Score Calculator.

Input:  Clinical Note / Lab Report (Text, PDF).
Protocol: MELD-Na (Model for End-Stage Liver Disease, Sodium-adjusted).
Chain:
  1. Extraction  — LLM pulls bilirubin, INR, creatinine, sodium, dialysis status.
  2. Calculation  — deterministic MELD-Na formula (UNOS 2016 revision).
  3. Output      — score, 90-day mortality estimate, transplant priority.

MELD(i) = 10 × (0.957 × ln(Cr) + 0.378 × ln(Bili) + 1.120 × ln(INR) + 0.643)
  then round to nearest integer, cap at 40, floor at 6.
If dialysis ≥2× in past week → Cr = 3.0.
MELD-Na = MELD(i) + 1.32 × (137 − Na) − 0.033 × MELD(i) × (137 − Na)
  Na clamped to [125, 137].
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

from workflows.base import ClinicalWorkflow, InputType
from llm_config import llm_completion

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """\
You are a clinical lab value extraction engine.
From the following text, extract the values needed for the MELD-Na score.

Return ONLY valid JSON:
{{
  "bilirubin_mg_dl": <float or null>,
  "inr": <float or null>,
  "creatinine_mg_dl": <float or null>,
  "sodium_meq_l": <float or null>,
  "dialysis_past_week": <true|false|null>,
  "etiology": "<e.g. alcoholic cirrhosis, hepatitis C, NASH, etc. or null>",
  "extraction_notes": "<brief notes>"
}}

Clinical text:
---
{text}
---
"""


def _calculate_meld_na(labs: dict) -> dict:
    """
    Deterministic MELD-Na score (UNOS 2016 formula).
    Returns dict with meld_i, meld_na, mortality_90d, priority.
    """
    cr = labs.get("creatinine_mg_dl")
    bili = labs.get("bilirubin_mg_dl")
    inr = labs.get("inr")
    na = labs.get("sodium_meq_l")
    dialysis = labs.get("dialysis_past_week", False)

    missing = []
    if cr is None: missing.append("creatinine")
    if bili is None: missing.append("bilirubin")
    if inr is None: missing.append("INR")
    if missing:
        return {
            "meld_i": None, "meld_na": None,
            "error": f"Missing lab values: {', '.join(missing)}",
        }

    # Floor all lab values at 1.0 per UNOS rules
    cr = max(float(cr), 1.0)
    bili = max(float(bili), 1.0)
    inr = max(float(inr), 1.0)

    # If on dialysis ≥2x past week → Cr capped at 3.0
    if dialysis:
        cr = min(cr, 3.0)
        cr = max(cr, 3.0)  # set to exactly 3.0

    # Cap creatinine at 4.0
    cr = min(cr, 4.0)

    # MELD(i)
    meld_i = 10 * (
        0.957 * math.log(cr) +
        0.378 * math.log(bili) +
        1.120 * math.log(inr) +
        0.643
    )
    meld_i = round(meld_i)
    meld_i = max(meld_i, 6)
    meld_i = min(meld_i, 40)

    # MELD-Na adjustment
    meld_na = meld_i
    if na is not None:
        na_clamped = max(min(float(na), 137), 125)
        meld_na = (
            meld_i
            + 1.32 * (137 - na_clamped)
            - 0.033 * meld_i * (137 - na_clamped)
        )
        meld_na = round(meld_na)
        meld_na = max(meld_na, 6)
        meld_na = min(meld_na, 40)

    # 90-day mortality estimate (approximate from literature)
    mortality_table = {
        (6, 9): "<2%", (10, 19): "6%", (20, 29): "20%",
        (30, 39): "53%", (40, 40): ">70%",
    }
    mortality = "N/A"
    for (lo, hi), pct in mortality_table.items():
        if lo <= meld_na <= hi:
            mortality = pct
            break

    # Transplant priority
    if meld_na >= 25:
        priority = "HIGH — strong transplant candidacy"
    elif meld_na >= 15:
        priority = "MODERATE — evaluate for listing"
    else:
        priority = "LOW — monitor, re-assess periodically"

    return {
        "meld_i": meld_i,
        "meld_na": meld_na,
        "mortality_90d": mortality,
        "priority": priority,
        "cr_used": cr,
        "bili_used": bili,
        "inr_used": inr,
        "na_clamped": na_clamped if na is not None else None,
        "dialysis": dialysis,
    }


class MeldCalcWorkflow(ClinicalWorkflow):
    workflow_id  = "meld_calc"
    name         = "MELD-Na Calc"
    icon         = "🫘"
    description  = "MELD-Na score calculator for end-stage liver disease severity and transplant priority."
    input_types  = [InputType.TEXT, InputType.PDF]
    protocol     = "MELD-Na (UNOS 2016)"
    specialty    = "Hepatology"

    def validate_input(self, data: dict[str, Any]) -> list[str]:
        errors = []
        if not data.get("text") and not data.get("pdf_text"):
            errors.append("Clinical note or lab report text is required.")
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
            {"role": "system", "content": "You are a clinical lab extraction assistant."},
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
            labs = json.loads(clean)
        except (json.JSONDecodeError, IndexError):
            labs = {
                "bilirubin_mg_dl": None, "inr": None,
                "creatinine_mg_dl": None, "sodium_meq_l": None,
                "extraction_notes": raw,
            }

        # ── Step 2: Deterministic calculation ────────────────────────────
        calc = _calculate_meld_na(labs)

        if calc.get("error"):
            return {
                "summary": f"MELD-Na calculation failed: {calc['error']}",
                "metrics": {},
                "protocol_adherence": False,
                "raw_output": f"Error: {calc['error']}\nExtracted: {json.dumps(labs, indent=2)}",
            }

        # ── Step 3: Assemble output ──────────────────────────────────────
        summary = (
            f"MELD-Na Score: {calc['meld_na']} (MELD(i)={calc['meld_i']}). "
            f"90-day mortality: {calc['mortality_90d']}. {calc['priority']}."
        )

        report_lines = [
            "═══ MELD-Na LIVER DISEASE SEVERITY REPORT ═══",
            "",
            f"MELD(i) Score : {calc['meld_i']}",
            f"MELD-Na Score : {calc['meld_na']}",
            f"90-day Mortality Estimate: {calc['mortality_90d']}",
            f"Transplant Priority: {calc['priority']}",
            "",
            "─── Lab Values Used ───",
            f"  Creatinine : {calc['cr_used']} mg/dL  (input: {labs.get('creatinine_mg_dl')})",
            f"  Bilirubin  : {calc['bili_used']} mg/dL  (input: {labs.get('bilirubin_mg_dl')})",
            f"  INR        : {calc['inr_used']}        (input: {labs.get('inr')})",
            f"  Sodium     : {calc.get('na_clamped', '?')} mEq/L (input: {labs.get('sodium_meq_l')},"
            f" clamped [125,137])",
            f"  Dialysis   : {calc['dialysis']}",
            "",
            "─── Etiology ───",
            f"  {labs.get('etiology', 'Not specified')}",
            "",
            "─── Notes ───",
            f"  {labs.get('extraction_notes', '')}",
        ]

        return {
            "summary": summary,
            "metrics": {
                "MELD_i": str(calc["meld_i"]),
                "MELD_Na": str(calc["meld_na"]),
                "mortality_90d": calc["mortality_90d"],
                "priority": calc["priority"],
            },
            "protocol_adherence": True,
            "raw_output": "\n".join(report_lines),
        }
