"""
workflows/growth_plotter.py — Pediatrics: Growth Percentile Calculator.

Input:  Intake Note (Text) with height, weight, age.
Protocol: CDC/WHO Growth Charts.
Chain:
  1. Extraction — pull numerical vitals and patient age in months.
  2. Math Step  — BMI = weight(kg) / height(m)².
  3. Reference  — map values to growth percentiles.
  4. Output     — summary of growth trends and percentile status.
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
You are a pediatric data extraction engine.
From the following intake note, extract the child's growth parameters.

Return ONLY valid JSON:
{
  "age_months": <int>,
  "sex": "<male|female>",
  "weight_kg": <float>,
  "height_cm": <float>,
  "head_circumference_cm": <float or null>,
  "extraction_notes": "<any >"
}

If the age is given in years, convert to months (e.g., 3 years = 36 months).
If weight is in pounds, convert to kg (1 lb = 0.4536 kg).
If height is in inches, convert to cm (1 in = 2.54 cm).

Intake Note:
---
{text}
---
"""

# ── Simplified CDC/WHO percentile reference tables ───────────────────────────
# In production, use the full LMS tables. Here we use approximate medians and
# SDs for demonstration. Tables indexed by (sex, age_months).

# BMI-for-age approximate percentile boundaries (simplified)
# These are rough approximations of CDC 2000 reference data.
_BMI_REFS = {
    # (sex, age_range) -> {"p5": val, "p10": val, "p25": val, "p50": val, "p75": val, "p85": val, "p95": val}
    ("male", "0-24"):    {"p5": 14.5, "p10": 15.0, "p25": 15.8, "p50": 16.8, "p75": 17.8, "p85": 18.3, "p95": 19.5},
    ("male", "25-60"):   {"p5": 13.8, "p10": 14.2, "p25": 14.9, "p50": 15.7, "p75": 16.5, "p85": 17.0, "p95": 18.0},
    ("male", "61-120"):  {"p5": 13.5, "p10": 14.0, "p25": 14.8, "p50": 15.8, "p75": 17.2, "p85": 18.2, "p95": 20.5},
    ("male", "121-240"): {"p5": 14.5, "p10": 15.3, "p25": 16.5, "p50": 18.5, "p75": 21.0, "p85": 23.0, "p95": 26.5},
    ("female", "0-24"):    {"p5": 14.2, "p10": 14.8, "p25": 15.5, "p50": 16.5, "p75": 17.5, "p85": 18.0, "p95": 19.2},
    ("female", "25-60"):   {"p5": 13.5, "p10": 13.9, "p25": 14.6, "p50": 15.4, "p75": 16.2, "p85": 16.8, "p95": 17.8},
    ("female", "61-120"):  {"p5": 13.2, "p10": 13.7, "p25": 14.5, "p50": 15.6, "p75": 17.0, "p85": 18.0, "p95": 20.2},
    ("female", "121-240"): {"p5": 14.2, "p10": 15.0, "p25": 16.3, "p50": 18.3, "p75": 21.0, "p85": 23.2, "p95": 27.0},
}

# Weight-for-age approximate 50th percentile (kg) — simplified
_WEIGHT_P50 = {
    "male":   {0: 3.3, 6: 7.9, 12: 10.0, 24: 12.7, 36: 14.3, 48: 16.3, 60: 18.4,
               72: 20.7, 96: 25.5, 120: 31.4, 144: 39.8, 168: 50.8, 192: 60.4, 216: 67.5},
    "female": {0: 3.2, 6: 7.3, 12: 9.5, 24: 12.1, 36: 13.9, 48: 16.0, 60: 18.2,
               72: 20.5, 96: 25.7, 120: 32.5, 144: 41.5, 168: 49.2, 192: 55.0, 216: 57.5},
}

# Height-for-age approximate 50th percentile (cm) — simplified
_HEIGHT_P50 = {
    "male":   {0: 49.9, 6: 67.6, 12: 75.7, 24: 87.1, 36: 95.3, 48: 102.5, 60: 109.2,
               72: 115.5, 96: 127.0, 120: 137.5, 144: 149.1, 168: 163.2, 192: 173.5, 216: 176.8},
    "female": {0: 49.1, 6: 65.7, 12: 74.0, 24: 85.5, 36: 94.1, 48: 101.5, 60: 108.4,
               72: 115.0, 96: 127.0, 120: 138.0, 144: 151.5, 168: 160.0, 192: 163.0, 216: 163.5},
}


def _get_age_range(months: int) -> str:
    if months <= 24: return "0-24"
    if months <= 60: return "25-60"
    if months <= 120: return "61-120"
    return "121-240"


def _estimate_percentile(bmi: float, sex: str, age_months: int) -> dict:
    """Estimate BMI percentile from simplified reference data."""
    sex_key = sex.lower() if sex else "male"
    age_range = _get_age_range(age_months)
    key = (sex_key, age_range)
    ref = _BMI_REFS.get(key, _BMI_REFS.get(("male", age_range)))

    if not ref:
        return {"percentile": "Unknown", "category": "Unknown"}

    if bmi < ref["p5"]:
        return {"percentile": "<5th", "category": "Underweight"}
    elif bmi < ref["p10"]:
        return {"percentile": "5th-10th", "category": "Low-normal"}
    elif bmi < ref["p25"]:
        return {"percentile": "10th-25th", "category": "Normal"}
    elif bmi < ref["p50"]:
        return {"percentile": "25th-50th", "category": "Normal"}
    elif bmi < ref["p75"]:
        return {"percentile": "50th-75th", "category": "Normal"}
    elif bmi < ref["p85"]:
        return {"percentile": "75th-85th", "category": "Normal"}
    elif bmi < ref["p95"]:
        return {"percentile": "85th-95th", "category": "Overweight"}
    else:
        return {"percentile": ">95th", "category": "Obese"}


def _nearest_ref(age_months: int, ref_table: dict) -> float | None:
    """Find nearest reference value for the given age."""
    ages = sorted(ref_table.keys())
    closest = min(ages, key=lambda a: abs(a - age_months))
    return ref_table[closest]


class GrowthPlotterWorkflow(ClinicalWorkflow):
    workflow_id  = "growth_plotter"
    name         = "GrowthPlotter"
    icon         = "📏"
    description  = "Pediatric growth assessment — BMI calculation & CDC/WHO percentile mapping."
    input_types  = [InputType.TEXT]
    protocol     = "CDC/WHO Growth Charts"
    specialty    = "Pediatrics"

    def validate_input(self, data: dict[str, Any]) -> list[str]:
        errors = []
        if not data.get("text"):
            errors.append("Intake note text with growth parameters is required.")
        return errors

    def execute(self, data: dict[str, Any]) -> dict[str, Any]:
        # ── Step 1: LLM extraction ───────────────────────────────────────────
        prompt = EXTRACTION_PROMPT.replace("{text}", data["text"])
        messages = [
            {"role": "system", "content": "You are a pediatric data extraction assistant."},
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
            vitals = json.loads(clean)
        except (json.JSONDecodeError, IndexError):
            vitals = {"age_months": None, "sex": "unknown", "weight_kg": None,
                      "height_cm": None, "head_circumference_cm": None,
                      "extraction_notes": raw}

        age_months = vitals.get("age_months")
        sex = vitals.get("sex", "unknown")
        weight_kg = vitals.get("weight_kg")
        height_cm = vitals.get("height_cm")
        head_cm = vitals.get("head_circumference_cm")

        # ── Step 2: BMI calculation ──────────────────────────────────────────
        bmi = None
        if weight_kg and height_cm and height_cm > 0:
            height_m = height_cm / 100.0
            bmi = round(weight_kg / (height_m ** 2), 1)

        # ── Step 3: Percentile mapping ───────────────────────────────────────
        bmi_percentile = {"percentile": "N/A", "category": "N/A"}
        if bmi is not None and age_months is not None:
            bmi_percentile = _estimate_percentile(bmi, sex, age_months)

        # Weight & height vs reference
        weight_status = "N/A"
        height_status = "N/A"
        sex_key = sex.lower() if sex in ("male", "female") else "male"

        if weight_kg and age_months is not None:
            w_ref = _nearest_ref(age_months, _WEIGHT_P50.get(sex_key, {}))
            if w_ref:
                w_pct = round((weight_kg / w_ref) * 100, 1)
                weight_status = f"{w_pct}% of 50th percentile ({w_ref} kg)"

        if height_cm and age_months is not None:
            h_ref = _nearest_ref(age_months, _HEIGHT_P50.get(sex_key, {}))
            if h_ref:
                h_pct = round((height_cm / h_ref) * 100, 1)
                height_status = f"{h_pct}% of 50th percentile ({h_ref} cm)"

        # ── Step 4: Assemble output ──────────────────────────────────────────
        age_str = f"{age_months} months" if age_months else "Unknown"
        if age_months and age_months >= 24:
            age_str += f" ({age_months // 12} years {age_months % 12} months)"

        summary = (
            f"Patient: {sex.title()}, {age_str}. "
            f"BMI: {bmi or 'N/A'} kg/m² — {bmi_percentile['category']} "
            f"({bmi_percentile['percentile']} percentile)."
        )

        report_lines = [
            "═══ PEDIATRIC GROWTH ASSESSMENT ═══",
            f"Protocol: CDC/WHO Growth Charts",
            "",
            f"Age:    {age_str}",
            f"Sex:    {sex.title()}",
            f"Weight: {weight_kg or '?'} kg",
            f"Height: {height_cm or '?'} cm",
            f"Head C: {head_cm or 'N/A'} cm",
            "",
            "─── Calculated Values ───",
            f"  BMI: {bmi or 'N/A'} kg/m²",
            f"  BMI Percentile: {bmi_percentile['percentile']}",
            f"  BMI Category: {bmi_percentile['category']}",
            "",
            "─── Reference Comparison (vs 50th Percentile) ───",
            f"  Weight: {weight_status}",
            f"  Height: {height_status}",
        ]

        if bmi_percentile["category"] == "Underweight":
            report_lines += ["", "⚠️  Below 5th percentile — evaluate for nutritional deficiency or chronic illness."]
        elif bmi_percentile["category"] == "Overweight":
            report_lines += ["", "⚠️  85th-95th percentile — overweight. Lifestyle counseling recommended."]
        elif bmi_percentile["category"] == "Obese":
            report_lines += ["", "🛑 Above 95th percentile — obese. Comprehensive evaluation recommended."]

        return {
            "summary": summary,
            "metrics": {
                "age": age_str,
                "bmi": f"{bmi} kg/m²" if bmi else "N/A",
                "bmi_percentile": bmi_percentile["percentile"],
                "bmi_category": bmi_percentile["category"],
                "weight_vs_ref": weight_status,
                "height_vs_ref": height_status,
            },
            "protocol_adherence": True,
            "raw_output": "\n".join(report_lines),
        }
