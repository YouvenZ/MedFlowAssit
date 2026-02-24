"""
workflows/gcs_calc.py — Neurology / Emergency: Glasgow Coma Scale.

Input:  Clinical Note (Text) + optional PDF.
Protocol: Glasgow Coma Scale (GCS).
Chain:
  1. Extraction  — LLM identifies Eye, Verbal, Motor responses.
  2. Calculation  — deterministic GCS = E + V + M (3-15).
  3. Output      — severity classification and TBI grade.

GCS Components:
  Eye (E):   1-4   (1=None, 2=Pain, 3=Voice, 4=Spontaneous)
  Verbal (V): 1-5  (1=None, 2=Incomprehensible, 3=Inappropriate, 4=Confused, 5=Oriented)
  Motor (M):  1-6  (1=None, 2=Extension, 3=Flexion, 4=Withdrawal, 5=Localizing, 6=Obeys)

Severity:
  13-15 = Mild  |  9-12 = Moderate  |  3-8 = Severe
"""

from __future__ import annotations

import json
import logging
from typing import Any

from workflows.base import ClinicalWorkflow, InputType
from llm_config import llm_completion

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """\
You are a clinical extraction engine for neurological assessment.
From the following text, extract the Glasgow Coma Scale components.

GCS Scoring reference:
  Eye Opening (E): 4=Spontaneous, 3=To voice/command, 2=To pain, 1=None
  Verbal (V): 5=Oriented, 4=Confused, 3=Inappropriate words, 2=Incomprehensible sounds, 1=None
  Motor (M): 6=Obeys commands, 5=Localizes pain, 4=Withdrawal, 3=Abnormal flexion, 2=Extension, 1=None

Return ONLY valid JSON:
{{
  "eye_opening": <int 1-4 or null>,
  "verbal": <int 1-5 or null>,
  "motor": <int 1-6 or null>,
  "pupil_reactivity": "<both reactive|one reactive|none reactive|unknown>",
  "mechanism_of_injury": "<description or null>",
  "intubated": <true|false|null>,
  "extraction_notes": "<brief notes>"
}}

Clinical text:
---
{text}
---
"""

# GCS component descriptions for the report
_EYE_DESC = {1: "None", 2: "To pressure/pain", 3: "To voice/command", 4: "Spontaneous"}
_VERBAL_DESC = {1: "None", 2: "Incomprehensible sounds", 3: "Inappropriate words",
                4: "Confused", 5: "Oriented"}
_MOTOR_DESC = {1: "None", 2: "Extension", 3: "Abnormal flexion",
               4: "Normal flexion/withdrawal", 5: "Localizes pain", 6: "Obeys commands"}


def _calculate_gcs(params: dict) -> dict:
    """Deterministic GCS calculation."""
    e = params.get("eye_opening")
    v = params.get("verbal")
    m = params.get("motor")

    missing = []
    if e is None: missing.append("Eye opening (E)")
    if v is None: missing.append("Verbal (V)")
    if m is None: missing.append("Motor (M)")
    if missing:
        return {"total": None, "error": f"Missing components: {', '.join(missing)}"}

    e = max(1, min(int(e), 4))
    v = max(1, min(int(v), 5))
    m = max(1, min(int(m), 6))
    total = e + v + m

    # Severity classification
    if total >= 13:
        severity = "MILD"
        tbi_grade = "Mild TBI / Concussion"
        rec = ("Mild head injury. Observe for 4-6 hours. Discharge with head injury advice "
               "if no red flags (vomiting, seizures, worsening headache, focal deficits). "
               "CT head if any high-risk features per Canadian CT Head Rule.")
    elif total >= 9:
        severity = "MODERATE"
        tbi_grade = "Moderate TBI"
        rec = ("Moderate head injury. CT head STAT. Admit for neurological observation. "
               "Repeat GCS q1h. Neurosurgery consultation. Consider ICU if deteriorating.")
    else:
        severity = "SEVERE"
        tbi_grade = "Severe TBI"
        rec = ("Severe head injury — GCS ≤8. Intubate for airway protection. CT head URGENT. "
               "Neurosurgery STAT. ICP monitoring. ICU admission mandatory. "
               "Avoid hypotension (SBP >90) and hypoxia (SpO₂ >90%).")

    # Pupil reactivity score (GCS-P for prognostication)
    pupil = params.get("pupil_reactivity", "unknown")
    pupil_score = {"both reactive": 0, "one reactive": -1, "none reactive": -2}.get(pupil, 0)
    gcs_p = total + pupil_score

    return {
        "eye": e, "verbal": v, "motor": m,
        "total": total, "max_score": 15,
        "severity": severity,
        "tbi_grade": tbi_grade,
        "pupil_reactivity": pupil,
        "pupil_score": pupil_score,
        "gcs_p": gcs_p,
        "recommendation": rec,
        "eye_desc": _EYE_DESC.get(e, "?"),
        "verbal_desc": _VERBAL_DESC.get(v, "?"),
        "motor_desc": _MOTOR_DESC.get(m, "?"),
    }


class GCSCalcWorkflow(ClinicalWorkflow):
    workflow_id  = "gcs_calc"
    name         = "GCS Calc"
    icon         = "🧠"
    description  = "Glasgow Coma Scale — E+V+M neurological assessment with TBI classification."
    input_types  = [InputType.TEXT, InputType.PDF]
    protocol     = "Glasgow Coma Scale"
    specialty    = "Neurology"

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
                "eye_opening": None, "verbal": None, "motor": None,
                "extraction_notes": raw,
            }

        # ── Step 2: Deterministic calculation ────────────────────────────
        calc = _calculate_gcs(params)

        if calc.get("error"):
            return {
                "summary": f"GCS calculation failed: {calc['error']}",
                "metrics": {},
                "protocol_adherence": False,
                "raw_output": f"Error: {calc['error']}\nExtracted: {json.dumps(params, indent=2)}",
            }

        # ── Step 3: Assemble output ──────────────────────────────────────
        summary = (
            f"GCS Total: {calc['total']}/{calc['max_score']} "
            f"(E{calc['eye']}V{calc['verbal']}M{calc['motor']}) — "
            f"{calc['severity']} ({calc['tbi_grade']})"
        )

        report_lines = [
            "═══ GLASGOW COMA SCALE REPORT ═══",
            "",
            f"GCS Total: {calc['total']} / {calc['max_score']}  "
            f"(E{calc['eye']} V{calc['verbal']} M{calc['motor']})",
            f"Severity: {calc['severity']}",
            f"Classification: {calc['tbi_grade']}",
            "",
            "─── Component Breakdown ───",
            f"  Eye Opening (E)  : {calc['eye']}/4  — {calc['eye_desc']}",
            f"  Verbal (V)       : {calc['verbal']}/5  — {calc['verbal_desc']}",
            f"  Motor (M)        : {calc['motor']}/6  — {calc['motor_desc']}",
            "",
            "─── Pupil Reactivity ───",
            f"  Reactivity: {calc['pupil_reactivity']}",
            f"  GCS-Pupils (GCS-P): {calc['gcs_p']} (adjustment: {calc['pupil_score']})",
            "",
            "─── Recommendation ───",
            calc["recommendation"],
            "",
            "─── Additional ───",
            f"  Intubated: {params.get('intubated', '?')}",
            f"  Mechanism: {params.get('mechanism_of_injury', 'Not specified')}",
            f"  Notes: {params.get('extraction_notes', '')}",
        ]

        return {
            "summary": summary,
            "metrics": {
                "GCS_total": f"{calc['total']}/{calc['max_score']}",
                "components": f"E{calc['eye']}V{calc['verbal']}M{calc['motor']}",
                "severity": calc["severity"],
                "GCS_P": str(calc["gcs_p"]),
            },
            "protocol_adherence": True,
            "raw_output": "\n".join(report_lines),
        }
