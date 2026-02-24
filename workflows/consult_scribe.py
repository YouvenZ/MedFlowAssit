"""
workflows/consult_scribe.py — General Practice: SOAP Note Generator.

Input:  Raw dictation/notes (Text).
Protocol: SOAP Format (Subjective, Objective, Assessment, Plan).
Chain:
  1. Structuring — reorganize prose into SOAP sections.
  2. Coding      — suggest top 3 most relevant ICD-10 codes.
  3. Output      — standardized clinical note ready for EMR copy-paste.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from workflows.base import ClinicalWorkflow, InputType
from llm_config import llm_completion

logger = logging.getLogger(__name__)

SOAP_PROMPT = """\
You are a clinical documentation specialist. Convert the following raw
clinical dictation into a structured SOAP note.

Return ONLY valid JSON:
{
  "subjective": "<patient's reported symptoms, history, concerns>",
  "objective": "<vital signs, physical exam findings, lab/imaging results>",
  "assessment": "<clinical assessment, differential diagnoses>",
  "plan": "<treatment plan, medications, follow-up, referrals>",
  "icd10_codes": [
    {"code": "<ICD-10 code>", "description": "<code description>", "confidence": "<high|medium|low>"}
  ],
  "chief_complaint": "<one-line chief complaint>",
  "structured_notes": "<any additional structured observations>"
}

Important:
- If information for a section is missing, write "Not documented" for that section.
- Suggest the top 3 most relevant ICD-10 codes based on the assessment.
- Be concise but thorough.

Raw Dictation:
---
{text}
---
"""


class ConsultScribeWorkflow(ClinicalWorkflow):
    workflow_id  = "consult_scribe"
    name         = "ConsultScribe"
    icon         = "📝"
    description  = "Raw dictation → structured SOAP note with ICD-10 coding for EMR."
    input_types  = [InputType.TEXT]
    protocol     = "SOAP Format"
    specialty    = "General Practice"

    def validate_input(self, data: dict[str, Any]) -> list[str]:
        errors = []
        if not data.get("text"):
            errors.append("Clinical dictation text is required.")
        return errors

    def execute(self, data: dict[str, Any]) -> dict[str, Any]:
        text = data["text"]

        # ── Step 1 + 2: LLM structuring & coding ────────────────────────────
        prompt = SOAP_PROMPT.replace("{text}", text)
        messages = [
            {"role": "system", "content": "You are a clinical documentation and coding assistant."},
            {"role": "user", "content": prompt},
        ]

        resp = llm_completion(messages=messages, max_tokens=1500, temperature=0.2)
        raw = resp.choices[0].message.content.strip()

        try:
            clean = raw
            if "```" in clean:
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            soap = json.loads(clean)
        except (json.JSONDecodeError, IndexError):
            soap = {
                "subjective": raw, "objective": "Not documented",
                "assessment": "Not documented", "plan": "Not documented",
                "icd10_codes": [], "chief_complaint": "", "structured_notes": "",
            }

        icd_codes = soap.get("icd10_codes", [])

        # ── Step 3: Assemble output ──────────────────────────────────────────
        cc = soap.get("chief_complaint", "Not specified")
        summary = (
            f"SOAP note generated. Chief complaint: {cc}. "
            f"{len(icd_codes)} ICD-10 code(s) suggested."
        )

        report_lines = [
            "═══════════════════════════════════════",
            "       CLINICAL NOTE — SOAP FORMAT     ",
            "═══════════════════════════════════════",
            "",
            f"Chief Complaint: {cc}",
            "",
            "─── S: SUBJECTIVE ───",
            soap.get("subjective", "Not documented"),
            "",
            "─── O: OBJECTIVE ───",
            soap.get("objective", "Not documented"),
            "",
            "─── A: ASSESSMENT ───",
            soap.get("assessment", "Not documented"),
            "",
            "─── P: PLAN ───",
            soap.get("plan", "Not documented"),
            "",
        ]

        if icd_codes:
            report_lines.append("─── ICD-10 Codes ───")
            for code in icd_codes:
                conf = code.get("confidence", "")
                report_lines.append(
                    f"  {code.get('code', '?')} — {code.get('description', '?')}"
                    + (f"  [{conf}]" if conf else "")
                )
            report_lines.append("")

        notes = soap.get("structured_notes", "")
        if notes:
            report_lines.append("─── Additional Notes ───")
            report_lines.append(notes)

        return {
            "summary": summary,
            "metrics": {
                "chief_complaint": cc,
                "icd10_codes": ", ".join(c.get("code", "?") for c in icd_codes),
                "sections_complete": str(sum(1 for k in ("subjective", "objective", "assessment", "plan")
                                             if soap.get(k, "").lower() != "not documented")),
            },
            "protocol_adherence": True,
            "raw_output": "\n".join(report_lines),
        }
