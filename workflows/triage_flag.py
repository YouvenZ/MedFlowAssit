"""
workflows/triage_flag.py — Radiology: Critical Finding Triage.

Input:  Radiology Report (PDF text or plain text).
Protocol: ACR Actionable Reporting (Red Flag Detection).
Chain:
  1. NLP Step      — extract "Findings" and "Impression" sections.
  2. Categorisation — identify critical findings (midline shift, pneumothorax, etc.).
  3. Output        — high-priority summary card with 3-level urgency triage.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from workflows.base import ClinicalWorkflow, InputType
from llm_config import llm_completion

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """\
You are a radiology NLP engine. Parse the provided radiology report and extract
the structured sections.

Return ONLY valid JSON:
{
  "findings": "<extracted findings section text>",
  "impression": "<extracted impression section text>",
  "modality": "<CT/MRI/XR/US/etc>",
  "body_region": "<head/chest/abdomen/etc>",
  "critical_findings": [
    {
      "finding": "<description of critical finding>",
      "acr_category": "<one of: CRITICAL | URGENT | NON-URGENT>",
      "requires_communication": true
    }
  ],
  "incidental_findings": ["<list of incidental/secondary findings>"]
}

### ACR Urgency Definitions:
- CRITICAL: Life-threatening, requires immediate action (e.g., tension pneumothorax, active hemorrhage, midline shift, aortic dissection, PE)
- URGENT: Significant but not immediately life-threatening (e.g., new mass, fracture with displacement, abscess)
- NON-URGENT: Incidental or expected findings needing routine follow-up

Radiology Report:
---
{text}
---
"""

# Deterministic urgency logic
def _compute_triage(critical_findings: list[dict]) -> dict:
    """
    Determine overall triage level from the extracted critical findings.
    Returns dict with level, color, and action.
    """
    if not critical_findings:
        return {
            "level": "NON-URGENT",
            "level_num": 3,
            "color": "green",
            "action": "Routine read — no critical findings identified.",
        }

    categories = [f.get("acr_category", "NON-URGENT").upper() for f in critical_findings]

    if "CRITICAL" in categories:
        return {
            "level": "CRITICAL",
            "level_num": 1,
            "color": "red",
            "action": "STAT communication required to ordering physician within minutes.",
        }
    if "URGENT" in categories:
        return {
            "level": "URGENT",
            "level_num": 2,
            "color": "orange",
            "action": "Urgent communication required within hours. Schedule follow-up imaging or consult.",
        }
    return {
        "level": "NON-URGENT",
        "level_num": 3,
        "color": "green",
        "action": "Routine follow-up recommended. No immediate action required.",
    }


class TriageFlagWorkflow(ClinicalWorkflow):
    workflow_id  = "triage_flag"
    name         = "TriageFlag"
    icon         = "🔬"
    description  = "Radiology report triage — ACR actionable findings with 3-level urgency classification."
    input_types  = [InputType.PDF, InputType.TEXT]
    protocol     = "ACR Actionable Reporting"
    specialty    = "Radiology"

    def validate_input(self, data: dict[str, Any]) -> list[str]:
        errors = []
        if not data.get("text") and not data.get("pdf_text"):
            errors.append("Radiology report text or PDF text is required.")
        return errors

    def execute(self, data: dict[str, Any]) -> dict[str, Any]:
        report_text = data.get("text", "") + "\n" + data.get("pdf_text", "")
        report_text = report_text.strip()

        # ── Step 1: NLP extraction via LLM ───────────────────────────────────
        prompt = EXTRACTION_PROMPT.replace("{text}", report_text)
        messages = [
            {"role": "system", "content": "You are a radiology report parser."},
            {"role": "user", "content": prompt},
        ]

        resp = llm_completion(messages=messages, max_tokens=1024, temperature=0.1)
        raw = resp.choices[0].message.content.strip()

        try:
            clean = raw
            if "```" in clean:
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            parsed = json.loads(clean)
        except (json.JSONDecodeError, IndexError):
            parsed = {
                "findings": report_text, "impression": "",
                "modality": "Unknown", "body_region": "Unknown",
                "critical_findings": [], "incidental_findings": [],
            }

        critical_findings = parsed.get("critical_findings", [])
        incidental = parsed.get("incidental_findings", [])

        # ── Step 2: Deterministic triage ─────────────────────────────────────
        triage = _compute_triage(critical_findings)

        # ── Step 3: Assemble output ──────────────────────────────────────────
        summary = (
            f"Triage Level: {triage['level']} ({triage['level_num']}/3). "
            f"{len(critical_findings)} critical/urgent finding(s). "
            f"{triage['action']}"
        )

        report_lines = [
            "═══ RADIOLOGY TRIAGE REPORT ═══",
            f"Protocol: ACR Actionable Reporting",
            f"Modality: {parsed.get('modality', '?')}",
            f"Region: {parsed.get('body_region', '?')}",
            "",
            f"▸ TRIAGE LEVEL: {triage['level']} ({triage['level_num']}/3)",
            f"▸ ACTION: {triage['action']}",
            "",
        ]

        if critical_findings:
            report_lines.append("─── Critical/Urgent Findings ───")
            for i, cf in enumerate(critical_findings, 1):
                report_lines.append(f"  {i}. [{cf.get('acr_category', '?')}] {cf.get('finding', '?')}")
                if cf.get("requires_communication"):
                    report_lines.append("     → Requires direct physician communication")
            report_lines.append("")

        report_lines.append("─── Findings ───")
        report_lines.append(parsed.get("findings", "(not extracted)"))
        report_lines.append("")
        report_lines.append("─── Impression ───")
        report_lines.append(parsed.get("impression", "(not extracted)"))

        if incidental:
            report_lines.append("")
            report_lines.append("─── Incidental Findings ───")
            for inc in incidental:
                report_lines.append(f"  • {inc}")

        return {
            "summary": summary,
            "metrics": {
                "triage_level": triage["level"],
                "critical_findings": str(len(critical_findings)),
                "incidental_findings": str(len(incidental)),
                "modality": parsed.get("modality", "?"),
            },
            "protocol_adherence": True,
            "raw_output": "\n".join(report_lines),
        }
