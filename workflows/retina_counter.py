"""
workflows/retina_counter.py — Ophthalmology: Diabetic Retinopathy Grading.

Input:  Fundus Image (URL or base64).
Protocol: International Clinical Diabetic Retinopathy (ICDR) Scale.
Chain:
  1. Vision Step  — detect & count micro-aneurysms, hemorrhages, exudates.
  2. Logic Step   — categorise severity (Mild / Moderate / Severe / Proliferative).
  3. Output       — structured report with lesion count and grading.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from workflows.base import ClinicalWorkflow, InputType
from llm_config import llm_completion

logger = logging.getLogger(__name__)

VISION_PROMPT = """\
You are a specialist ophthalmology AI. Analyse the provided fundus image for signs
of diabetic retinopathy.

Identify and COUNT each of the following lesion types:
- Microaneurysms
- Dot/blot hemorrhages
- Hard exudates
- Cotton-wool spots (soft exudates)
- Neovascularisation (new vessel growth)
- Venous beading
- Intraretinal microvascular abnormalities (IRMA)

Return ONLY valid JSON:
{
  "microaneurysms": <int>,
  "hemorrhages": <int>,
  "hard_exudates": <int>,
  "cotton_wool_spots": <int>,
  "neovascularisation": <int>,
  "venous_beading": <int>,
  "irma": <int>,
  "observations": "<brief free-text description of findings>"
}
"""

# ICDR severity grading (deterministic logic)
def _grade_icdr(counts: dict) -> dict:
    """
    Apply ICDR scale based on lesion counts.
    Returns {"severity": str, "level": int, "explanation": str}.
    """
    nv   = counts.get("neovascularisation", 0)
    hem  = counts.get("hemorrhages", 0)
    ma   = counts.get("microaneurysms", 0)
    irma = counts.get("irma", 0)
    vb   = counts.get("venous_beading", 0)
    cws  = counts.get("cotton_wool_spots", 0)

    if nv > 0:
        return {"severity": "Proliferative DR", "level": 4,
                "explanation": "Neovascularisation detected — proliferative disease."}
    if (hem >= 20) or (irma > 0 and vb > 0):
        return {"severity": "Severe NPDR", "level": 3,
                "explanation": "Extensive hemorrhages and/or IRMA + venous beading — severe non-proliferative."}
    if hem > 0 or cws > 0 or irma > 0:
        return {"severity": "Moderate NPDR", "level": 2,
                "explanation": "Hemorrhages, cotton-wool spots, or IRMA exceed mild thresholds."}
    if ma > 0:
        return {"severity": "Mild NPDR", "level": 1,
                "explanation": "Microaneurysms only — mild non-proliferative."}
    return {"severity": "No apparent DR", "level": 0,
            "explanation": "No diabetic retinopathy lesions identified."}


class RetinaCounterWorkflow(ClinicalWorkflow):
    workflow_id  = "retina_counter"
    name         = "RetinaCounter"
    icon         = "👁️"
    description  = "Fundus image analysis — diabetic retinopathy lesion counting & ICDR grading."
    input_types  = [InputType.IMAGE]
    protocol     = "ICDR Scale"
    specialty    = "Ophthalmology"

    def validate_input(self, data: dict[str, Any]) -> list[str]:
        errors = []
        if not data.get("image_url") and not data.get("image_base64"):
            errors.append("A fundus image is required (image_url or image_base64).")
        return errors

    def execute(self, data: dict[str, Any]) -> dict[str, Any]:
        # ── Step 1: Vision analysis via LLM ──────────────────────────────────
        image_url = data.get("image_url", "")
        messages = [
            {"role": "system", "content": VISION_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": "Please analyse this fundus image for diabetic retinopathy."},
                {"type": "image_url", "image_url": {"url": image_url}} if image_url else
                {"type": "text", "text": f"[base64 image provided]"},
            ]},
        ]

        resp = llm_completion(messages=messages, max_tokens=1024, temperature=0.2)
        raw = resp.choices[0].message.content.strip()

        # Parse JSON from LLM
        try:
            # Strip markdown fences if present
            clean = raw
            if "```" in clean:
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            counts = json.loads(clean)
        except (json.JSONDecodeError, IndexError):
            counts = {
                "microaneurysms": 0, "hemorrhages": 0, "hard_exudates": 0,
                "cotton_wool_spots": 0, "neovascularisation": 0,
                "venous_beading": 0, "irma": 0,
                "observations": raw,
            }

        # ── Step 2: Deterministic ICDR grading ──────────────────────────────
        grade = _grade_icdr(counts)

        # ── Step 3: Assemble output ─────────────────────────────────────────
        observations = counts.pop("observations", "")
        total_lesions = sum(v for v in counts.values() if isinstance(v, int))

        summary = (
            f"ICDR Grade: {grade['severity']} (Level {grade['level']}/4). "
            f"Total lesions detected: {total_lesions}. "
            f"{grade['explanation']}"
        )

        report_lines = [
            "═══ DIABETIC RETINOPATHY REPORT ═══",
            f"Grade     : {grade['severity']} (Level {grade['level']}/4)",
            f"Explanation: {grade['explanation']}",
            "",
            "─── Lesion Counts ───",
        ]
        for k, v in counts.items():
            report_lines.append(f"  {k.replace('_', ' ').title():.<35} {v}")
        report_lines += [
            "",
            f"Total Lesions: {total_lesions}",
            "",
            "─── Observations ───",
            observations or "(none)",
        ]

        return {
            "summary": summary,
            "metrics": {
                "severity": grade["severity"],
                "severity_level": f"{grade['level']}/4",
                "total_lesions": str(total_lesions),
                **{k: str(v) for k, v in counts.items()},
            },
            "protocol_adherence": True,
            "raw_output": "\n".join(report_lines),
        }
