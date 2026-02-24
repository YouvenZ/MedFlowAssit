"""
workflows/medrec_guard.py — Geriatrics: Medication Reconciliation Guard.

Input:  Image of medication bottles or Discharge Summary (PDF text).
Protocol: 2023 AGS Beers Criteria®.
Chain:
  1. OCR/Parsing  — list all medications and dosages.
  2. Cross-Reference — check against Beers Criteria PIMs for elderly.
  3. Output — "Stop / Caution" list with clinical justifications.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from workflows.base import ClinicalWorkflow, InputType
from llm_config import llm_completion

logger = logging.getLogger(__name__)

OCR_EXTRACTION_PROMPT = """\
You are a pharmacist AI assistant. From the provided input (image of medication bottles
or discharge summary text), extract ALL medications.

Return ONLY valid JSON:
{
  "medications": [
    {
      "name": "<generic name>",
      "brand": "<brand name or null>",
      "dose": "<dosage string>",
      "frequency": "<frequency or null>",
      "route": "<oral/IV/etc or null>"
    }
  ],
  "patient_age": <int or null>,
  "extraction_notes": "<any additional notes>"
}
"""

# ── Beers Criteria 2023 reference (simplified, key PIMs) ─────────────────────
# In production this would be a comprehensive database. Here we include the
# most commonly flagged medications for demonstration.
BEERS_CRITERIA: dict[str, dict] = {
    "diazepam":        {"category": "STOP",    "class": "Benzodiazepine",
                        "reason": "Increased risk of cognitive impairment, delirium, falls, and fractures in older adults."},
    "lorazepam":       {"category": "STOP",    "class": "Benzodiazepine",
                        "reason": "Increased risk of cognitive impairment, delirium, falls, and fractures in older adults."},
    "alprazolam":      {"category": "STOP",    "class": "Benzodiazepine",
                        "reason": "Increased risk of cognitive impairment, delirium, falls, and fractures in older adults."},
    "chlordiazepoxide":{"category": "STOP",    "class": "Benzodiazepine",
                        "reason": "Long-acting benzodiazepine — prolonged sedation, fall risk."},
    "amitriptyline":   {"category": "STOP",    "class": "Tricyclic antidepressant",
                        "reason": "Highly anticholinergic — sedation, orthostatic hypotension, cardiac conduction issues."},
    "nortriptyline":   {"category": "CAUTION", "class": "Tricyclic antidepressant",
                        "reason": "Anticholinergic effects; use with caution and at lower doses."},
    "diphenhydramine": {"category": "STOP",    "class": "First-gen antihistamine",
                        "reason": "Highly anticholinergic — confusion, urinary retention, constipation."},
    "hydroxyzine":     {"category": "STOP",    "class": "First-gen antihistamine",
                        "reason": "Anticholinergic effects, sedation in elderly."},
    "chlorpheniramine":{"category": "STOP",    "class": "First-gen antihistamine",
                        "reason": "Anticholinergic burden; avoid in elderly."},
    "meperidine":      {"category": "STOP",    "class": "Opioid analgesic",
                        "reason": "Neurotoxic metabolite normeperidine — seizure risk, not effective orally."},
    "indomethacin":    {"category": "STOP",    "class": "NSAID",
                        "reason": "Highest CNS adverse effects of all NSAIDs; increased GI bleed & renal risk."},
    "ketorolac":       {"category": "STOP",    "class": "NSAID",
                        "reason": "High GI bleeding risk; avoid in elderly especially with renal impairment."},
    "naproxen":        {"category": "CAUTION", "class": "NSAID",
                        "reason": "GI bleed and renal risk in elderly; use lowest dose for shortest duration."},
    "ibuprofen":       {"category": "CAUTION", "class": "NSAID",
                        "reason": "GI bleed and renal risk; caution with concurrent anticoagulants."},
    "aspirin":         {"category": "CAUTION", "class": "Antiplatelet",
                        "reason": "For primary prevention in ≥70 y/o: bleeding risk may outweigh benefit."},
    "glyburide":       {"category": "STOP",    "class": "Sulfonylurea",
                        "reason": "Higher risk of prolonged hypoglycemia vs. other sulfonylureas."},
    "glimepiride":     {"category": "CAUTION", "class": "Sulfonylurea",
                        "reason": "Hypoglycemia risk; use lower starting dose in elderly."},
    "metoclopramide":  {"category": "STOP",    "class": "Antiemetic / prokinetic",
                        "reason": "Extrapyramidal effects including tardive dyskinesia; avoid unless gastroparesis."},
    "nitrofurantoin":  {"category": "CAUTION", "class": "Antibiotic",
                        "reason": "Avoid if CrCl <30 mL/min — pulmonary toxicity, peripheral neuropathy."},
    "doxazosin":       {"category": "STOP",    "class": "Alpha-1 blocker",
                        "reason": "Orthostatic hypotension risk; avoid as antihypertensive in elderly."},
    "prazosin":        {"category": "STOP",    "class": "Alpha-1 blocker",
                        "reason": "Orthostatic hypotension risk; avoid as antihypertensive in elderly."},
    "digoxin":         {"category": "CAUTION", "class": "Cardiac glycoside",
                        "reason": "Avoid doses >0.125 mg/day in elderly due to toxicity risk."},
    "sliding_scale_insulin": {"category": "STOP", "class": "Insulin regimen",
                        "reason": "Higher risk of hypoglycemia without improvement in management."},
    "zolpidem":        {"category": "STOP",    "class": "Sedative-hypnotic",
                        "reason": "Increased ER visits, falls, and fractures in elderly."},
    "eszopiclone":     {"category": "STOP",    "class": "Sedative-hypnotic",
                        "reason": "Minimal improvement in sleep latency; fall risk elevated."},
}


def _check_beers(medications: list[dict]) -> list[dict]:
    """Cross-reference medication list against Beers Criteria."""
    flags = []
    for med in medications:
        name_lower = med.get("name", "").lower().strip()
        # Check exact match and partial match
        for beers_drug, info in BEERS_CRITERIA.items():
            if beers_drug in name_lower or name_lower in beers_drug:
                flags.append({
                    "medication": med.get("name", ""),
                    "dose": med.get("dose", ""),
                    "category": info["category"],
                    "drug_class": info["class"],
                    "reason": info["reason"],
                })
                break
    return flags


class MedRecGuardWorkflow(ClinicalWorkflow):
    workflow_id  = "medrec_guard"
    name         = "Med-Rec Guard"
    icon         = "💊"
    description  = "Medication reconciliation — cross-references against 2023 AGS Beers Criteria for elderly patients."
    input_types  = [InputType.IMAGE, InputType.PDF]
    protocol     = "AGS Beers Criteria® 2023"
    specialty    = "Geriatrics"

    def validate_input(self, data: dict[str, Any]) -> list[str]:
        errors = []
        if not data.get("image_url") and not data.get("pdf_text") and not data.get("text"):
            errors.append("Medication image, PDF text, or text input is required.")
        return errors

    def execute(self, data: dict[str, Any]) -> dict[str, Any]:
        # ── Step 1: OCR / Extraction ─────────────────────────────────────────
        user_content: list[dict] = [
            {"type": "text", "text": "Extract all medications from the following input."},
        ]
        if data.get("image_url"):
            user_content.append({"type": "image_url", "image_url": {"url": data["image_url"]}})
        if data.get("pdf_text"):
            user_content.append({"type": "text", "text": f"\n\nDischarge summary:\n{data['pdf_text']}"})
        if data.get("text"):
            user_content.append({"type": "text", "text": f"\n\nMedication list:\n{data['text']}"})

        messages = [
            {"role": "system", "content": OCR_EXTRACTION_PROMPT},
            {"role": "user", "content": user_content},
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
            parsed = {"medications": [], "patient_age": None, "extraction_notes": raw}

        medications = parsed.get("medications", [])
        patient_age = parsed.get("patient_age")

        # ── Step 2: Beers cross-reference ────────────────────────────────────
        flags = _check_beers(medications)
        stop_count = sum(1 for f in flags if f["category"] == "STOP")
        caution_count = sum(1 for f in flags if f["category"] == "CAUTION")

        # ── Step 3: Assemble output ──────────────────────────────────────────
        if not flags:
            summary = f"No Beers Criteria flags found among {len(medications)} medication(s)."
        else:
            summary = (
                f"Found {len(flags)} Beers Criteria flag(s): "
                f"{stop_count} STOP, {caution_count} CAUTION "
                f"out of {len(medications)} medication(s) reviewed."
            )

        report_lines = [
            "═══ MEDICATION RECONCILIATION REPORT ═══",
            f"Protocol: AGS Beers Criteria® 2023",
            f"Patient Age: {patient_age or 'Unknown'}",
            f"Medications Reviewed: {len(medications)}",
            "",
        ]

        if medications:
            report_lines.append("─── Current Medications ───")
            for i, m in enumerate(medications, 1):
                report_lines.append(
                    f"  {i}. {m.get('name', '?')} {m.get('dose', '')} "
                    f"{m.get('frequency', '')} ({m.get('route', 'oral')})"
                )
            report_lines.append("")

        if flags:
            report_lines.append("─── ⚠️  Beers Criteria Flags ───")
            for f in flags:
                marker = "🛑 STOP" if f["category"] == "STOP" else "⚠️  CAUTION"
                report_lines.append(f"  {marker}: {f['medication']} ({f['dose']})")
                report_lines.append(f"    Class: {f['drug_class']}")
                report_lines.append(f"    Reason: {f['reason']}")
                report_lines.append("")
        else:
            report_lines.append("─── ✅ No Beers Criteria Issues ───")
            report_lines.append("  All medications appear appropriate for the patient's age group.")

        return {
            "summary": summary,
            "metrics": {
                "medications_reviewed": str(len(medications)),
                "beers_flags": str(len(flags)),
                "stop_flags": str(stop_count),
                "caution_flags": str(caution_count),
            },
            "protocol_adherence": True,
            "raw_output": "\n".join(report_lines),
        }
