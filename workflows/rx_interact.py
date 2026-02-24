"""
workflows/rx_interact.py — Pharmacy: CSV Drug Interaction Checker.

Input:  CSV with medication list.
Protocol: Known drug-drug interaction database (curated subset).
Chain:
  1. Parse CSV  — extract drug names + optional dose/route.
  2. Pairwise check — deterministic O(n²) interaction lookup.
  3. Output    — interaction matrix, severity, clinical advice.

Expected CSV columns: drug | medication | name,  [dose], [route], [frequency]
"""

from __future__ import annotations

import csv
import io
import logging
import re
from typing import Any

from workflows.base import ClinicalWorkflow, InputType

logger = logging.getLogger(__name__)

# ── Interaction database (curated high-priority pairs) ────────────────────────
# Key: frozenset({drug_a, drug_b}) → dict(severity, effect, advice)
# Drug names stored lowercase.
_INTERACTIONS: dict[frozenset, dict] = {}


def _add(a: str, b: str, severity: str, effect: str, advice: str):
    _INTERACTIONS[frozenset({a.lower(), b.lower()})] = {
        "severity": severity, "effect": effect, "advice": advice,
    }


# ── HIGH SEVERITY (Contraindicated / Major) ──────────────────────────────────
_add("warfarin",     "aspirin",       "HIGH",     "Increased bleeding risk",
     "Avoid combination unless specifically indicated. Monitor INR closely.")
_add("warfarin",     "ibuprofen",     "HIGH",     "Increased bleeding risk, GI ulceration",
     "Avoid NSAIDs with warfarin. Use acetaminophen for pain if needed.")
_add("warfarin",     "naproxen",      "HIGH",     "Increased bleeding risk",
     "Avoid NSAIDs with anticoagulants.")
_add("warfarin",     "fluconazole",   "HIGH",     "Inhibits CYP2C9 → increased INR",
     "Reduce warfarin dose by 50%. Monitor INR within 3-5 days.")
_add("warfarin",     "metronidazole", "HIGH",     "Inhibits warfarin metabolism → increased INR",
     "Reduce warfarin dose. Monitor INR closely.")
_add("warfarin",     "amiodarone",    "HIGH",     "Inhibits CYP2C9/3A4 → increased INR",
     "Reduce warfarin dose by 30-50%. Monitor INR weekly.")
_add("methotrexate", "trimethoprim",  "HIGH",     "Additive anti-folate toxicity → pancytopenia",
     "Avoid combination. If unavoidable, monitor CBC weekly.")
_add("methotrexate", "nsaids",        "HIGH",     "Decreased renal clearance of methotrexate",
     "Avoid NSAIDs during methotrexate therapy. Risk of fatal toxicity.")
_add("lithium",      "ibuprofen",     "HIGH",     "NSAIDs decrease lithium clearance → toxicity",
     "Avoid combination. Monitor lithium levels if unavoidable.")
_add("lithium",      "lisinopril",    "HIGH",     "ACE inhibitors decrease lithium clearance",
     "Monitor lithium levels closely. Consider dose reduction.")
_add("lithium",      "furosemide",    "HIGH",     "Diuretic-induced sodium loss raises lithium levels",
     "Monitor lithium levels. Risk of lithium toxicity.")
_add("simvastatin",  "clarithromycin", "HIGH",    "Strong CYP3A4 inhibition → rhabdomyolysis risk",
     "Contraindicated. Use azithromycin instead, or hold statin.")
_add("simvastatin",  "itraconazole",  "HIGH",     "CYP3A4 inhibition → rhabdomyolysis",
     "Contraindicated. Hold statin during azole therapy.")
_add("digoxin",      "amiodarone",    "HIGH",     "Increased digoxin levels → toxicity",
     "Reduce digoxin dose by 50%. Monitor levels and ECG.")
_add("digoxin",      "verapamil",     "HIGH",     "Increased digoxin levels + AV block",
     "Reduce digoxin dose. Monitor HR and rhythm.")
_add("clopidogrel",  "omeprazole",    "HIGH",     "CYP2C19 inhibition → reduced clopidogrel efficacy",
     "Switch to pantoprazole (does not inhibit CYP2C19).")
_add("sildenafil",   "nitroglycerin", "HIGH",     "Severe hypotension → cardiovascular collapse",
     "CONTRAINDICATED. Wait ≥24h after sildenafil before nitrates.")
_add("maois",        "ssris",         "HIGH",     "Serotonin syndrome — potentially fatal",
     "CONTRAINDICATED. ≥14 day washout between MAOIs and SSRIs.")
_add("potassium",    "spironolactone", "HIGH",    "Hyperkalemia risk",
     "Avoid K+ supplements with K+-sparing diuretics. Monitor K+ levels.")

# ── MODERATE SEVERITY ─────────────────────────────────────────────────────────
_add("metformin",    "furosemide",    "MODERATE", "Furosemide may increase metformin levels",
     "Monitor blood glucose and renal function.")
_add("lisinopril",   "potassium",     "MODERATE", "ACE inhibitors + K+ → hyperkalemia",
     "Monitor potassium levels regularly.")
_add("lisinopril",   "spironolactone", "MODERATE","Dual RAAS blockade → hyperkalemia",
     "Monitor potassium and renal function closely.")
_add("amlodipine",   "simvastatin",   "MODERATE", "Increased statin exposure → myopathy risk",
     "Limit simvastatin to 20 mg/day with amlodipine.")
_add("metformin",    "contrast dye",  "MODERATE", "Risk of lactic acidosis",
     "Hold metformin 48h before/after iodinated contrast. Check eGFR.")
_add("ciprofloxacin", "antacids",     "MODERATE", "Reduced ciprofloxacin absorption",
     "Give ciprofloxacin 2h before or 6h after antacids.")
_add("levothyroxine", "calcium",      "MODERATE", "Calcium decreases levothyroxine absorption",
     "Separate by ≥4 hours.")
_add("levothyroxine", "iron",         "MODERATE", "Iron decreases levothyroxine absorption",
     "Separate by ≥4 hours.")
_add("ssris",        "tramadol",      "MODERATE", "Increased serotonin syndrome risk + seizures",
     "Use with caution. Monitor for serotonin syndrome symptoms.")
_add("aspirin",      "ibuprofen",     "MODERATE", "Ibuprofen blocks aspirin's antiplatelet effect",
     "Take aspirin ≥30 min before ibuprofen, or use acetaminophen instead.")
_add("metoprolol",   "verapamil",     "MODERATE", "Additive bradycardia and AV block",
     "Avoid combination. If needed, monitor HR and ECG closely.")
_add("prednisone",   "nsaids",        "MODERATE", "Increased GI ulceration risk",
     "Add PPI gastroprotection if combination necessary.")
_add("fluoxetine",   "tramadol",      "MODERATE", "Serotonin syndrome risk + reduced tramadol efficacy",
     "Avoid if possible. Monitor for serotonin syndrome.")
_add("carbamazepine", "erythromycin", "MODERATE",  "CYP3A4 inhibition → carbamazepine toxicity",
     "Monitor carbamazepine levels. Consider azithromycin instead.")

# ── LOW SEVERITY ──────────────────────────────────────────────────────────────
_add("metformin",    "alcohol",       "LOW",      "Increased lactic acidosis risk",
     "Advise moderate alcohol consumption.")
_add("antihypertensives", "nsaids",   "LOW",      "NSAIDs may blunt antihypertensive effect",
     "Monitor blood pressure. Prefer acetaminophen for pain.")
_add("insulin",      "beta_blockers", "LOW",      "Beta-blockers may mask hypoglycemia symptoms",
     "Educate patient on non-adrenergic hypoglycemia signs.")


# Build a set of all known drug names for fuzzy matching
_ALL_DRUG_NAMES: set[str] = set()
for pair in _INTERACTIONS:
    _ALL_DRUG_NAMES.update(pair)

# Common drug class members for class-level matching
_DRUG_CLASSES: dict[str, list[str]] = {
    "nsaids": ["ibuprofen", "naproxen", "diclofenac", "indomethacin", "ketorolac",
               "meloxicam", "celecoxib", "piroxicam", "aspirin"],
    "ssris": ["fluoxetine", "sertraline", "paroxetine", "citalopram",
              "escitalopram", "fluvoxamine"],
    "maois": ["phenelzine", "tranylcypromine", "isocarboxazid", "selegiline"],
    "antihypertensives": ["lisinopril", "enalapril", "losartan", "valsartan",
                          "amlodipine", "metoprolol", "atenolol", "hydrochlorothiazide"],
    "beta_blockers": ["metoprolol", "atenolol", "propranolol", "carvedilol", "bisoprolol"],
}


def _normalize_drug(name: str) -> list[str]:
    """Return list of canonical drug identifiers for interaction lookup."""
    n = name.strip().lower()
    n = re.sub(r"\s+\d+\s*mg.*$", "", n)  # strip dose info
    n = re.sub(r"\s+\(.*\)$", "", n)       # strip parenthetical
    n = n.strip()

    results = [n]
    # Check if this drug belongs to any class
    for class_name, members in _DRUG_CLASSES.items():
        if n in members:
            results.append(class_name)
    return results


def _parse_medication_csv(csv_text: str) -> list[dict]:
    """Parse CSV → list of medication dicts."""
    reader = csv.DictReader(io.StringIO(csv_text.strip()))
    if not reader.fieldnames:
        return []

    drug_col = next((f for f in reader.fieldnames if f.strip().lower() in
                     ("drug", "medication", "name", "med", "medicine", "rx", "drug_name")), None)
    dose_col = next((f for f in reader.fieldnames if f.strip().lower() in
                     ("dose", "dosage", "strength")), None)
    route_col = next((f for f in reader.fieldnames if f.strip().lower() in
                      ("route", "administration")), None)
    freq_col = next((f for f in reader.fieldnames if f.strip().lower() in
                     ("frequency", "freq", "schedule", "sig")), None)

    if not drug_col:
        cols = reader.fieldnames
        drug_col = cols[0] if cols else None

    meds = []
    for row in reader:
        name = (row.get(drug_col) or "").strip()
        if not name:
            continue
        meds.append({
            "drug": name,
            "dose": (row.get(dose_col) or "").strip() if dose_col else "",
            "route": (row.get(route_col) or "").strip() if route_col else "",
            "frequency": (row.get(freq_col) or "").strip() if freq_col else "",
        })
    return meds


def _check_interactions(meds: list[dict]) -> dict:
    """Pairwise check all medications for known interactions."""
    interactions_found = []
    drug_names = []

    for med in meds:
        drug_names.append({
            "original": med["drug"],
            "canonical": _normalize_drug(med["drug"]),
            "dose": med.get("dose", ""),
        })

    # O(n²) pairwise check
    for i in range(len(drug_names)):
        for j in range(i + 1, len(drug_names)):
            a_names = drug_names[i]["canonical"]
            b_names = drug_names[j]["canonical"]

            for a in a_names:
                for b in b_names:
                    pair = frozenset({a, b})
                    if pair in _INTERACTIONS:
                        info = _INTERACTIONS[pair]
                        interactions_found.append({
                            "drug_a": drug_names[i]["original"],
                            "drug_b": drug_names[j]["original"],
                            "severity": info["severity"],
                            "effect": info["effect"],
                            "advice": info["advice"],
                        })

    # Deduplicate (same pair may match on multiple name variants)
    seen = set()
    unique = []
    for ix in interactions_found:
        key = frozenset({ix["drug_a"].lower(), ix["drug_b"].lower(), ix["severity"]})
        if key not in seen:
            seen.add(key)
            unique.append(ix)

    # Sort by severity
    severity_order = {"HIGH": 0, "MODERATE": 1, "LOW": 2}
    unique.sort(key=lambda x: severity_order.get(x["severity"], 3))

    high = [x for x in unique if x["severity"] == "HIGH"]
    moderate = [x for x in unique if x["severity"] == "MODERATE"]
    low = [x for x in unique if x["severity"] == "LOW"]

    return {
        "interactions": unique,
        "total_meds": len(meds),
        "total_interactions": len(unique),
        "high_count": len(high),
        "moderate_count": len(moderate),
        "low_count": len(low),
        "high": high,
        "moderate": moderate,
        "low": low,
    }


class RxInteractWorkflow(ClinicalWorkflow):
    workflow_id  = "rx_interact"
    name         = "RxInteract"
    icon         = "💊"
    description  = "Upload a CSV medication list → detect drug-drug interactions with severity grading."
    input_types  = [InputType.CSV]
    protocol     = "Drug Interaction DB"
    specialty    = "Pharmacy"

    def validate_input(self, data: dict[str, Any]) -> list[str]:
        errors = []
        if not data.get("csv_text"):
            errors.append("CSV data is required. Upload a CSV with medication list.")
        return errors

    def execute(self, data: dict[str, Any]) -> dict[str, Any]:
        csv_text = data["csv_text"]

        # ── Step 1: Parse CSV ────────────────────────────────────────────
        meds = _parse_medication_csv(csv_text)
        if not meds:
            return {
                "summary": "No medications found in CSV. Expected column: drug, medication, or name.",
                "metrics": {},
                "protocol_adherence": False,
                "raw_output": f"Input CSV ({len(csv_text)} chars), 0 medications parsed.",
            }

        # ── Step 2: Pairwise interaction check ───────────────────────────
        result = _check_interactions(meds)

        # ── Step 3: Assemble output ──────────────────────────────────────
        summary_parts = [f"{result['total_meds']} medications analyzed."]
        if result["total_interactions"] == 0:
            summary_parts.append("No known interactions detected.")
        else:
            summary_parts.append(f"{result['total_interactions']} interaction(s) found.")
            if result["high_count"]:
                summary_parts.append(f"⚠ {result['high_count']} HIGH severity!")
        summary = " ".join(summary_parts)

        report_lines = [
            "═══ DRUG INTERACTION REPORT ═══",
            "",
            f"Medications: {result['total_meds']}",
            f"Interactions: {result['total_interactions']} "
            f"(HIGH: {result['high_count']}, MODERATE: {result['moderate_count']}, "
            f"LOW: {result['low_count']})",
            "",
            "─── Medication List ───",
        ]
        for med in meds:
            line = f"  • {med['drug']}"
            if med.get("dose"):
                line += f" {med['dose']}"
            if med.get("frequency"):
                line += f" {med['frequency']}"
            report_lines.append(line)

        if result["high"]:
            report_lines += ["", "⚠⚠⚠ HIGH SEVERITY INTERACTIONS ⚠⚠⚠"]
            for ix in result["high"]:
                report_lines += [
                    f"  {ix['drug_a']} ↔ {ix['drug_b']}",
                    f"    Effect: {ix['effect']}",
                    f"    Action: {ix['advice']}",
                    "",
                ]

        if result["moderate"]:
            report_lines += ["─── Moderate Interactions ───"]
            for ix in result["moderate"]:
                report_lines += [
                    f"  {ix['drug_a']} ↔ {ix['drug_b']}",
                    f"    Effect: {ix['effect']}",
                    f"    Action: {ix['advice']}",
                    "",
                ]

        if result["low"]:
            report_lines += ["─── Low Interactions ───"]
            for ix in result["low"]:
                report_lines += [
                    f"  {ix['drug_a']} ↔ {ix['drug_b']}  — {ix['effect']}",
                    f"    {ix['advice']}",
                ]

        if result["total_interactions"] == 0:
            report_lines += ["", "✓ No known interactions found in the current database."]

        return {
            "summary": summary,
            "metrics": {
                "medications": str(result["total_meds"]),
                "interactions": str(result["total_interactions"]),
                "high_severity": str(result["high_count"]),
                "moderate_severity": str(result["moderate_count"]),
            },
            "protocol_adherence": True,
            "raw_output": "\n".join(report_lines),
        }
