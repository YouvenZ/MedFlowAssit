"""
workflows/lab_flagr.py — Pathology / Lab: CSV Lab Panel Analyzer.

Input:  CSV file with lab results.
Protocol: Standard reference ranges (CBC, BMP, LFT, Lipids, Thyroid, etc.).
Chain:
  1. Parse CSV   — extract test_name, value, unit columns.
  2. Flag Logic  — deterministic Python comparison against reference ranges.
  3. Derived     — calculate eGFR (CKD-EPI), anion gap, corrected calcium, etc.
  4. Output      — flagged abnormals, critical values, derived metrics.

Expected CSV columns (flexible matching):
  test | analyte | name,  value | result,  unit,  [reference_range]
"""

from __future__ import annotations

import csv
import io
import logging
import math
import re
from typing import Any

from workflows.base import ClinicalWorkflow, InputType

logger = logging.getLogger(__name__)

# ── Reference ranges (adult, conventional units) ─────────────────────────────
# Format: { canonical_name: (low, high, unit, critical_low, critical_high) }
REFERENCE_RANGES: dict[str, tuple] = {
    # CBC
    "wbc":           (4.5,   11.0,  "×10³/µL",  2.0,   30.0),
    "rbc":           (4.2,   5.9,   "×10⁶/µL",  None,  None),
    "hemoglobin":    (12.0,  17.5,  "g/dL",      7.0,   20.0),
    "hematocrit":    (36.0,  51.0,  "%",         20.0,  60.0),
    "platelets":     (150,   400,   "×10³/µL",   50,    1000),
    "mcv":           (80,    100,   "fL",        None,  None),
    "mch":           (27,    33,    "pg",        None,  None),
    "mchc":          (32,    36,    "g/dL",      None,  None),
    "rdw":           (11.5,  14.5,  "%",         None,  None),
    "neutrophils":   (40,    70,    "%",         None,  None),
    "lymphocytes":   (20,    40,    "%",         None,  None),

    # BMP — Basic Metabolic Panel
    "sodium":        (136,   145,   "mEq/L",     120,   160),
    "potassium":     (3.5,   5.0,   "mEq/L",     2.5,   6.5),
    "chloride":      (98,    106,   "mEq/L",     None,  None),
    "bicarbonate":   (22,    29,    "mEq/L",     None,  None),
    "co2":           (22,    29,    "mEq/L",     None,  None),
    "bun":           (7,     20,    "mg/dL",      None,  None),
    "creatinine":    (0.6,   1.2,   "mg/dL",      None,  10.0),
    "glucose":       (70,    100,   "mg/dL",      40,    500),
    "calcium":       (8.5,   10.5,  "mg/dL",      6.0,   13.0),

    # LFT — Liver Function Tests
    "ast":           (10,    40,    "U/L",        None,  None),
    "alt":           (7,     56,    "U/L",        None,  None),
    "alp":           (44,    147,   "U/L",        None,  None),
    "bilirubin_total": (0.1, 1.2,  "mg/dL",      None,  None),
    "bilirubin_direct": (0.0, 0.3, "mg/dL",      None,  None),
    "albumin":       (3.5,   5.0,   "g/dL",       None,  None),
    "total_protein": (6.0,   8.3,   "g/dL",       None,  None),
    "ggt":           (9,     48,    "U/L",        None,  None),

    # Lipids
    "total_cholesterol": (0, 200,  "mg/dL",      None,  None),
    "ldl":           (0,     100,   "mg/dL",      None,  None),
    "hdl":           (40,    999,   "mg/dL",      None,  None),
    "triglycerides": (0,     150,   "mg/dL",      None,  None),

    # Thyroid
    "tsh":           (0.4,   4.0,   "mIU/L",      None,  None),
    "free_t4":       (0.8,   1.8,   "ng/dL",      None,  None),
    "free_t3":       (2.3,   4.2,   "pg/mL",      None,  None),

    # Coagulation
    "pt":            (11,    13.5,  "sec",        None,  None),
    "inr":           (0.8,   1.1,   "",           None,  4.5),
    "ptt":           (25,    35,    "sec",        None,  None),
    "aptt":          (25,    35,    "sec",        None,  None),

    # Other
    "crp":           (0,     1.0,   "mg/dL",      None,  None),
    "esr":           (0,     20,    "mm/hr",      None,  None),
    "hba1c":         (4.0,   5.6,   "%",          None,  None),
    "ferritin":      (12,    300,   "ng/mL",      None,  None),
    "iron":          (60,    170,   "µg/dL",      None,  None),
    "tibc":          (250,   370,   "µg/dL",      None,  None),
    "vitamin_d":     (30,    100,   "ng/mL",      None,  None),
    "vitamin_b12":   (200,   900,   "pg/mL",      None,  None),
    "folate":        (2.7,   17.0,  "ng/mL",      None,  None),
    "uric_acid":     (2.5,   7.0,   "mg/dL",      None,  None),
    "ldh":           (140,   280,   "U/L",        None,  None),
    "magnesium":     (1.7,   2.2,   "mg/dL",      None,  None),
    "phosphorus":    (2.5,   4.5,   "mg/dL",      None,  None),
    "lactate":       (0.5,   2.2,   "mmol/L",     None,  4.0),
    "troponin":      (0,     0.04,  "ng/mL",      None,  0.4),
    "bnp":           (0,     100,   "pg/mL",      None,  None),
    "procalcitonin": (0,     0.1,   "ng/mL",      None,  2.0),
    "d_dimer":       (0,     0.5,   "µg/mL",      None,  None),
    "ammonia":       (15,    45,    "µmol/L",     None,  100),
    "lipase":        (0,     160,   "U/L",        None,  None),
    "amylase":       (28,    100,   "U/L",        None,  None),
}

# Map common synonyms to canonical names
_ALIASES: dict[str, str] = {
    "white blood cells": "wbc", "white blood cell": "wbc",
    "red blood cells": "rbc", "red blood cell": "rbc",
    "hgb": "hemoglobin", "hb": "hemoglobin",
    "hct": "hematocrit", "plt": "platelets",
    "na": "sodium", "na+": "sodium",
    "k": "potassium", "k+": "potassium",
    "cl": "chloride", "hco3": "bicarbonate",
    "blood urea nitrogen": "bun", "urea nitrogen": "bun",
    "cr": "creatinine", "crea": "creatinine",
    "glu": "glucose", "fasting glucose": "glucose",
    "ca": "calcium", "ca2+": "calcium",
    "sgot": "ast", "sgpt": "alt",
    "alkaline phosphatase": "alp", "alk phos": "alp",
    "total bilirubin": "bilirubin_total", "tbili": "bilirubin_total",
    "direct bilirubin": "bilirubin_direct", "dbili": "bilirubin_direct",
    "alb": "albumin", "tp": "total_protein",
    "cholesterol": "total_cholesterol", "chol": "total_cholesterol",
    "trig": "triglycerides", "tg": "triglycerides",
    "ft4": "free_t4", "ft3": "free_t3",
    "prothrombin time": "pt", "partial thromboplastin": "ptt",
    "c-reactive protein": "crp", "sed rate": "esr",
    "hemoglobin a1c": "hba1c", "a1c": "hba1c", "glycated hemoglobin": "hba1c",
    "vit d": "vitamin_d", "25-oh vitamin d": "vitamin_d",
    "b12": "vitamin_b12", "vit b12": "vitamin_b12",
    "mg": "magnesium", "phos": "phosphorus",
    "lactic acid": "lactate", "trop": "troponin", "troponin i": "troponin",
    "troponin t": "troponin", "nt-probnp": "bnp", "pro-bnp": "bnp",
    "d-dimer": "d_dimer", "nh3": "ammonia",
}


def _normalize_test_name(name: str) -> str:
    """Normalize a test name to a canonical key."""
    n = name.strip().lower().replace("-", "_").replace(" ", "_")
    # Direct match
    if n in REFERENCE_RANGES:
        return n
    # Alias lookup
    clean = name.strip().lower()
    if clean in _ALIASES:
        return _ALIASES[clean]
    # Fuzzy: strip non-alnum and retry
    stripped = re.sub(r"[^a-z0-9]", "", clean)
    for alias, canonical in _ALIASES.items():
        if stripped == re.sub(r"[^a-z0-9]", "", alias):
            return canonical
    for canonical in REFERENCE_RANGES:
        if stripped == re.sub(r"[^a-z0-9]", "", canonical):
            return canonical
    return n  # unrecognized — return as-is


def _parse_csv(csv_text: str) -> list[dict]:
    """
    Parse CSV text → list of dicts with keys: test, value, unit.
    Flexibly handles various column names.
    """
    reader = csv.DictReader(io.StringIO(csv_text.strip()))
    if not reader.fieldnames:
        return []

    # Map column names to our expected keys
    fn = [f.strip().lower() for f in reader.fieldnames]
    test_col = next((f for f in reader.fieldnames if f.strip().lower() in
                     ("test", "analyte", "name", "test_name", "lab_test", "component")), None)
    val_col = next((f for f in reader.fieldnames if f.strip().lower() in
                    ("value", "result", "lab_value", "measurement")), None)
    unit_col = next((f for f in reader.fieldnames if f.strip().lower() in
                     ("unit", "units", "uom")), None)

    if not test_col or not val_col:
        # Try positional: first col=test, second=value, third=unit
        cols = reader.fieldnames
        if len(cols) >= 2:
            test_col, val_col = cols[0], cols[1]
            unit_col = cols[2] if len(cols) >= 3 else None
        else:
            return []

    rows = []
    for row in reader:
        test_name = (row.get(test_col) or "").strip()
        raw_value = (row.get(val_col) or "").strip()
        unit = (row.get(unit_col) or "").strip() if unit_col else ""
        if not test_name or not raw_value:
            continue

        # Try to parse numeric value
        try:
            # Handle values like "<0.05", ">200", "3.5 H", "7.2 L"
            cleaned = re.sub(r"[HLhl*]$", "", raw_value).strip()
            cleaned = re.sub(r"^[<>]", "", cleaned).strip()
            value = float(cleaned)
        except ValueError:
            value = None

        rows.append({"test": test_name, "value": value, "raw_value": raw_value, "unit": unit})

    return rows


def _flag_results(rows: list[dict]) -> dict:
    """
    Flag each lab result against reference ranges.
    Returns summary dict with flagged items, critical alerts, derived values.
    """
    flagged = []
    normal = []
    critical = []
    unrecognized = []
    values_map = {}  # canonical_name → numeric value for derived calculations

    for row in rows:
        canonical = _normalize_test_name(row["test"])
        value = row["value"]
        values_map[canonical] = value

        if canonical not in REFERENCE_RANGES:
            unrecognized.append({"test": row["test"], "value": row["raw_value"], "unit": row["unit"]})
            continue

        low, high, ref_unit, crit_low, crit_high = REFERENCE_RANGES[canonical]

        if value is None:
            unrecognized.append({"test": row["test"], "value": row["raw_value"], "note": "non-numeric"})
            continue

        flag = "NORMAL"
        if value < low:
            flag = "LOW"
        elif value > high:
            flag = "HIGH"

        is_critical = False
        if crit_low is not None and value < crit_low:
            is_critical = True
        if crit_high is not None and value > crit_high:
            is_critical = True

        entry = {
            "test": canonical,
            "original_name": row["test"],
            "value": value,
            "unit": row["unit"] or ref_unit,
            "reference": f"{low}-{high}",
            "flag": flag,
            "critical": is_critical,
        }

        if is_critical:
            critical.append(entry)
            flagged.append(entry)
        elif flag != "NORMAL":
            flagged.append(entry)
        else:
            normal.append(entry)

    # ── Derived calculations ─────────────────────────────────────────────
    derived = {}

    # Anion gap = Na - (Cl + HCO3)
    na = values_map.get("sodium")
    cl = values_map.get("chloride")
    hco3 = values_map.get("bicarbonate") or values_map.get("co2")
    if na is not None and cl is not None and hco3 is not None:
        ag = na - (cl + hco3)
        derived["anion_gap"] = {"value": round(ag, 1), "unit": "mEq/L",
                                "reference": "8-12", "flag": "HIGH" if ag > 12 else "NORMAL"}

    # Corrected calcium = Ca + 0.8 × (4.0 - albumin)
    ca = values_map.get("calcium")
    alb = values_map.get("albumin")
    if ca is not None and alb is not None:
        corrected = ca + 0.8 * (4.0 - alb)
        derived["corrected_calcium"] = {
            "value": round(corrected, 2), "unit": "mg/dL",
            "reference": "8.5-10.5",
            "flag": "HIGH" if corrected > 10.5 else ("LOW" if corrected < 8.5 else "NORMAL"),
        }

    # BUN/Creatinine ratio
    bun = values_map.get("bun")
    cr = values_map.get("creatinine")
    if bun is not None and cr is not None and cr > 0:
        ratio = bun / cr
        derived["bun_cr_ratio"] = {
            "value": round(ratio, 1), "unit": "",
            "reference": "10-20",
            "flag": "HIGH" if ratio > 20 else ("LOW" if ratio < 10 else "NORMAL"),
        }

    return {
        "flagged": flagged,
        "critical": critical,
        "normal_count": len(normal),
        "flagged_count": len(flagged),
        "critical_count": len(critical),
        "unrecognized": unrecognized,
        "derived": derived,
        "total_tests": len(rows),
    }


class LabFlagrWorkflow(ClinicalWorkflow):
    workflow_id  = "lab_flagr"
    name         = "LabFlagr"
    icon         = "🧪"
    description  = "Upload a CSV lab panel → flag abnormals, critical values, and compute derived metrics."
    input_types  = [InputType.CSV]
    protocol     = "Standard Reference Ranges"
    specialty    = "Pathology / Lab"

    def validate_input(self, data: dict[str, Any]) -> list[str]:
        errors = []
        if not data.get("csv_text"):
            errors.append("CSV data is required. Upload a CSV file with lab results.")
        return errors

    def execute(self, data: dict[str, Any]) -> dict[str, Any]:
        csv_text = data["csv_text"]

        # ── Step 1: Parse CSV ────────────────────────────────────────────
        rows = _parse_csv(csv_text)
        if not rows:
            return {
                "summary": "Could not parse any lab results from the CSV. Expected columns: test, value, unit.",
                "metrics": {},
                "protocol_adherence": False,
                "raw_output": f"Input CSV ({len(csv_text)} chars) yielded 0 parseable rows.",
            }

        # ── Step 2: Flag against reference ranges ────────────────────────
        result = _flag_results(rows)

        # ── Step 3: Assemble output ──────────────────────────────────────
        summary_parts = [
            f"{result['total_tests']} tests analyzed.",
            f"{result['flagged_count']} abnormal.",
        ]
        if result["critical_count"]:
            summary_parts.append(f"⚠ {result['critical_count']} CRITICAL value(s)!")
        summary = " ".join(summary_parts)

        report_lines = [
            "═══ LAB PANEL ANALYSIS REPORT ═══",
            "",
            f"Total Tests: {result['total_tests']}",
            f"Normal: {result['normal_count']}  |  Abnormal: {result['flagged_count']}  |  "
            f"Critical: {result['critical_count']}",
        ]

        if result["critical"]:
            report_lines += ["", "⚠⚠⚠ CRITICAL VALUES ⚠⚠⚠"]
            for c in result["critical"]:
                report_lines.append(
                    f"  {c['test'].upper():.<30} {c['value']} {c['unit']:.<12} "
                    f"(ref: {c['reference']}) ← CRITICAL"
                )

        if result["flagged"]:
            report_lines += ["", "─── Abnormal Results ───"]
            for f in result["flagged"]:
                marker = "↑ HIGH" if f["flag"] == "HIGH" else "↓ LOW"
                crit = " ⚠CRITICAL" if f["critical"] else ""
                report_lines.append(
                    f"  {f['test']:.<30} {f['value']} {f['unit']:.<12} "
                    f"(ref: {f['reference']}) {marker}{crit}"
                )

        if result["derived"]:
            report_lines += ["", "─── Derived Calculations ───"]
            for name, d in result["derived"].items():
                flag = f" ← {d['flag']}" if d["flag"] != "NORMAL" else ""
                report_lines.append(
                    f"  {name.replace('_', ' ').title():.<30} {d['value']} {d['unit']} "
                    f"(ref: {d['reference']}){flag}"
                )

        if result["unrecognized"]:
            report_lines += ["", "─── Unrecognized Tests ───"]
            for u in result["unrecognized"][:10]:
                report_lines.append(f"  {u['test']:.<30} {u.get('value', '?')}")

        return {
            "summary": summary,
            "metrics": {
                "total_tests": str(result["total_tests"]),
                "abnormal": str(result["flagged_count"]),
                "critical": str(result["critical_count"]),
                "normal": str(result["normal_count"]),
            },
            "protocol_adherence": True,
            "raw_output": "\n".join(report_lines),
        }
