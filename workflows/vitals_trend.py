"""
workflows/vitals_trend.py — ICU / Nursing: CSV Vitals Trend Monitor.

Input:  CSV with time-series vital signs.
Protocol: NEWS2 (National Early Warning Score 2) — Royal College of Physicians.
Chain:
  1. Parse CSV  — extract timestamp + vitals columns.
  2. NEWS2 calc — deterministic scoring for each row.
  3. Trend      — detect deterioration (rising NEWS2 over time).
  4. Output     — per-row scores, aggregate risk, escalation recommendation.

Expected CSV columns:
  timestamp | datetime,  hr | heart_rate,  sbp | systolic_bp,  rr | respiratory_rate,
  temp | temperature,  spo2 | oxygen_sat,  consciousness | avpu,  [o2_supplement]

NEWS2 Parameters (0-3 each, total 0-20):
  • Respiratory rate  • SpO₂  • SpO₂ Scale 2 (for O₂ target 88-92%)
  • Air or Oxygen  • Systolic BP  • Heart rate  • Consciousness  • Temperature
"""

from __future__ import annotations

import csv
import io
import logging
import re
from datetime import datetime
from typing import Any, Optional

from workflows.base import ClinicalWorkflow, InputType

logger = logging.getLogger(__name__)


# ── NEWS2 scoring tables ─────────────────────────────────────────────────────

def _score_rr(rr: float) -> int:
    if rr <= 8: return 3
    if rr <= 11: return 1
    if rr <= 20: return 0
    if rr <= 24: return 2
    return 3

def _score_spo2_scale1(spo2: float) -> int:
    """SpO₂ Scale 1 (most patients)."""
    if spo2 <= 91: return 3
    if spo2 <= 93: return 2
    if spo2 <= 95: return 1
    return 0

def _score_spo2_scale2(spo2: float) -> int:
    """SpO₂ Scale 2 (target 88-92%, e.g. COPD)."""
    if spo2 <= 83: return 3
    if spo2 <= 85: return 2
    if spo2 <= 86: return 1
    if spo2 <= 92: return 0
    if spo2 <= 94: return 1
    if spo2 <= 96: return 2
    return 3

def _score_o2(on_o2: bool) -> int:
    return 2 if on_o2 else 0

def _score_sbp(sbp: float) -> int:
    if sbp <= 90: return 3
    if sbp <= 100: return 2
    if sbp <= 110: return 1
    if sbp <= 219: return 0
    return 3

def _score_hr(hr: float) -> int:
    if hr <= 40: return 3
    if hr <= 50: return 1
    if hr <= 90: return 0
    if hr <= 110: return 1
    if hr <= 130: return 2
    return 3

def _score_temp(temp: float) -> int:
    if temp <= 35.0: return 3
    if temp <= 36.0: return 1
    if temp <= 38.0: return 0
    if temp <= 39.0: return 1
    return 2

def _score_consciousness(avpu: str) -> int:
    """A=Alert, V=Voice, P=Pain, U=Unresponsive; also handle 'C' for Confusion."""
    avpu = avpu.strip().upper()
    if avpu in ("A", "ALERT"):
        return 0
    if avpu in ("C", "CONFUSION", "CONFUSED", "V", "VOICE", "P", "PAIN", "U", "UNRESPONSIVE"):
        return 3
    return 0


def _calculate_news2(vitals: dict, use_scale2: bool = False) -> dict:
    """Calculate NEWS2 from a single set of vitals."""
    components = {}
    total = 0
    available = 0

    rr = vitals.get("rr")
    if rr is not None:
        s = _score_rr(rr)
        components["Respiratory rate"] = {"value": rr, "score": s}
        total += s
        available += 1

    spo2 = vitals.get("spo2")
    if spo2 is not None:
        if use_scale2:
            s = _score_spo2_scale2(spo2)
            components["SpO₂ (Scale 2)"] = {"value": spo2, "score": s}
        else:
            s = _score_spo2_scale1(spo2)
            components["SpO₂ (Scale 1)"] = {"value": spo2, "score": s}
        total += s
        available += 1

    on_o2 = vitals.get("on_o2", False)
    s = _score_o2(on_o2)
    components["Air or Oxygen"] = {"value": "Oxygen" if on_o2 else "Air", "score": s}
    total += s

    sbp = vitals.get("sbp")
    if sbp is not None:
        s = _score_sbp(sbp)
        components["Systolic BP"] = {"value": sbp, "score": s}
        total += s
        available += 1

    hr = vitals.get("hr")
    if hr is not None:
        s = _score_hr(hr)
        components["Heart rate"] = {"value": hr, "score": s}
        total += s
        available += 1

    temp = vitals.get("temp")
    if temp is not None:
        s = _score_temp(temp)
        components["Temperature"] = {"value": temp, "score": s}
        total += s
        available += 1

    avpu = vitals.get("consciousness", "A")
    s = _score_consciousness(avpu)
    components["Consciousness"] = {"value": avpu, "score": s}
    total += s

    # Individual ≥3 in any parameter triggers escalation
    any_single_3 = any(c["score"] >= 3 for c in components.values())

    # Clinical risk
    if total >= 7:
        risk = "HIGH"
        trigger = "Aggregate ≥7"
        response = ("URGENT/EMERGENCY response. Continuous monitoring. "
                    "Senior clinician review STAT. Consider ICU transfer.")
    elif total >= 5 or any_single_3:
        risk = "MEDIUM"
        trigger = "Score 5-6 or single-param 3" if any_single_3 else "Aggregate 5-6"
        response = ("URGENT response. Increase monitoring to q1h minimum. "
                    "Senior clinician assessment within 30 minutes.")
    elif total >= 1:
        risk = "LOW"
        trigger = "Aggregate 1-4"
        response = ("LOW risk. Minimum q4-6h monitoring. "
                    "Inform nurse in charge. Reassess if trending upward.")
    else:
        risk = "NONE"
        trigger = "Aggregate 0"
        response = "Continue routine monitoring q12h."

    return {
        "total": total,
        "max_score": 20,
        "components": components,
        "risk": risk,
        "trigger": trigger,
        "response": response,
        "vitals_available": available,
    }


def _parse_vitals_csv(csv_text: str) -> list[dict]:
    """Parse a vitals CSV → list of vitals dicts with optional timestamp."""
    reader = csv.DictReader(io.StringIO(csv_text.strip()))
    if not reader.fieldnames:
        return []

    # Flexible column mapping
    fn_lower = {f.strip().lower(): f for f in reader.fieldnames}
    col_map = {}

    for target, candidates in {
        "timestamp": ["timestamp", "datetime", "date_time", "time", "date"],
        "hr":        ["hr", "heart_rate", "heartrate", "pulse"],
        "sbp":       ["sbp", "systolic_bp", "systolic", "sys_bp"],
        "rr":        ["rr", "respiratory_rate", "resp_rate", "resp"],
        "temp":      ["temp", "temperature", "body_temp"],
        "spo2":      ["spo2", "oxygen_sat", "o2sat", "o2_sat", "saturation", "sao2"],
        "consciousness": ["consciousness", "avpu", "gcs", "mental_status", "loc"],
        "on_o2":     ["on_o2", "o2_supplement", "oxygen", "supplemental_o2", "o2"],
    }.items():
        for cand in candidates:
            if cand in fn_lower:
                col_map[target] = fn_lower[cand]
                break

    rows = []
    for row in reader:
        vitals = {}

        # Timestamp
        ts_col = col_map.get("timestamp")
        if ts_col and row.get(ts_col):
            vitals["timestamp_raw"] = row[ts_col].strip()

        # Numeric vitals
        for key in ("hr", "sbp", "rr", "temp", "spo2"):
            col = col_map.get(key)
            if col and row.get(col):
                try:
                    vitals[key] = float(re.sub(r"[^0-9.]", "", row[col].strip()))
                except ValueError:
                    pass

        # Consciousness (string)
        c_col = col_map.get("consciousness")
        if c_col and row.get(c_col):
            vitals["consciousness"] = row[c_col].strip()

        # On O₂ (boolean)
        o2_col = col_map.get("on_o2")
        if o2_col and row.get(o2_col):
            val = row[o2_col].strip().lower()
            vitals["on_o2"] = val in ("true", "yes", "1", "y", "oxygen")

        if any(k in vitals for k in ("hr", "sbp", "rr", "temp", "spo2")):
            rows.append(vitals)

    return rows


class VitalsTrendWorkflow(ClinicalWorkflow):
    workflow_id  = "vitals_trend"
    name         = "VitalsTrend"
    icon         = "📈"
    description  = "Upload a CSV of serial vital signs → NEWS2 scoring with deterioration trend detection."
    input_types  = [InputType.CSV]
    protocol     = "NEWS2 (RCP)"
    specialty    = "ICU / Nursing"

    def validate_input(self, data: dict[str, Any]) -> list[str]:
        errors = []
        if not data.get("csv_text"):
            errors.append("CSV data is required. Upload a CSV with time-series vitals.")
        return errors

    def execute(self, data: dict[str, Any]) -> dict[str, Any]:
        csv_text = data["csv_text"]

        # ── Step 1: Parse CSV ────────────────────────────────────────────
        rows = _parse_vitals_csv(csv_text)
        if not rows:
            return {
                "summary": "No vital signs parsed from CSV. Expected columns: hr, sbp, rr, temp, spo2.",
                "metrics": {},
                "protocol_adherence": False,
                "raw_output": f"Input CSV ({len(csv_text)} chars) → 0 rows parseable.",
            }

        # ── Step 2: Calculate NEWS2 for each row ─────────────────────────
        scored = []
        for i, vitals in enumerate(rows):
            news = _calculate_news2(vitals)
            scored.append({
                "index": i + 1,
                "timestamp": vitals.get("timestamp_raw", f"Row {i+1}"),
                "news2_total": news["total"],
                "risk": news["risk"],
                "components": news["components"],
                "response": news["response"],
            })

        # ── Step 3: Trend analysis ───────────────────────────────────────
        scores = [s["news2_total"] for s in scored]
        latest = scored[-1]
        peak = max(scores)
        peak_idx = scores.index(peak)
        mean_score = sum(scores) / len(scores)

        # Detect deterioration: rising trend in last 3+ readings
        deteriorating = False
        if len(scores) >= 3:
            last_3 = scores[-3:]
            if last_3[-1] > last_3[-2] > last_3[-3]:
                deteriorating = True
            # Also flag if last score is significantly above mean
            if scores[-1] >= mean_score + 2:
                deteriorating = True

        # ── Step 4: Assemble output ──────────────────────────────────────
        summary_parts = [
            f"{len(scored)} vital sign sets analyzed.",
            f"Latest NEWS2: {latest['news2_total']}/20 ({latest['risk']} risk).",
            f"Peak NEWS2: {peak}/20 at {scored[peak_idx]['timestamp']}.",
        ]
        if deteriorating:
            summary_parts.append("⚠ DETERIORATION TREND DETECTED — escalate care.")
        summary = " ".join(summary_parts)

        report_lines = [
            "═══ VITALS TREND & NEWS2 REPORT ═══",
            "",
            f"Total Readings: {len(scored)}",
            f"Latest NEWS2: {latest['news2_total']}/20 — {latest['risk']} risk",
            f"Peak NEWS2: {peak}/20 at {scored[peak_idx]['timestamp']}",
            f"Mean NEWS2: {mean_score:.1f}",
            f"Trend: {'⚠ DETERIORATING' if deteriorating else '→ Stable/Improving'}",
            "",
        ]

        if deteriorating:
            report_lines += [
                "⚠⚠⚠ DETERIORATION DETECTED ⚠⚠⚠",
                latest["response"],
                "",
            ]

        report_lines.append("─── NEWS2 Timeline ───")
        for s in scored:
            marker = " ⚠" if s["risk"] in ("HIGH", "MEDIUM") else ""
            report_lines.append(
                f"  {s['timestamp']:.<25} NEWS2={s['news2_total']:>2}/20  "
                f"[{s['risk']}]{marker}"
            )

        # Show component detail for latest reading
        report_lines += ["", "─── Latest Vitals Detail ───"]
        for comp_name, comp_data in latest["components"].items():
            score_str = f"+{comp_data['score']}" if comp_data["score"] > 0 else " 0"
            report_lines.append(
                f"  {comp_name:.<30} {comp_data['value']}  → {score_str}"
            )

        report_lines += [
            "",
            "─── Recommended Response (Latest) ───",
            latest["response"],
        ]

        return {
            "summary": summary,
            "metrics": {
                "readings": str(len(scored)),
                "latest_NEWS2": f"{latest['news2_total']}/20",
                "latest_risk": latest["risk"],
                "peak_NEWS2": f"{peak}/20",
                "trend": "DETERIORATING" if deteriorating else "Stable",
            },
            "protocol_adherence": True,
            "raw_output": "\n".join(report_lines),
        }
