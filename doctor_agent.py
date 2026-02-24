"""
doctor_agent.py — Doctor / scheduling-facing agent.

Responsibilities:
  • Receive handoff data from the Patient Agent (via Orchestrator)
  • Generate a structured clinical note (MedGemma)
  • Check appointment availability filtered by recommended specialty
  • Book, retrieve, update, and cancel appointments
  • Communicate scheduling details professionally
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from base_agent import BaseAgent
from db import MedicalAppointmentDB
from medgemma_client import MedGemmaClient
from llm_config import DEFAULT_MODEL

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Shared MedGemma singleton (reuse the same one as PatientAgent)
# ══════════════════════════════════════════════════════════════════════════════

_medgemma: MedGemmaClient | None = None


def _get_medgemma() -> MedGemmaClient:
    global _medgemma
    if _medgemma is None:
        logger.info("Loading MedGemma singleton (DoctorAgent) …")
        _medgemma = MedGemmaClient(
            specialty="general",
            quantization="nf4",
            max_new_tokens=768,
            max_history_turns=4,
        )
    return _medgemma


# ══════════════════════════════════════════════════════════════════════════════
# DoctorAgent
# ══════════════════════════════════════════════════════════════════════════════

class DoctorAgent(BaseAgent):
    """
    Clinical & scheduling agent.

    Receives structured patient data (patient_info, triage, imaging) from
    the Orchestrator and:
      1. Generates a SOAP clinical note (MedGemma)
      2. Checks availability for the recommended specialty
      3. Books / updates / cancels appointments
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        self.db = MedicalAppointmentDB()

        # Populated by the Orchestrator at handoff
        self.patient_info: dict[str, Any] = {}
        self.triage_result: dict[str, Any] = {}
        self.imaging_results: list[dict[str, Any]] = []
        self.clinical_note: str = ""

        super().__init__(name="DoctorAgent", model=model)

    @property
    def medgemma(self) -> MedGemmaClient:
        return _get_medgemma()

    # ── receive handoff ───────────────────────────────────────────────────────

    def receive_handoff(self, data: dict):
        """Accept patient data from the PatientAgent."""
        self.patient_info    = data.get("patient_info", {})
        self.triage_result   = data.get("triage_result", {})
        self.imaging_results = data.get("imaging_results", [])
        logger.info("[DoctorAgent] handoff received for %s",
                    self.patient_info.get("name", "Unknown"))

    # ── system prompt ─────────────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        return (
            "You are a professional medical scheduling and clinical documentation assistant "
            "working on the doctor's side of the workflow.\n\n"

            "YOUR TOOLS:\n"
            "  1. generate_clinical_note    — synthesise all data into a SOAP note (MedGemma)\n"
            "  2. check_availability        — find open slots (filter by specialty)\n"
            "  3. book_appointment           — confirm a booking\n"
            "  4. get_appointment            — retrieve appointment details\n"
            "  5. update_appointment         — reschedule to a new slot\n"
            "  6. cancel_appointment         — cancel an existing booking\n\n"

            "MANDATORY WORKFLOW (new patient):\n"
            "  • First, call generate_clinical_note to create documentation.\n"
            "  • Then check_availability filtered by the recommended specialty.\n"
            "  • Present available slots and book once the patient confirms.\n\n"

            "URGENCY RULES:\n"
            "  IMMEDIATE → Do NOT book. Instruct patient to call 15 / go to ER.\n"
            "  URGENT    → Book same-day or next-day slot. Prioritise.\n"
            "  ROUTINE   → Schedule normally.\n\n"

            "Always confirm slot details (date, time, doctor, specialty) before booking.\n"
            "Be professional, clear, and efficient.\n"
        )

    # ── tool implementations ──────────────────────────────────────────────────

    def generate_clinical_note(self) -> dict:
        """Synthesise all gathered data into a SOAP clinical note."""
        logger.info("[DoctorAgent] generating clinical note")
        if not self.patient_info:
            return {"status": "error", "message": "No patient data available."}

        imaging_section = ""
        if self.imaging_results:
            lines = []
            for i, r in enumerate(self.imaging_results, 1):
                lines.append(
                    f"Image {i} ({r.get('image_type', 'unknown')}): "
                    f"{r.get('report', 'No report.')}"
                )
            imaging_section = "\n\nIMAGING FINDINGS\n" + "\n\n".join(lines)

        triage_section = ""
        if self.triage_result:
            t = self.triage_result
            triage_section = (
                f"\n\nTRIAGE ASSESSMENT (AI-generated)\n"
                f"Urgency        : {t.get('urgency_level', 'Unknown')}\n"
                f"Possible Dx    : {', '.join(t.get('possible_conditions', []))}\n"
                f"Specialty      : {t.get('recommended_specialty', 'Unknown')}\n"
                f"Reasoning      : {t.get('reasoning', '')}\n"
                f"Red flags      : {', '.join(t.get('red_flags', [])) or 'None'}\n"
                f"Investigations : {', '.join(t.get('suggested_investigations', [])) or 'None'}"
            )

        prompt = f"""You are a clinical documentation assistant. Write a concise SOAP clinical note.

PATIENT INFORMATION
-------------------
Name               : {self.patient_info.get('name', 'Unknown')}
Phone              : {self.patient_info.get('phone', 'Unknown')}
Presenting symptoms: {self.patient_info.get('symptoms', 'Not specified')}
Duration           : {self.patient_info.get('duration', 'Not specified')}
Severity           : {self.patient_info.get('severity', 'Not specified')}
Medical history    : {self.patient_info.get('medical_history', 'None reported')}
Current medications: {self.patient_info.get('current_medications', 'None reported')}
{triage_section}{imaging_section}

Write the clinical note in standard SOAP format. Keep it under 300 words.
Use professional medical language. Begin directly — no preamble."""

        self.medgemma.reset()
        try:
            note = self.medgemma.ask(prompt)
            self.clinical_note = note
            return {
                "status": "success", "clinical_note": note,
                "includes_imaging": bool(self.imaging_results),
                "includes_triage": bool(self.triage_result),
            }
        except Exception as exc:
            logger.error("[DoctorAgent] note error: %s", exc, exc_info=True)
            return {"status": "error", "message": f"Note generation failed: {exc}"}

    def check_availability(self, specialty: str | None = None, date: str | None = None) -> list:
        """Check available slots, optionally filtered."""
        logger.info("[DoctorAgent] checking availability  specialty=%s  date=%s", specialty, date)
        return self.db.check_availability(specialty=specialty, date=date)

    def book_appointment(self, patient_name: str, patient_phone: str, time_slot_id: int) -> dict:
        """Book a slot. Attaches clinical note if available."""
        name  = self.patient_info.get("name",  patient_name)
        phone = self.patient_info.get("phone", patient_phone)
        logger.info("[DoctorAgent] booking slot %d for %s", time_slot_id, name)

        result = self.db.book_appointment(
            patient_name=name, patient_phone=phone, time_slot_id=time_slot_id,
        )
        if self.clinical_note and isinstance(result, dict):
            result["clinical_note_attached"] = True
            result["clinical_note_preview"] = self.clinical_note[:200] + "…"
        return result

    def get_appointment(self, appointment_id: int) -> dict | None:
        return self.db.get_appointment(appointment_id)

    def cancel_appointment(self, appointment_id: int) -> dict:
        return self.db.cancel_appointment(appointment_id)

    def update_appointment(self, appointment_id: int, new_time_slot_id: int) -> dict:
        return self.db.update_appointment(appointment_id, new_time_slot_id)

    # ── tool dispatch ─────────────────────────────────────────────────────────

    def available_function(self, function_name: str):
        _map = {
            "generate_clinical_note": self.generate_clinical_note,
            "check_availability":     self.check_availability,
            "book_appointment":       self.book_appointment,
            "get_appointment":        self.get_appointment,
            "cancel_appointment":     self.cancel_appointment,
            "update_appointment":     self.update_appointment,
        }
        if function_name not in _map:
            raise ValueError(f"[DoctorAgent] unknown tool: {function_name!r}")
        return _map[function_name]

    # ── tool schemas ──────────────────────────────────────────────────────────

    def _build_tool_schemas(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "generate_clinical_note",
                    "description": (
                        "Synthesise patient data into a SOAP clinical note. "
                        "Call BEFORE booking."
                    ),
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "check_availability",
                    "description": "Check available appointment slots, optionally filtered by specialty or date.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "specialty": {"type": "string", "description": "Specialty to filter"},
                            "date":      {"type": "string", "description": "YYYY-MM-DD"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "book_appointment",
                    "description": "Confirm and book a selected appointment slot.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "patient_name":  {"type": "string", "description": "Full name"},
                            "patient_phone": {"type": "string", "description": "Phone"},
                            "time_slot_id":  {"type": "integer", "description": "Slot ID"},
                        },
                        "required": ["patient_name", "patient_phone", "time_slot_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_appointment",
                    "description": "Retrieve details of an existing appointment.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "appointment_id": {"type": "integer", "description": "Appointment ID"},
                        },
                        "required": ["appointment_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "cancel_appointment",
                    "description": "Cancel an existing appointment.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "appointment_id": {"type": "integer", "description": "Appointment ID"},
                        },
                        "required": ["appointment_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "update_appointment",
                    "description": "Reschedule an appointment to a new time slot.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "appointment_id":   {"type": "integer", "description": "Appointment ID"},
                            "new_time_slot_id": {"type": "integer", "description": "New slot ID"},
                        },
                        "required": ["appointment_id", "new_time_slot_id"],
                    },
                },
            },
        ]

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset_state(self):
        self.reset()
        self.patient_info    = {}
        self.triage_result   = {}
        self.imaging_results = []
        self.clinical_note   = ""
        logger.info("[DoctorAgent] state reset")
