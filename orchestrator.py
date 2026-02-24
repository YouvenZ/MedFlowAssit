"""
orchestrator.py — Multi-agent orchestrator for the patient journey.

Manages the handoff between:
  • PatientAgent  — intake, triage, imaging
  • DoctorAgent   — clinical note, scheduling, appointment management

The Orchestrator uses an LLM (via LiteLLM/Groq) to decide which agent
should handle each user message and when to trigger the handoff.
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Any

from llm_config import llm_completion, DEFAULT_MODEL
from patient_agent import PatientAgent
from doctor_agent import DoctorAgent

logger = logging.getLogger(__name__)


class Phase(str, Enum):
    """Which phase of the patient journey we are in."""
    INTAKE     = "intake"       # PatientAgent handles
    SCHEDULING = "scheduling"   # DoctorAgent handles


class Orchestrator:
    """
    Routes user messages to the correct agent.

    Lifecycle for a single patient:
      1. INTAKE phase   → PatientAgent  (collect info, triage, image)
      2. Handoff        → patient data transferred to DoctorAgent
      3. SCHEDULING     → DoctorAgent   (note, availability, booking, mgmt)

    The orchestrator itself uses a lightweight LLM classifier to decide
    whether the user's intent matches the current phase or triggers a
    phase transition.
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.patient_agent = PatientAgent(model=model)
        self.doctor_agent  = DoctorAgent(model=model)
        self.phase: Phase  = Phase.INTAKE
        self._handoff_done = False
        logger.info("[Orchestrator] ready  model=%s", model)

    # ── routing classifier ────────────────────────────────────────────────────

    def _classify_intent(self, user_input: str) -> Phase:
        """
        Use a cheap LLM call to decide whether the message belongs to
        INTAKE or SCHEDULING.  Falls back to the current phase on error.
        """
        # Fast heuristics first
        scheduling_keywords = [
            "book", "schedule", "appointment", "cancel", "reschedule",
            "update", "available", "slot", "time", "doctor", "when",
            "details of my appointment", "show my appointment",
        ]
        intake_keywords = [
            "symptom", "pain", "hurt", "fever", "nausea", "image",
            "x-ray", "xray", "mri", "ct scan", "photo", "rash",
            "bleeding", "dizzy", "headache", "cough",
        ]

        lower = user_input.lower()
        sched_score  = sum(1 for kw in scheduling_keywords if kw in lower)
        intake_score = sum(1 for kw in intake_keywords if kw in lower)

        if sched_score > intake_score and self._handoff_done:
            return Phase.SCHEDULING
        if intake_score > sched_score:
            return Phase.INTAKE

        # Ambiguous — fall back to current phase
        return self.phase

    # ── handoff ───────────────────────────────────────────────────────────────

    def _do_handoff(self):
        """Transfer patient data from PatientAgent → DoctorAgent."""
        if self._handoff_done:
            return

        data = self.patient_agent.get_handoff_data()
        if not data.get("patient_info"):
            logger.warning("[Orchestrator] handoff skipped — no patient info yet")
            return

        self.doctor_agent.receive_handoff(data)

        # Inject context summary into doctor agent conversation
        summary = self._build_handoff_summary(data)
        self.doctor_agent.add_context_message("system", summary)

        self._handoff_done = True
        self.phase = Phase.SCHEDULING
        logger.info("[Orchestrator] handoff complete → SCHEDULING phase")
        print(f"\n{'▓'*60}")
        print("  🔀  HANDOFF: Patient Agent → Doctor Agent")
        print(f"{'▓'*60}\n")

    def _build_handoff_summary(self, data: dict) -> str:
        pi = data.get("patient_info", {})
        tr = data.get("triage_result", {})
        imgs = data.get("imaging_results", [])

        parts = [
            "=== PATIENT HANDOFF DATA ===",
            f"Name       : {pi.get('name', '?')}",
            f"Phone      : {pi.get('phone', '?')}",
            f"Symptoms   : {pi.get('symptoms', '?')}",
            f"Duration   : {pi.get('duration', '?')}",
            f"Severity   : {pi.get('severity', '?')}",
            f"History    : {pi.get('medical_history', '?')}",
            f"Medications: {pi.get('current_medications', '?')}",
        ]
        if tr:
            parts += [
                "\n--- TRIAGE ---",
                f"Urgency    : {tr.get('urgency_level', '?')}",
                f"Specialty  : {tr.get('recommended_specialty', '?')}",
                f"Conditions : {', '.join(tr.get('possible_conditions', []))}",
                f"Red flags  : {', '.join(tr.get('red_flags', [])) or 'None'}",
                f"Reasoning  : {tr.get('reasoning', '')}",
            ]
            if tr.get("emergency_warning"):
                parts.append(f"⚠ EMERGENCY: {tr['emergency_warning']}")

        if imgs:
            parts.append("\n--- IMAGING ---")
            for i, r in enumerate(imgs, 1):
                parts.append(f"Image {i} ({r.get('image_type', '?')}): "
                             f"{r.get('report', 'N/A')[:300]}")

        return "\n".join(parts)

    # ── public API ────────────────────────────────────────────────────────────

    def chat(self, user_input: str) -> str:
        """
        Main entry point.  Routes to the appropriate agent and manages
        phase transitions.
        """
        logger.info("=" * 70)
        logger.info("[Orchestrator] phase=%s  input=%s", self.phase.value, user_input[:100])

        print(f"\n{'═'*70}")
        print(f"  USER: {user_input}")
        print(f"{'═'*70}")

        # Decide which agent should handle this message
        target_phase = self._classify_intent(user_input)

        # If the patient agent has collected enough data and intent is
        # scheduling, trigger handoff
        if (
            target_phase == Phase.SCHEDULING
            and not self._handoff_done
            and self.patient_agent.patient_info
            and self.patient_agent.triage_result
        ):
            self._do_handoff()

        # If explicitly scheduling but handoff hasn't happened yet,
        # check if we should force it
        if (
            self.phase == Phase.INTAKE
            and self._should_auto_handoff()
        ):
            self._do_handoff()
            target_phase = Phase.SCHEDULING

        self.phase = target_phase

        # Route to the right agent
        if self.phase == Phase.INTAKE:
            reply = self.patient_agent.run(user_input)
            # Check if intake is complete after this turn
            if self._should_auto_handoff():
                self._do_handoff()
        else:
            reply = self.doctor_agent.run(user_input)

        logger.info("[Orchestrator] reply_len=%d  phase=%s", len(reply), self.phase.value)

        print(f"\n{'═'*70}")
        print(f"  ASSISTANT: {reply}")
        print(f"{'═'*70}\n")

        return reply

    def _should_auto_handoff(self) -> bool:
        """
        Auto-handoff when patient agent has completed intake + triage
        AND urgency is NOT IMMEDIATE (emergencies don't get scheduled).
        """
        if self._handoff_done:
            return False
        pa = self.patient_agent
        if not pa.patient_info or not pa.triage_result:
            return False
        urgency = pa.triage_result.get("urgency_level", "").upper()
        if urgency == "IMMEDIATE":
            logger.info("[Orchestrator] IMMEDIATE urgency — no handoff to scheduling")
            return False
        return True

    def reset(self):
        """Reset everything for a new patient."""
        logger.info("[Orchestrator] full reset")
        self.patient_agent.reset_state()
        self.doctor_agent.reset_state()
        self.phase = Phase.INTAKE
        self._handoff_done = False
        print("\n🔄  Session reset — ready for next patient.\n")


# ══════════════════════════════════════════════════════════════════════════════
# Quick standalone test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(funcName)-25s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    orch = Orchestrator()

    # Simple intake → scheduling flow
    orch.chat(
        "I have a severe headache that started 2 days ago, "
        "it's getting worse and I have nausea."
    )
    orch.chat(
        "My name is Rachid Zeghlache, phone 0676574873. "
        "I have a history of migraines and I'm on sumatriptan."
    )
    # At this point the orchestrator should auto-handoff to doctor agent
    # Doctor agent generates a note, checks availability, and presents slots
