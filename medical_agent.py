"""
medical_agent.py — Multi-agent medical appointment system.

Architecture
============
  Orchestrator  ─┬─▸  PatientAgent   (intake, MedGemma triage, imaging)
                 └─▸  DoctorAgent    (clinical note, scheduling, appt mgmt)

All LLM calls go through LiteLLM (Groq provider) — see llm_config.py.
MedGemma runs locally for clinical reasoning (triage, imaging, notes).

This file is the main entry point:
  • MedicalAppointmentAgent  — high-level facade (wraps the Orchestrator)
  • Demo scenarios at the bottom

Legacy single-agent behaviour is preserved: calling `agent.chat(msg)` or
`agent.agent_loop(msg)` both work.
"""

from __future__ import annotations

import logging

from dotenv import load_dotenv

from orchestrator import Orchestrator
from llm_config import DEFAULT_MODEL

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(funcName)-25s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ══════════════════════════════════════════════════════════════════════════════
# MedicalAppointmentAgent — public facade
# ══════════════════════════════════════════════════════════════════════════════

class MedicalAppointmentAgent:
    """
    High-level facade wrapping the multi-agent Orchestrator.

    Usage:
        agent = MedicalAppointmentAgent()
        agent.chat("I have a headache …")          # patient-facing intake
        agent.chat("My name is …, phone …")        # auto-handoff when ready
        agent.chat("Book the first slot please")    # doctor-side scheduling
        agent.reset()                               # new patient session
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        logger.info("Initialising MedicalAppointmentAgent (multi-agent)")
        self.orchestrator = Orchestrator(model=model)

    # ── primary API ───────────────────────────────────────────────────────────

    def chat(self, message: str) -> str:
        """Send a user message through the orchestrator pipeline."""
        return self.orchestrator.chat(message)

    # Alias for backward-compat with the old single-agent demo
    agent_loop = chat

    def reset(self):
        """Reset all agents and start a fresh patient session."""
        self.orchestrator.reset()

    # Alias
    reset_conversation = reset

    # ── convenience accessors ─────────────────────────────────────────────────

    @property
    def patient_info(self):
        return self.orchestrator.patient_agent.patient_info

    @property
    def triage_result(self):
        return self.orchestrator.patient_agent.triage_result

    @property
    def imaging_results(self):
        return self.orchestrator.patient_agent.imaging_results

    @property
    def clinical_note(self):
        return self.orchestrator.doctor_agent.clinical_note

    @property
    def phase(self):
        return self.orchestrator.phase


# ══════════════════════════════════════════════════════════════════════════════
# Demo scenarios
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("Starting Multi-Agent Medical Appointment Demo")
    logger.info("=" * 70)

    agent = MedicalAppointmentAgent()

    # ── Scenario 1: Text-only symptoms → triage → note → booking ─────────────
    print("\n" + "█" * 70)
    print("  SCENARIO 1 — Text triage + clinical note + booking")
    print("█" * 70)

    agent.chat(
        "I have a severe headache that started 2 days ago, "
        "it's getting worse and I have nausea"
    )
    agent.chat(
        "My name is Rachid Zeghlache, phone 0676574873. "
        "I have a history of migraines and I'm currently on sumatriptan."
    )

    # ── Scenario 2: Image submission + triage + booking ──────────────────────
    agent.reset()

    print("\n" + "█" * 70)
    print("  SCENARIO 2 — Image submission + triage + booking")
    print("█" * 70)

    agent.chat(
        "Hi, I've been having chest pain and shortness of breath for 3 days. "
        "I also have this chest X-ray from last week: "
        "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
    )
    agent.chat(
        "My name is Sara Benali, phone 0655443322. "
        "I'm 58 years old, I have hypertension and I'm on amlodipine 5mg."
    )
    agent.chat(
        "this chest X-ray from last week: "
        "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
    )

    # ── Scenario 3: IMMEDIATE urgency — should NOT book ──────────────────────
    agent.reset()

    print("\n" + "█" * 70)
    print("  SCENARIO 3 — Emergency triage (IMMEDIATE urgency)")
    print("█" * 70)

    agent.chat(
        "I'm having crushing chest pain radiating to my left arm, "
        "severe shortness of breath, and I'm sweating heavily. "
        "This started 20 minutes ago."
    )
    agent.chat(
        "My name is Ahmed Khalil, phone 0698765432. "
        "I'm 62, diabetic, smoker, and on metformin."
    )

    # ── Scenario 4: Appointment management workflow ──────────────────────────
    agent.reset()

    print("\n" + "█" * 70)
    print("  SCENARIO 4 — Book → Retrieve → Update → Cancel")
    print("█" * 70)

    agent.chat(
        "I have persistent lower back pain for 2 weeks, mild but constant. "
        "No numbness or weakness."
    )
    agent.chat(
        "I'm Fatima El Mansouri, phone 0634567890. "
        "No significant medical history, no medications."
    )
    agent.chat("Can you show me the details of my appointment?")
    agent.chat("Actually, can I reschedule to a different time?")
    agent.chat("Never mind, I need to cancel the appointment.")

    # ── Scenario 5: Multi-symptom with multiple images ───────────────────────
    agent.reset()

    print("\n" + "█" * 70)
    print("  SCENARIO 5 — Multiple symptoms + multiple images")
    print("█" * 70)

    agent.chat(
        "I have a skin rash on my arm and also joint pain in my knees. "
        "The rash has been there for 5 days, joints hurt for 2 weeks. "
        "Here's a photo of the rash: https://example.com/rash_photo.jpg"
    )
    agent.chat(
        "I also got knee X-rays done yesterday: "
        "https://example.com/knee_xray.jpg"
    )
    agent.chat(
        "My name is Karim Touati, phone 0687654321. "
        "I have rheumatoid arthritis and take methotrexate 15mg weekly."
    )

    # ── Scenario 6: Pediatric case ───────────────────────────────────────────
    agent.reset()

    print("\n" + "█" * 70)
    print("  SCENARIO 6 — Pediatric appointment (parent booking)")
    print("█" * 70)

    agent.chat(
        "My 6-year-old daughter has had a high fever (39.5°C) for 3 days, "
        "she's very tired and has a sore throat. She's refusing to eat."
    )
    agent.chat(
        "I'm her mother, Salma Mansouri, phone 0612348765. "
        "My daughter's name is Lina. No medical history, no allergies."
    )

    # ── Scenario 7: Follow-up appointment ────────────────────────────────────
    agent.reset()

    print("\n" + "█" * 70)
    print("  SCENARIO 7 — Follow-up for chronic condition")
    print("█" * 70)

    agent.chat(
        "I need a follow-up for my diabetes. Last visit was 3 months ago. "
        "Fasting glucose around 180 mg/dL. "
        "On metformin 1000mg BID and glimepiride 2mg."
    )
    agent.chat(
        "I'm Hassan Bennani, phone 0678901234. "
        "Diabetic 8 years, also hypertensive on lisinopril 10mg."
    )

    # ── Scenario 8: Incomplete information → agent prompts ───────────────────
    agent.reset()

    print("\n" + "█" * 70)
    print("  SCENARIO 8 — Missing info → agent should prompt")
    print("█" * 70)

    agent.chat("I need to see a doctor.")
    agent.chat("I have stomach pain.")
    agent.chat("It's been about a week, moderate pain.")
    agent.chat("I'm Youssef Amrani, 0645678901.")

    # ── Scenario 9: Specialty-specific availability ──────────────────────────
    agent.reset()

    print("\n" + "█" * 70)
    print("  SCENARIO 9 — Specialty-specific availability check")
    print("█" * 70)

    agent.chat(
        "I've been experiencing blurry vision and eye pain for 4 days. "
        "Especially when looking at bright lights."
    )
    agent.chat(
        "I'm Nadia Benjelloun, phone 0656789012. "
        "I wear glasses for myopia, no other medical issues."
    )

    # ── Scenario 10: Complex medication history ──────────────────────────────
    agent.reset()

    print("\n" + "█" * 70)
    print("  SCENARIO 10 — Complex medication history")
    print("█" * 70)

    agent.chat(
        "I'm feeling very dizzy and tired all the time. Started a week ago. "
        "Also some nausea."
    )
    agent.chat(
        "I'm Mohamed Alami, phone 0667890123. I'm 75. "
        "Medications: warfarin 5mg, metoprolol 50mg BID, furosemide 40mg, "
        "atorvastatin 40mg, omeprazole 20mg, aspirin 100mg daily."
    )

    # ── Scenario 11: Mental health ───────────────────────────────────────────
    agent.reset()

    print("\n" + "█" * 70)
    print("  SCENARIO 11 — Mental health symptoms")
    print("█" * 70)

    agent.chat(
        "I've been very anxious with panic attacks for the past month. "
        "Poor sleep, trouble concentrating at work."
    )
    agent.chat(
        "I'm Leila Fassi, phone 0689012345. "
        "No medical conditions, no medications. Stressful work situation."
    )

    # ── Scenario 12: Preventive care ─────────────────────────────────────────
    agent.reset()

    print("\n" + "█" * 70)
    print("  SCENARIO 12 — Routine preventive care")
    print("█" * 70)

    agent.chat(
        "I'd like a routine health checkup. "
        "Haven't seen a doctor in 2 years, turning 50 next month."
    )
    agent.chat(
        "I'm Driss Hammoudi, phone 0690123456. "
        "Generally healthy, no meds. Family history of heart disease."
    )

    logger.info("All scenarios completed.")



