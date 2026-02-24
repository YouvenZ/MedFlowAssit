#!/usr/bin/env python3
"""
main.py — Run the multi-agent medical appointment system.

Modes
─────
  interactive   Chat with the system in your terminal (default)
  demo          Run all pre-built scenarios end-to-end
  scenario N    Run a single numbered scenario (1-12)

Usage
─────
  python main.py                      # interactive REPL
  python main.py interactive          # same
  python main.py demo                 # all 12 scenarios
  python main.py scenario 3           # just scenario 3
  python main.py --model groq/llama-3.3-70b-versatile interactive
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap

from dotenv import load_dotenv

load_dotenv()

from medical_agent import MedicalAppointmentAgent
from paperwork_agent import PaperworkAgent
from llm_config import DEFAULT_MODEL

# ── logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(funcName)-25s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  Scenario registry
# ══════════════════════════════════════════════════════════════════════════════

SCENARIOS: dict[int, dict] = {
    1: {
        "title": "Text triage + clinical note + booking",
        "description": "Patient with headache → triage → note → schedule",
        "messages": [
            (
                "I have a severe headache that started 2 days ago, "
                "it's getting worse and I have nausea."
            ),
            (
                "My name is Rachid Zeghlache, phone 0676574873. "
                "I have a history of migraines and I'm currently on sumatriptan."
            ),
        ],
    },
    2: {
        "title": "Image submission + triage + booking",
        "description": "Chest pain patient with X-ray URL → image analysis → schedule",
        "messages": [
            (
                "Hi, I've been having chest pain and shortness of breath for 3 days. "
                "I also have this chest X-ray from last week: "
                "https://upload.wikimedia.org/wikipedia/commons/c/c8/"
                "Chest_Xray_PA_3-8-2010.png"
            ),
            (
                "My name is Sara Benali, phone 0655443322. "
                "I'm 58 years old, I have hypertension and I'm on amlodipine 5mg."
            ),
        ],
    },
    3: {
        "title": "Emergency triage (IMMEDIATE urgency)",
        "description": "Acute MI symptoms → agent should NOT book, redirect to ER",
        "messages": [
            (
                "I'm having crushing chest pain radiating to my left arm, "
                "severe shortness of breath, and I'm sweating heavily. "
                "This started 20 minutes ago."
            ),
            (
                "My name is Ahmed Khalil, phone 0698765432. "
                "I'm 62, diabetic, smoker, and on metformin."
            ),
        ],
    },
    4: {
        "title": "Book → Retrieve → Update → Cancel",
        "description": "Full appointment lifecycle management",
        "messages": [
            (
                "I have persistent lower back pain for 2 weeks, mild but constant. "
                "No numbness or weakness."
            ),
            (
                "I'm Fatima El Mansouri, phone 0634567890. "
                "No significant medical history, no medications."
            ),
            "Can you show me the details of my appointment?",
            "Actually, can I reschedule to a different time?",
            "Never mind, I need to cancel the appointment.",
        ],
    },
    5: {
        "title": "Multiple symptoms + multiple images",
        "description": "Rash + joint pain with two image uploads → multi-modal triage",
        "messages": [
            (
                "I have a skin rash on my arm and also joint pain in my knees. "
                "The rash has been there for 5 days, joints hurt for 2 weeks. "
                "Here's a photo of the rash: https://example.com/rash_photo.jpg"
            ),
            (
                "I also got knee X-rays done yesterday: "
                "https://example.com/knee_xray.jpg"
            ),
            (
                "My name is Karim Touati, phone 0687654321. "
                "I have rheumatoid arthritis and take methotrexate 15mg weekly."
            ),
        ],
    },
    6: {
        "title": "Pediatric appointment (parent booking)",
        "description": "Child with fever — parent provides details",
        "messages": [
            (
                "My 6-year-old daughter has had a high fever (39.5 C) for 3 days, "
                "she's very tired and has a sore throat. She's refusing to eat."
            ),
            (
                "I'm her mother, Salma Mansouri, phone 0612348765. "
                "My daughter's name is Lina. No medical history, no allergies."
            ),
        ],
    },
    7: {
        "title": "Follow-up for chronic condition",
        "description": "Diabetic patient needing medication review",
        "messages": [
            (
                "I need a follow-up for my diabetes. Last visit was 3 months ago. "
                "Fasting glucose around 180 mg/dL. "
                "On metformin 1000mg BID and glimepiride 2mg."
            ),
            (
                "I'm Hassan Bennani, phone 0678901234. "
                "Diabetic 8 years, also hypertensive on lisinopril 10mg."
            ),
        ],
    },
    8: {
        "title": "Missing info → agent should prompt",
        "description": "Patient provides info gradually — agent asks follow-ups",
        "messages": [
            "I need to see a doctor.",
            "I have stomach pain.",
            "It's been about a week, moderate pain.",
            "I'm Youssef Amrani, 0645678901.",
        ],
    },
    9: {
        "title": "Specialty-specific availability",
        "description": "Eye symptoms → ophthalmology slots",
        "messages": [
            (
                "I've been experiencing blurry vision and eye pain for 4 days. "
                "Especially when looking at bright lights."
            ),
            (
                "I'm Nadia Benjelloun, phone 0656789012. "
                "I wear glasses for myopia, no other medical issues."
            ),
        ],
    },
    10: {
        "title": "Complex medication history (polypharmacy)",
        "description": "Elderly patient on 6+ meds with dizziness/fatigue",
        "messages": [
            (
                "I'm feeling very dizzy and tired all the time. Started a week ago. "
                "Also some nausea."
            ),
            (
                "I'm Mohamed Alami, phone 0667890123. I'm 75. "
                "Medications: warfarin 5mg, metoprolol 50mg BID, furosemide 40mg, "
                "atorvastatin 40mg, omeprazole 20mg, aspirin 100mg daily."
            ),
        ],
    },
    11: {
        "title": "Mental health symptoms",
        "description": "Anxiety & panic attacks → psychiatry referral",
        "messages": [
            (
                "I've been very anxious with panic attacks for the past month. "
                "Poor sleep, trouble concentrating at work."
            ),
            (
                "I'm Leila Fassi, phone 0689012345. "
                "No medical conditions, no medications. Stressful work situation."
            ),
        ],
    },
    12: {
        "title": "Routine preventive care",
        "description": "Healthy patient turning 50, family hx of heart disease",
        "messages": [
            (
                "I'd like a routine health checkup. "
                "Haven't seen a doctor in 2 years, turning 50 next month."
            ),
            (
                "I'm Driss Hammoudi, phone 0690123456. "
                "Generally healthy, no meds. Family history of heart disease."
            ),
        ],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _banner(text: str, char: str = "█"):
    width = 70
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def _run_scenario(agent: MedicalAppointmentAgent, num: int):
    """Execute a single scenario by number."""
    sc = SCENARIOS[num]
    _banner(f"SCENARIO {num} — {sc['title']}")
    print(f"  {sc['description']}\n")

    for msg in sc["messages"]:
        agent.chat(msg)

    # After the scenario, show a summary via PaperworkAgent if data exists
    if agent.patient_info and agent.triage_result:
        _generate_paperwork(agent)


def _generate_paperwork(agent: MedicalAppointmentAgent):
    """Use PaperworkAgent to produce post-scenario documents."""
    pw = PaperworkAgent(model=agent.orchestrator.model)
    pw.set_context(
        patient_info=agent.patient_info,
        triage_result=agent.triage_result,
        clinical_note=agent.clinical_note or "",
        imaging_results=agent.imaging_results,
    )

    print(f"\n{'─'*70}")
    print("  PAPERWORK AGENT — Generating documents")
    print(f"{'─'*70}")

    summary = pw.generate_appointment_summary()
    print(f"\n{summary.get('summary', '')}\n")

    referral = pw.generate_referral_letter()
    print(f"\n{referral.get('letter', '')}\n")

    reminder = pw.generate_followup_reminder()
    print(f"\n{reminder.get('reminder', '')}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  Interactive mode
# ══════════════════════════════════════════════════════════════════════════════

HELP_TEXT = textwrap.dedent("""\
    Commands:
      /help             Show this help
      /reset            Start fresh (new patient)
      /phase            Show current phase (intake / scheduling)
      /patient          Show collected patient info
      /triage           Show triage result
      /note             Show clinical note
      /paperwork        Generate summary, referral & reminder
      /scenario N       Run scenario N (1-12)
      /scenarios        List all available scenarios
      /quit             Exit
""")


def _interactive(agent: MedicalAppointmentAgent):
    """Chat loop in the terminal."""
    print("\n" + "=" * 70)
    print("  Medical Appointment System — Interactive Mode")
    print("  Type /help for commands, or just chat naturally.")
    print("=" * 70 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # ── slash commands ────────────────────────────────────────────────
        if user_input.startswith("/"):
            cmd = user_input.lower().split()
            command = cmd[0]

            if command == "/quit":
                print("Goodbye!")
                break

            elif command == "/help":
                print(HELP_TEXT)

            elif command == "/reset":
                agent.reset()

            elif command == "/phase":
                print(f"  Current phase: {agent.phase.value}")

            elif command == "/patient":
                if agent.patient_info:
                    for k, v in agent.patient_info.items():
                        print(f"  {k:20s}: {v}")
                else:
                    print("  No patient info collected yet.")

            elif command == "/triage":
                if agent.triage_result:
                    import json
                    print(json.dumps(agent.triage_result, indent=2))
                else:
                    print("  No triage result yet.")

            elif command == "/note":
                if agent.clinical_note:
                    print(agent.clinical_note)
                else:
                    print("  No clinical note generated yet.")

            elif command == "/paperwork":
                if agent.patient_info:
                    _generate_paperwork(agent)
                else:
                    print("  No patient data — nothing to generate.")

            elif command == "/scenarios":
                print("\n  Available scenarios:")
                for n, sc in SCENARIOS.items():
                    print(f"    {n:2d}. {sc['title']}")
                    print(f"        {sc['description']}")
                print()

            elif command == "/scenario":
                if len(cmd) < 2 or not cmd[1].isdigit():
                    print("  Usage: /scenario N  (1-12)")
                else:
                    n = int(cmd[1])
                    if n in SCENARIOS:
                        agent.reset()
                        _run_scenario(agent, n)
                    else:
                        print(f"  Scenario {n} not found. Use /scenarios to list.")

            else:
                print(f"  Unknown command: {command}. Type /help for options.")

            continue

        # ── normal chat ───────────────────────────────────────────────────
        agent.chat(user_input)


# ══════════════════════════════════════════════════════════════════════════════
#  Demo mode — run all scenarios
# ══════════════════════════════════════════════════════════════════════════════

def _demo(agent: MedicalAppointmentAgent, scenario_nums: list[int] | None = None):
    """Run specified scenarios, or all if none given."""
    nums = scenario_nums or sorted(SCENARIOS.keys())
    total = len(nums)

    for idx, n in enumerate(nums, 1):
        logger.info("Running scenario %d/%d (#%d)", idx, total, n)
        _run_scenario(agent, n)
        if idx < total:
            agent.reset()

    logger.info("All %d scenarios completed.", total)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-agent medical appointment system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              python main.py                           # interactive chat
              python main.py demo                      # all 12 scenarios
              python main.py scenario 3                # just scenario 3
              python main.py scenario 1 3 5            # scenarios 1, 3, 5
              python main.py --model groq/llama-3.3-70b-versatile demo
        """),
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"LiteLLM model string (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG logging",
    )

    sub = parser.add_subparsers(dest="mode")

    sub.add_parser("interactive", help="Chat in the terminal (default)")
    sub.add_parser("demo", help="Run all 12 demo scenarios")

    sc_parser = sub.add_parser("scenario", help="Run specific scenario(s)")
    sc_parser.add_argument(
        "numbers", nargs="+", type=int,
        help="Scenario number(s) to run (1-12)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Starting with model=%s  mode=%s", args.model, args.mode or "interactive")

    agent = MedicalAppointmentAgent(model=args.model)

    mode = args.mode or "interactive"

    if mode == "interactive":
        _interactive(agent)

    elif mode == "demo":
        _demo(agent)

    elif mode == "scenario":
        valid = [n for n in args.numbers if n in SCENARIOS]
        invalid = [n for n in args.numbers if n not in SCENARIOS]
        if invalid:
            print(f"  Unknown scenario(s): {invalid}. Valid: 1-{max(SCENARIOS)}")
        if valid:
            _demo(agent, valid)


if __name__ == "__main__":
    main()
