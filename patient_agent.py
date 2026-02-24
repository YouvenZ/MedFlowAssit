"""
patient_agent.py — Patient-facing agent.

Responsibilities:
  • Collect patient demographics & clinical intake
  • Run MedGemma symptom triage
  • Run MedGemma medical image analysis
  • Communicate results in empathetic, plain language
  • Hand off structured data to the Doctor Agent via the Orchestrator
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from base_agent import BaseAgent
from medgemma_client import MedGemmaClient
from llm_config import DEFAULT_MODEL

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Shared MedGemma singleton
# ══════════════════════════════════════════════════════════════════════════════

_medgemma: MedGemmaClient | None = None


def _get_medgemma() -> MedGemmaClient:
    global _medgemma
    if _medgemma is None:
        logger.info("Loading MedGemma singleton …")
        _medgemma = MedGemmaClient(
            specialty="general",
            quantization="nf4",
            max_new_tokens=768,
            max_history_turns=4,
        )
        logger.info("MedGemma ready.")
    return _medgemma


# ══════════════════════════════════════════════════════════════════════════════
# PatientAgent
# ══════════════════════════════════════════════════════════════════════════════

class PatientAgent(BaseAgent):
    """
    Handles the patient-facing side of the journey:
      1. Greet and collect information
      2. Symptom triage via MedGemma
      3. Image analysis via MedGemma
      4. Present results to the patient in plain language
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        # Domain state — shared later with the doctor agent
        self.patient_info: dict[str, Any] = {}
        self.triage_result: dict[str, Any] = {}
        self.imaging_results: list[dict[str, Any]] = []
        super().__init__(name="PatientAgent", model=model)

    # ── lazy MedGemma ─────────────────────────────────────────────────────────

    @property
    def medgemma(self) -> MedGemmaClient:
        return _get_medgemma()

    # ── system prompt ─────────────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        return (
            "You are a warm, empathetic medical intake assistant helping patients "
            "prepare for their appointment.\n\n"

            "YOUR TOOLS (use them in order):\n"
            "  1. collect_patient_information — gather name, phone, symptoms, "
            "duration, severity, history, medications\n"
            "  2. get_medical_assessment — run an AI triage on the symptoms\n"
            "  3. analyze_uploaded_image — analyse any medical image the patient provides\n\n"

            "WORKFLOW:\n"
            "  • First, ask for any missing information (name, phone, symptoms at minimum).\n"
            "  • Once you have enough data, call collect_patient_information.\n"
            "  • Then call get_medical_assessment to triage.\n"
            "  • If the patient shares an image URL/path, call analyze_uploaded_image.\n"
            "  • Summarise the results to the patient in plain, reassuring language.\n"
            "  • Tell the patient you will now connect them with the scheduling team.\n\n"

            "URGENCY RULES:\n"
            "  IMMEDIATE → Tell the patient to call emergency services (15 / 112) NOW.\n"
            "              Do NOT proceed to scheduling.\n"
            "  URGENT    → Reassure but stress that a same-day appointment is needed.\n"
            "  ROUTINE   → Schedule normally.\n\n"

            "TONE: Professional, caring, clear. Avoid jargon.\n"
        )

    # ── tool implementations ──────────────────────────────────────────────────

    def collect_patient_information(
        self,
        patient_name: str,
        patient_phone: str,
        symptoms: str,
        duration: str | None = None,
        severity: str | None = None,
        medical_history: str | None = None,
        current_medications: str | None = None,
    ) -> dict:
        """Store demographics + symptoms."""
        logger.info("[PatientAgent] collecting info for %s", patient_name)
        self.patient_info = {
            "name": patient_name,
            "phone": patient_phone,
            "symptoms": symptoms,
            "duration": duration or "Not specified",
            "severity": severity or "Not specified",
            "medical_history": medical_history or "None reported",
            "current_medications": current_medications or "None reported",
            "collected_at": datetime.utcnow().isoformat(),
        }
        return {"status": "success", "message": f"Info collected for {patient_name}.",
                "data": self.patient_info}

    def get_medical_assessment(
        self,
        symptoms: str,
        duration: str | None = None,
        severity: str | None = None,
        medical_history: str | None = None,
        current_medications: str | None = None,
    ) -> dict:
        """MedGemma symptom triage — returns structured JSON."""
        logger.info("[PatientAgent] triage for: %s", symptoms[:80])
        med_hx  = medical_history     or self.patient_info.get("medical_history",     "None reported")
        curr_mx = current_medications or self.patient_info.get("current_medications", "None reported")
        dur     = duration            or self.patient_info.get("duration",            "Not specified")
        sev     = severity            or self.patient_info.get("severity",            "Not specified")

        prompt = f"""You are performing a clinical triage assessment. Based on the information below,
produce a structured JSON response.

PATIENT PRESENTATION
--------------------
Symptoms          : {symptoms}
Duration          : {dur}
Severity          : {sev}
Relevant history  : {med_hx}
Current medications: {curr_mx}

Respond ONLY with a valid JSON object using this exact schema:
{{
  "possible_conditions": ["<condition1>", "<condition2>", "<condition3>"],
  "urgency_level": "<ROUTINE | URGENT | IMMEDIATE>",
  "recommended_specialty": "<specialty name>",
  "reasoning": "<2-3 sentence clinical rationale>",
  "red_flags": ["<flag1>", "<flag2>"],
  "suggested_investigations": ["<investigation1>", "<investigation2>"],
  "emergency_warning": "<critical warning string, or null>"
}}

Rules:
- IMMEDIATE  → life-threatening presentation
- URGENT     → needs same-day or next-day review
- ROUTINE    → can be managed within days to weeks
- Do NOT include markdown, code fences, or any text outside the JSON object.
"""
        self.medgemma.reset()
        try:
            raw = self.medgemma.ask(prompt)
            clean = raw.strip()
            if clean.startswith("```"):
                clean = "\n".join(
                    l for l in clean.splitlines() if not l.strip().startswith("```")
                ).strip()
            self.triage_result = json.loads(clean)
        except json.JSONDecodeError:
            logger.warning("[PatientAgent] MedGemma non-JSON triage")
            self.triage_result = {
                "possible_conditions": ["Undetermined — clinical review required"],
                "urgency_level": "ROUTINE",
                "recommended_specialty": "General Practice",
                "reasoning": raw if "raw" in dir() else "Assessment unavailable.",
                "red_flags": [], "suggested_investigations": [],
                "emergency_warning": None,
            }
        except Exception as exc:
            logger.error("[PatientAgent] triage error: %s", exc, exc_info=True)
            self.triage_result = {
                "possible_conditions": [],
                "urgency_level": "ROUTINE",
                "recommended_specialty": "General Practice",
                "reasoning": f"Assessment error: {exc}",
                "red_flags": [], "suggested_investigations": [],
                "emergency_warning": None,
            }
        return {"status": "success", "assessment": self.triage_result}

    def analyze_uploaded_image(
        self,
        image_source: str,
        image_type: str = "medical image",
        clinical_question: str = "",
    ) -> dict:
        """MedGemma vision analysis of a medical image."""
        logger.info("[PatientAgent] analysing image: %s", image_source)
        patient_ctx = ""
        if self.patient_info:
            patient_ctx = (
                f"\nPatient context: {self.patient_info.get('symptoms', 'N/A')} "
                f"({self.patient_info.get('duration', '?')}, "
                f"{self.patient_info.get('severity', '?')})."
            )
        question_str = f"\nClinical question: {clinical_question}" if clinical_question else ""

        prompt = (
            f"You are analysing a {image_type} for a clinical appointment scheduling system."
            f"{patient_ctx}{question_str}\n\n"
            "Provide a structured report with:\n"
            "1. Image quality and adequacy\n"
            "2. Key findings\n"
            "3. Abnormalities or areas of concern\n"
            "4. Clinical significance and urgency\n"
            "5. Recommended specialty for follow-up\n"
            "6. Findings warranting IMMEDIATE attention\n\n"
            "Be concise but clinically precise."
        )
        self.medgemma.reset()
        try:
            report = self.medgemma.ask(prompt, images=image_source)
            result = {
                "status": "success", "image_source": image_source,
                "image_type": image_type, "report": report,
                "analyzed_at": datetime.utcnow().isoformat(),
            }
        except Exception as exc:
            logger.error("[PatientAgent] imaging error: %s", exc, exc_info=True)
            result = {
                "status": "error", "image_source": image_source,
                "image_type": image_type,
                "report": f"Analysis failed: {exc}",
                "analyzed_at": datetime.utcnow().isoformat(),
            }
        self.imaging_results.append(result)
        return result

    # ── tool dispatch ─────────────────────────────────────────────────────────

    def available_function(self, function_name: str):
        _map = {
            "collect_patient_information": self.collect_patient_information,
            "get_medical_assessment":      self.get_medical_assessment,
            "analyze_uploaded_image":      self.analyze_uploaded_image,
        }
        if function_name not in _map:
            raise ValueError(f"[PatientAgent] unknown tool: {function_name!r}")
        return _map[function_name]

    # ── tool schemas ──────────────────────────────────────────────────────────

    def _build_tool_schemas(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "collect_patient_information",
                    "description": (
                        "Collect patient demographics and clinical intake data. "
                        "Must be called FIRST before triage or imaging."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "patient_name":        {"type": "string", "description": "Full name"},
                            "patient_phone":       {"type": "string", "description": "Phone number"},
                            "symptoms":            {"type": "string", "description": "Current symptoms"},
                            "duration":            {"type": "string", "description": "Symptom duration"},
                            "severity":            {"type": "string", "description": "mild | moderate | severe"},
                            "medical_history":     {"type": "string", "description": "Past medical history"},
                            "current_medications": {"type": "string", "description": "Current medications"},
                        },
                        "required": ["patient_name", "patient_phone", "symptoms"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_medical_assessment",
                    "description": (
                        "Use MedGemma to perform structured clinical triage. "
                        "Returns urgency, specialty, conditions, red flags."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symptoms":            {"type": "string", "description": "Symptom description"},
                            "duration":            {"type": "string", "description": "Duration"},
                            "severity":            {"type": "string", "description": "mild | moderate | severe"},
                            "medical_history":     {"type": "string", "description": "History"},
                            "current_medications": {"type": "string", "description": "Medications"},
                        },
                        "required": ["symptoms"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_uploaded_image",
                    "description": (
                        "Use MedGemma vision to analyse a medical image "
                        "(X-ray, MRI, CT, skin photo, etc.)."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "image_source":      {"type": "string", "description": "URL or file path"},
                            "image_type":        {"type": "string", "description": "e.g. 'Chest X-Ray'"},
                            "clinical_question": {"type": "string", "description": "Focused question"},
                        },
                        "required": ["image_source"],
                    },
                },
            },
        ]

    # ── data export (used by orchestrator) ────────────────────────────────────

    def get_handoff_data(self) -> dict:
        """Return all collected data for handoff to the Doctor Agent."""
        return {
            "patient_info":    self.patient_info,
            "triage_result":   self.triage_result,
            "imaging_results": self.imaging_results,
        }

    def reset_state(self):
        """Full reset for a new patient."""
        self.reset()                # conversation
        self.patient_info    = {}
        self.triage_result   = {}
        self.imaging_results = []
        logger.info("[PatientAgent] state reset")
