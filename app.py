"""
app.py — Minimalist Flask web UI for the multi-agent medical system.

Run:
    python app.py
    → http://localhost:5000
"""

from __future__ import annotations

import base64
import json
import logging
import os
import uuid
from datetime import datetime

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, session

load_dotenv()

from medical_agent import MedicalAppointmentAgent
from paperwork_agent import PaperworkAgent
from llm_config import DEFAULT_MODEL
from workflows import get_all_cards, run_workflow
from medasr import transcribe_audio, is_medasr_available
from pdf_extract import extract_text_from_pdf

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "medgemma-demo-key-change-in-prod"

# ── In-memory agent store (keyed by session id) ──────────────────────────────
_agents: dict[str, MedicalAppointmentAgent] = {}

SCENARIOS: dict[int, dict] = {
    1:  {"title": "Headache + triage",               "icon": "🧠", "tag": "routine",
         "desc": "Severe headache with nausea — triage, note, booking"},
    2:  {"title": "Chest X-ray analysis",             "icon": "🫁", "tag": "imaging",
         "desc": "Chest pain + uploaded X-ray image → vision analysis"},
    3:  {"title": "Emergency (IMMEDIATE)",            "icon": "🚨", "tag": "emergency",
         "desc": "Acute MI symptoms — agent redirects to ER"},
    4:  {"title": "Full appointment lifecycle",       "icon": "📋", "tag": "management",
         "desc": "Book → retrieve → reschedule → cancel"},
    5:  {"title": "Multi-image + multi-symptom",      "icon": "🖼️", "tag": "imaging",
         "desc": "Rash photo + knee X-ray — two images analysed"},
    6:  {"title": "Pediatric case",                   "icon": "👶", "tag": "pediatric",
         "desc": "Parent booking for child with fever"},
    7:  {"title": "Diabetes follow-up",               "icon": "💉", "tag": "chronic",
         "desc": "Chronic condition medication review"},
    8:  {"title": "Gradual info collection",          "icon": "💬", "tag": "routine",
         "desc": "Patient provides info step-by-step"},
    9:  {"title": "Ophthalmology referral",           "icon": "👁️", "tag": "specialty",
         "desc": "Blurry vision → specialty-filtered slots"},
    10: {"title": "Polypharmacy / elderly",           "icon": "💊", "tag": "urgent",
         "desc": "75 y/o on 6 meds — drug interaction concern"},
    11: {"title": "Mental health",                    "icon": "🧘", "tag": "mental",
         "desc": "Anxiety & panic attacks → psychiatry"},
    12: {"title": "Preventive checkup",               "icon": "✅", "tag": "routine",
         "desc": "Healthy patient, family hx of heart disease"},
}

# Pre-built scenario messages (mirroring main.py)
SCENARIO_MESSAGES: dict[int, list[str]] = {
    1: [
        "I have a severe headache that started 2 days ago, it's getting worse and I have nausea.",
        "My name is Rachid Zeghlache, phone 0676574873. I have a history of migraines and I'm currently on sumatriptan.",
    ],
    2: [
        "Hi, I've been having chest pain and shortness of breath for 3 days. I also have this chest X-ray from last week: https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png",
        "My name is Sara Benali, phone 0655443322. I'm 58, hypertension, on amlodipine 5mg.",
    ],
    3: [
        "I'm having crushing chest pain radiating to my left arm, severe shortness of breath, sweating heavily. Started 20 minutes ago.",
        "My name is Ahmed Khalil, phone 0698765432. I'm 62, diabetic, smoker, on metformin.",
    ],
    4: [
        "I have persistent lower back pain for 2 weeks, mild but constant. No numbness or weakness.",
        "I'm Fatima El Mansouri, phone 0634567890. No significant medical history, no medications.",
        "Can you show me the details of my appointment?",
        "Actually, can I reschedule to a different time?",
        "Never mind, I need to cancel the appointment.",
    ],
    5: [
        "I have a skin rash on my arm and joint pain in my knees. Rash for 5 days, joints 2 weeks. Here's the rash: https://example.com/rash_photo.jpg",
        "I also got knee X-rays: https://example.com/knee_xray.jpg",
        "My name is Karim Touati, phone 0687654321. Rheumatoid arthritis, methotrexate 15mg weekly.",
    ],
    6: [
        "My 6-year-old daughter has had a high fever (39.5 C) for 3 days, very tired, sore throat, refusing to eat.",
        "I'm her mother, Salma Mansouri, phone 0612348765. Daughter Lina, no medical history, no allergies.",
    ],
    7: [
        "I need a follow-up for my diabetes. Last visit 3 months ago. Fasting glucose around 180 mg/dL. On metformin 1000mg BID and glimepiride 2mg.",
        "I'm Hassan Bennani, phone 0678901234. Diabetic 8 years, hypertensive on lisinopril 10mg.",
    ],
    8: [
        "I need to see a doctor.",
        "I have stomach pain.",
        "It's been about a week, moderate pain.",
        "I'm Youssef Amrani, 0645678901.",
    ],
    9: [
        "I've been experiencing blurry vision and eye pain for 4 days. Especially with bright lights.",
        "I'm Nadia Benjelloun, phone 0656789012. Glasses for myopia, no other issues.",
    ],
    10: [
        "I'm feeling very dizzy and tired all the time. Started a week ago. Some nausea.",
        "I'm Mohamed Alami, phone 0667890123. I'm 75. Meds: warfarin 5mg, metoprolol 50mg BID, furosemide 40mg, atorvastatin 40mg, omeprazole 20mg, aspirin 100mg daily.",
    ],
    11: [
        "I've been very anxious with panic attacks for the past month. Poor sleep, trouble concentrating.",
        "I'm Leila Fassi, phone 0689012345. No medical conditions, no meds. Stressful work situation.",
    ],
    12: [
        "I'd like a routine health checkup. Haven't seen a doctor in 2 years, turning 50 next month.",
        "I'm Driss Hammoudi, phone 0690123456. Generally healthy, no meds. Family history of heart disease.",
    ],
}


def _get_agent() -> MedicalAppointmentAgent:
    """Get or create the agent for the current session."""
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    sid = session["sid"]
    if sid not in _agents:
        _agents[sid] = MedicalAppointmentAgent()
        logger.info("Created agent for session %s", sid[:8])
    return _agents[sid]


# ══════════════════════════════════════════════════════════════════════════════
#  Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html", scenarios=SCENARIOS)


@app.route("/chat")
def chat():
    """Simple full-width chat — no sidebar, no scenarios."""
    return render_template("chat.html")


@app.route("/workflows")
def workflows_page():
    """Workflow Grid Interface — Page 3."""
    import json as _json
    cards = get_all_cards()
    return render_template(
        "workflows.html",
        workflows=cards,
        workflows_json=_json.dumps(cards),
    )


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Send a message and get the assistant reply."""
    data = request.get_json(force=True)
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "Empty message"}), 400

    agent = _get_agent()
    try:
        reply = agent.chat(message)
    except Exception as exc:
        logger.error("Chat error: %s", exc, exc_info=True)
        reply = f"An error occurred: {exc}"

    # Build status snapshot
    status = _build_status(agent)
    return jsonify({"reply": reply, "status": status})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    agent = _get_agent()
    agent.reset()
    return jsonify({"ok": True})


@app.route("/api/status")
def api_status():
    agent = _get_agent()
    return jsonify(_build_status(agent))


@app.route("/api/scenario/<int:num>", methods=["POST"])
def api_scenario(num: int):
    """Run a pre-built scenario, returning all messages."""
    if num not in SCENARIO_MESSAGES:
        return jsonify({"error": f"Unknown scenario {num}"}), 404

    agent = _get_agent()
    agent.reset()

    msgs = SCENARIO_MESSAGES[num]
    conversation = []
    for msg in msgs:
        try:
            reply = agent.chat(msg)
        except Exception as exc:
            reply = f"Error: {exc}"
        conversation.append({"user": msg, "assistant": reply})

    status = _build_status(agent)
    return jsonify({"conversation": conversation, "status": status})


@app.route("/api/paperwork", methods=["POST"])
def api_paperwork():
    """Generate paperwork documents from current session data."""
    agent = _get_agent()
    if not agent.patient_info:
        return jsonify({"error": "No patient data yet"}), 400

    pw = PaperworkAgent()
    pw.set_context(
        patient_info=agent.patient_info,
        triage_result=agent.triage_result or {},
        clinical_note=agent.clinical_note or "",
        imaging_results=agent.imaging_results or [],
    )

    docs = {}
    try:
        docs["summary"]  = pw.generate_appointment_summary().get("summary", "")
    except Exception:
        docs["summary"] = ""
    try:
        docs["referral"] = pw.generate_referral_letter().get("letter", "")
    except Exception:
        docs["referral"] = ""
    try:
        docs["reminder"] = pw.generate_followup_reminder().get("reminder", "")
    except Exception:
        docs["reminder"] = ""

    return jsonify(docs)


# ══════════════════════════════════════════════════════════════════════════════
#  Workflow API
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/workflows", methods=["GET"])
def api_workflows_list():
    """Return metadata for all registered clinical workflows."""
    return jsonify(get_all_cards())


@app.route("/api/workflows/run", methods=["POST"])
def api_workflows_run():
    """Execute a workflow by workflow_id with the provided input data."""
    data = request.get_json(force=True)
    wf_id = data.pop("workflow_id", "")
    if not wf_id:
        return jsonify({"status": "error",
                        "data": {"summary": "Missing workflow_id"}}), 400
    result = run_workflow(wf_id, data)
    return jsonify(result)


# ══════════════════════════════════════════════════════════════════════════════
#  Upload / Transcribe / PDF Extraction API
# ══════════════════════════════════════════════════════════════════════════════

# Upload directory for temporary files
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.route("/api/upload/csv", methods=["POST"])
def api_upload_csv():
    """Upload a CSV file → return the text content for CSV-based workflows."""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files["file"]
    if not f or f.filename == "":
        return jsonify({"error": "No file selected"}), 400

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in (".csv", ".tsv", ".txt"):
        return jsonify({"error": f"Expected CSV file, got {ext}"}), 400

    raw = f.read()
    # Try UTF-8 first, then latin-1 as fallback
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    logger.info("CSV uploaded: %s (%d bytes, %d chars)", f.filename, len(raw), len(text))
    row_count = text.strip().count("\n")
    return jsonify({"csv_text": text, "filename": f.filename, "chars": len(text), "rows": row_count})


@app.route("/api/upload/image", methods=["POST"])
def api_upload_image():
    """Upload an image file → return a data-URI (base64) for the workflow."""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files["file"]
    if not f or f.filename == "":
        return jsonify({"error": "No file selected"}), 400

    allowed = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff"}
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in allowed:
        return jsonify({"error": f"Unsupported image type: {ext}"}), 400

    raw = f.read()
    mime = f.content_type or f"image/{ext.lstrip('.')}"
    b64 = base64.b64encode(raw).decode()
    data_uri = f"data:{mime};base64,{b64}"
    logger.info("Image uploaded: %s (%d bytes)", f.filename, len(raw))
    return jsonify({"image_url": data_uri, "filename": f.filename, "size": len(raw)})


@app.route("/api/upload/pdf", methods=["POST"])
def api_upload_pdf():
    """Upload a PDF → extract text content and return it."""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files["file"]
    if not f or f.filename == "":
        return jsonify({"error": "No file selected"}), 400

    ext = os.path.splitext(f.filename)[1].lower()
    if ext != ".pdf":
        return jsonify({"error": f"Expected PDF, got {ext}"}), 400

    raw = f.read()
    try:
        text = extract_text_from_pdf(raw)
    except Exception as exc:
        logger.error("PDF extraction failed: %s", exc)
        return jsonify({"error": f"PDF extraction failed: {exc}"}), 500

    logger.info("PDF extracted: %s → %d chars", f.filename, len(text))
    return jsonify({"pdf_text": text, "filename": f.filename, "chars": len(text)})


@app.route("/api/transcribe", methods=["POST"])
def api_transcribe():
    """
    Transcribe audio using MedASR.
    Accepts:
      • multipart file upload  (field name: 'audio')
      • JSON body with base64  ({"audio_b64": "...", "content_type": "audio/wav"})
    """
    # ── File upload path ──────────────────────────────────────────────────────
    if "audio" in request.files:
        f = request.files["audio"]
        audio_bytes = f.read()
        ct = f.content_type or "audio/wav"
    # ── JSON base64 path ─────────────────────────────────────────────────────
    elif request.is_json:
        data = request.get_json(force=True)
        b64 = data.get("audio_b64", "")
        if not b64:
            return jsonify({"error": "No audio data provided"}), 400
        audio_bytes = base64.b64decode(b64)
        ct = data.get("content_type", "audio/wav")
    else:
        return jsonify({"error": "Send audio as file upload or JSON base64"}), 400

    if len(audio_bytes) < 100:
        return jsonify({"error": "Audio too short"}), 400

    try:
        text = transcribe_audio(audio_bytes, content_type=ct)
    except Exception as exc:
        logger.error("Transcription error: %s", exc, exc_info=True)
        return jsonify({"error": f"Transcription failed: {exc}"}), 500

    return jsonify({"text": text, "model": "MedASR", "chars": len(text)})


@app.route("/api/medasr/status")
def api_medasr_status():
    """Health check for MedASR availability."""
    return jsonify({"available": is_medasr_available()})


def _build_status(agent: MedicalAppointmentAgent) -> dict:
    """Snapshot of agent state for the UI sidebar."""
    pi = agent.patient_info or {}
    tr = agent.triage_result or {}
    return {
        "phase": agent.phase.value if hasattr(agent.phase, "value") else str(agent.phase),
        "patient_name": pi.get("name", ""),
        "patient_phone": pi.get("phone", ""),
        "symptoms": pi.get("symptoms", ""),
        "urgency": tr.get("urgency_level", ""),
        "specialty": tr.get("recommended_specialty", ""),
        "conditions": tr.get("possible_conditions", []),
        "red_flags": tr.get("red_flags", []),
        "has_note": bool(agent.clinical_note),
        "imaging_count": len(agent.imaging_results) if agent.imaging_results else 0,
    }


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
