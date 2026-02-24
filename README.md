# MedGemma Clinical Workflow Engine

A multi-agent medical assistant powered by **MedGemma** (vision-language) and **LiteLLM** (Groq), featuring a deterministic **Clinical Workflow Engine** with 13 specialty chains, voice dictation via **MedASR**, and file upload support (image, PDF & CSV).

---

## Features

| Page | Route | Description |
|------|-------|-------------|
| **Scenarios** | `/` | Pre-built clinical scenarios (headache triage, chest X-ray, emergency, etc.) |
| **Free Chat** | `/chat` | Open-ended multi-turn medical conversation |
| **Workflows** | `/workflows` | 3×N grid of protocol-adherent clinical workflows |

### Clinical Workflows

#### Text / PDF — Medicine Rules & Scoring

| Workflow | Specialty | Protocol | Input | Description |
|----------|-----------|----------|-------|-------------|
| **RetinaCounter** | Ophthalmology | ICDR Scale | Image | Diabetic retinopathy grading from fundus images |
| **StrokeRisk+** | Cardiology | CHA₂DS₂-VASc | Text, PDF | Stroke risk calculator — 7 factor scoring |
| **Med-Rec Guard** | Geriatrics | AGS Beers 2023 | Image, Text, PDF | Medication reconciliation against Beers Criteria |
| **TriageFlag** | Radiology | ACR Actionable | Text, PDF | Critical finding triage — 3-level urgency |
| **GrowthPlotter** | Pediatrics | CDC/WHO Charts | Text | Growth percentile calculator (BMI, weight, height) |
| **ConsultScribe** | General Practice | SOAP + ICD-10 | Text | SOAP note generator with ICD-10 coding |
| **qSOFA Calc** | Emergency Medicine | qSOFA (Sepsis-3) | Text, PDF | Quick SOFA sepsis screening — 3 bedside criteria |
| **WellsPE** | Pulmonology | Wells Criteria | Text, PDF | Pulmonary embolism probability — 7-item rule |
| **MELD-Na Calc** | Hepatology | MELD-Na (UNOS 2016) | Text, PDF | Liver disease severity & transplant priority |
| **GCS Calc** | Neurology | Glasgow Coma Scale | Text, PDF | E+V+M neurological assessment with TBI grade |

#### CSV-Based — Tabular Healthcare Data

| Workflow | Specialty | Protocol | Input | Description |
|----------|-----------|----------|-------|-------------|
| **LabFlagr** | Pathology / Lab | Standard Reference Ranges | CSV | Flag abnormals, critical values, compute derived metrics (anion gap, eGFR, corrected Ca) |
| **RxInteract** | Pharmacy | Drug Interaction DB | CSV | Pairwise drug-drug interaction check — HIGH/MODERATE/LOW severity |
| **VitalsTrend** | ICU / Nursing | NEWS2 (RCP) | CSV | Serial vital signs → NEWS2 scoring + deterioration trend detection |

All workflows follow a **chain-based** (not agentic) architecture:

```
Input → LLM Extraction / CSV Parse → Deterministic Python Logic → Structured JSON Output
```

### Input Methods

- **Text** — free-text clinical notes
- **Image upload** — drag-and-drop or file picker (PNG, JPG, TIFF, WebP, etc.)
- **PDF upload** — drag-and-drop; server extracts text via `pypdf`
- **CSV upload** — drag-and-drop or paste; supports flexible column names
- **Voice dictation** — browser microphone → MedASR (HuggingFace `google/medasr`, Conformer-CTC, 105M params, 16 kHz mono)

---

## Quick Start

### 1. Clone & install

```bash
git clone <repo-url>
cd MedGemmaChallenge

# Create environment (conda or venv)
conda create -n medgemma python=3.11 -y
conda activate medgemma

pip install -r requirements.txt
```

### 2. Download model weights (optional — pre-download for offline use)

```bash
python download_and_test.py --download-only
```

This pre-downloads **MedGemma** (~4 GB) and **MedASR** (~400 MB) from HuggingFace. Models are cached in `~/.cache/huggingface/` and only need to be downloaded once.

### 3. Install ffmpeg (required for voice dictation with non-WAV audio)

```bash
# Windows (chocolatey)
choco install ffmpeg

# macOS
brew install ffmpeg

# Linux
sudo apt install ffmpeg
```

### 4. Configure environment

Create a `.env` file in the project root:

```env
# Required — LLM API key (Groq recommended)
GROQ_API_KEY=gsk_...

# Optional — MedASR model override
# MEDASR_MODEL=google/medasr
# MEDASR_DEVICE=cpu          # or "cuda"
```

### 5. Run

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

### 6. Run tests (optional — verify everything works)

```bash
# Test LLM API + all app endpoints (app must be running)
python download_and_test.py --test-all

# Only test API endpoints (skip model downloads)
python download_and_test.py --test-api

# Only test LLM endpoints
python download_and_test.py --test-llm
```

---

## Project Structure

```
MedGemmaChallenge/
├── app.py                  # Flask application — all routes
├── main.py                 # CLI entry point (terminal mode)
├── llm_config.py           # LiteLLM wrapper (llm_completion, DEFAULT_MODEL)
├── base_agent.py           # Abstract BaseAgent class
├── medical_agent.py        # MedicalAppointmentAgent (orchestrator facade)
├── orchestrator.py         # Multi-agent orchestrator (patient ↔ doctor)
├── patient_agent.py        # Patient intake agent
├── doctor_agent.py         # Doctor triage & booking agent
├── paperwork_agent.py      # Clinical documentation agent
├── medgemma_client.py      # MedGemma vision-language pipeline (4B-IT)
├── specialty.py            # Specialty enum + system prompts (23 specialties)
├── db.py                   # SQLite database (doctors, slots, appointments)
├── medasr.py               # MedASR speech-to-text module
├── pdf_extract.py          # PDF text extraction (pypdf / PyPDF2 / pdfplumber)
├── requirements.txt        # Python dependencies
├── download_and_test.py    # Model downloader + API endpoint tester
├── workflows/
│   ├── __init__.py         # Workflow registry (get_all_cards, run_workflow)
│   ├── base.py             # ClinicalWorkflow ABC, WorkflowResult, InputType
│   ├── retina_counter.py   # ICDR diabetic retinopathy grading
│   ├── stroke_risk.py      # CHA₂DS₂-VASc stroke risk calculator
│   ├── medrec_guard.py     # AGS Beers Criteria medication reconciliation
│   ├── triage_flag.py      # ACR critical finding triage
│   ├── growth_plotter.py   # CDC/WHO growth percentile calculator
│   ├── consult_scribe.py   # SOAP note + ICD-10 generator
│   ├── qsofa_calc.py       # qSOFA sepsis screening (Sepsis-3)
│   ├── wells_pe.py         # Wells criteria for pulmonary embolism
│   ├── meld_calc.py        # MELD-Na liver disease score (UNOS 2016)
│   ├── gcs_calc.py         # Glasgow Coma Scale (E+V+M)
│   ├── lab_flagr.py        # CSV lab panel analyzer (100+ reference ranges)
│   ├── rx_interact.py      # CSV drug interaction checker (40+ pairs)
│   └── vitals_trend.py     # CSV vitals trend + NEWS2 scoring
├── templates/
│   ├── index.html          # Scenarios page
│   ├── chat.html           # Free Chat page
│   └── workflows.html      # Workflow Grid Interface
├── static/
│   └── style.css           # All styles
└── uploads/                # Temporary upload directory (auto-created)
```

---

## API Reference

### Pages

| Method | Route | Description |
|--------|-------|-------------|
| `GET`  | `/` | Scenarios page |
| `GET`  | `/chat` | Free Chat page |
| `GET`  | `/workflows` | Workflow Grid Interface |

### Agent / Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/chat` | Send message to medical agent |
| `POST` | `/api/reset` | Reset agent session |
| `GET`  | `/api/status` | Current agent state |
| `POST` | `/api/scenario/<num>` | Start a pre-built scenario (1-12) |
| `POST` | `/api/paperwork` | Generate clinical documentation |

### Workflows

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/api/workflows` | List all 13 workflow cards (metadata) |
| `POST` | `/api/workflows/run` | Execute a workflow by `workflow_id` |

**Request body** (`/api/workflows/run`):

```json
{
  "workflow_id": "stroke_risk",
  "text": "72-year-old male with atrial fibrillation, hypertension, diabetes…"
}
```

**CSV workflow example**:

```json
{
  "workflow_id": "lab_flagr",
  "csv_text": "test,value,unit\nHemoglobin,8.5,g/dL\nWBC,15.2,×10³/µL\nPotassium,6.1,mEq/L"
}
```

**Response** (standard schema for all workflows):

```json
{
  "workflow_id": "stroke_risk",
  "status": "success",
  "data": {
    "summary": "CHA₂DS₂-VASc Score: 4/9. Estimated annual stroke risk: 4.8%.",
    "metrics": { "score": "4/9", "annual_stroke_risk": "4.8%" },
    "protocol_adherence": true,
    "raw_output": "═══ CHA₂DS₂-VASc STROKE RISK REPORT ═══\n..."
  },
  "artifacts": []
}
```

### Upload & Transcription

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload/image` | Upload image → base64 data-URI |
| `POST` | `/api/upload/pdf` | Upload PDF → extracted text |
| `POST` | `/api/upload/csv` | Upload CSV → raw text content |
| `POST` | `/api/transcribe` | Audio → MedASR text (file or base64 JSON) |
| `GET`  | `/api/medasr/status` | MedASR model availability check |

---

## Architecture

### Multi-Agent System (Scenarios & Chat)

```
User → Orchestrator → PatientAgent (intake)
                    → DoctorAgent  (triage + booking)
                    → PaperworkAgent (notes)
                    → MedGemmaClient (vision analysis)
```

### Workflow Engine (deterministic chains)

```
UI Card click → /api/workflows/run
             → WorkflowRegistry.run_workflow(id, data)
             → ClinicalWorkflow.run()
                 ├── validate_input()
                 ├── execute()
                 │     ├── Step 1: LLM extraction / CSV parse
                 │     └── Step 2: Deterministic Python logic
                 └── WorkflowResult (standard JSON schema)
```

### Models

| Model | Source | Size | Purpose | Device |
|-------|--------|------|---------|--------|
| **MedGemma 1.5 4B-IT** | `google/medgemma-1.5-4b-it` | ~4 GB | Vision-language medical analysis | GPU (≥8 GB VRAM) |
| **MedASR** | `google/medasr` | ~400 MB | Medical speech-to-text (Conformer-CTC) | CPU or GPU |
| **GPT-OSS 120B** (via Groq) | `groq/openai/gpt-oss-120b` | API | LLM extraction, triage, SOAP notes | Cloud API |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | Groq API key (required for LLM workflows) |
| `MEDASR_MODEL` | `google/medasr` | HuggingFace model ID for speech-to-text |
| `MEDASR_DEVICE` | `cpu` | Torch device for MedASR (`cpu` or `cuda`) |

---

## Testing & Validation

```bash
# Full suite: download models + test LLM + test all API endpoints
python download_and_test.py --test-all

# Individual test modes:
python download_and_test.py --download-only    # Pre-download HF models
python download_and_test.py --test-llm         # Test Groq LLM API only
python download_and_test.py --test-api          # Test all Flask endpoints
python download_and_test.py --test-workflows    # Test all 13 workflow executions
```

---

## Notes

- **MedASR** downloads model weights (~400 MB) on first use. Subsequent starts are instant.
- **MedGemma vision** (`medgemma_client.py`) requires a GPU with ≥8 GB VRAM for quantized inference.
- **CSV workflows** (LabFlagr, RxInteract, VitalsTrend) are fully deterministic — no LLM calls needed.
- All workflows produce a **standard JSON output schema** (`WorkflowResult`) regardless of specialty.
- The SQLite database (`medical_appointments.db`) is auto-created on first run with seed data for doctors and time slots.
- Voice dictation works in Chrome, Edge, and Firefox (MediaRecorder API required).

---

## License

This project is for educational and research purposes.
