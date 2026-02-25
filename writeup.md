# MedGemma Care Pathway Engine

**Turning Medical Knowledge into Executable Patient Workflows**

---

**Team**

| Member | Role |
|--------|------|
| **Dr. Rachid Zeghlache** (France) | AI Engineer — Agentic system architecture and workflow automation |
| **Dr. Yassine Mamou** (France) | Nuclear Medicine Consultant — Clinical design and guideline-based care pathway modeling |

**Source:** [github.com/YouvenZ/MedFlowAssit](https://github.com/YouvenZ/MedFlowAssit)

---

## 1. Problem Statement

### Problem Domain

Modern medicine does not suffer from a lack of knowledge — clinical guidelines, specialist expertise, and imaging technology all exist. What is missing is the **workflow that connects them together**.

*For patients,* healthcare pathways are fragmented and opaque. Navigating referrals, coordinating appointments, and understanding next steps creates stress and confusion. These barriers lead to **delayed diagnoses**, **loss to follow-up**, and in some cases, **renunciation of care entirely**.

*For physicians,* the challenge is cognitive overload. Clinicians must simultaneously collect incomplete information, interpret guidelines, select tests, coordinate specialists, complete documentation, and track patient progress. Administrative tasks consume an estimated **50% of physician working time**, reducing the time available for direct patient care.

The result: patients do not receive the right care at the right time, and doctors cannot focus on clinical reasoning.

### Impact Potential

**For patients:**
- Continuous, guided navigation through care pathways
- Reduced diagnostic and treatment delays
- Fewer missed follow-ups and lost referrals
- Earlier treatment in time-critical conditions — directly improving life-saving outcomes

**For physicians:**
- Automated information extraction and clinical summarization
- Guideline-based recommendations with automatic identification of missing data
- Automated documentation, scheduling, and specialist coordination
- Reduced cognitive load, enabling physicians to focus on the patient

**For healthcare systems:**
Optimized pathways reduce redundant examinations, shorten hospital stays, and ensure guideline-adherent decisions — directly **lowering costs** while **improving quality of care**.

---

## 2. Overall Solution

We built a **full-stack clinical assistant** that combines two HAI-DEF models into two complementary use cases: an *agentic medical assistant* and a *deterministic clinical workflow engine*.

### HAI-DEF Models Used

> **🫀 MedGemma 1.5 4B-IT** — Local vision-language model for clinical reasoning, triage, image analysis, and SOAP note generation.

> **🎙️ MedASR** — Medical speech-to-text (Conformer-CTC, 105M params, trained on 5,000h of medical dictations) for voice-driven clinical note entry.

### Cloud based model 

> **☁️ OpenAI OSS 120B via Groq** — Cloud LLM for agentic orchestration, tool-call dispatch, data extraction, and patient-facing communication.

### Use Case 1: Medical Assistant

An **agent-based conversational system** that guides the patient from first contact to scheduled appointment:

1. **PatientAgent** collects demographics and symptoms through empathetic dialogue, then invokes **MedGemma for symptom triage** (urgency classification, differential diagnosis, red flags, recommended specialty) and **medical image analysis** (X-rays, fundus photos, skin lesions).
2. An **Orchestrator** detects when intake is complete and performs an automatic **handoff** of structured patient data.
3. **DoctorAgent** uses **MedGemma to generate a SOAP clinical note**, queries a scheduling database filtered by specialty, and books the appointment.
4. **PaperworkAgent** produces appointment summaries, referral letters, and follow-up reminders.

MedGemma's medical pre-training is critical here: general-purpose LLMs lack the calibration to reliably distinguish IMMEDIATE from ROUTINE urgency or interpret radiological findings from raw images.

### Use Case 2: Clinical Workflow Engine

We think the agentic framework it correct for the data collection and triage since it interact with the patient a needs this freedom and flexibilty to cover a wide range of usecase. But in order to help doctor reliably, controled and predinined sequencial agentic AI workflow are more suitable due to their expected and stardard output. In our case we provided some example of workflow that can be used.

A library of **13 protocol-adherent workflows** spanning 10 specialties:

| Workflow | Protocol | Input |
|----------|----------|-------|
| RetinaCounter | ICDR Scale | Image |
| StrokeRisk+ | CHA₂DS₂-VASc | Text/PDF |
| Med-Rec Guard | AGS Beers 2023 | Img/PDF |
| TriageFlag | ACR Actionable | Text/PDF |
| GrowthPlotter | CDC/WHO Charts | Text |
| ConsultScribe | SOAP + ICD-10 | Text |
| qSOFA Calc | Sepsis-3 | Text/PDF |
| WellsPE | Wells Criteria | Text/PDF |
| MELD-Na Calc | UNOS 2016 | Text/PDF |
| GCS Calc | Glasgow Coma | Text/PDF |
| LabFlagr | Ref. Ranges | CSV |
| RxInteract | Drug Interaction DB | CSV |
| VitalsTrend | NEWS2 (RCP) | CSV |

Every workflow follows a **chain-based architecture**:

```
Input → LLM Extraction → Deterministic Logic → JSON
```

The LLM extracts structured variables from free text; all scoring, grading, and flagging is performed by **deterministic Python logic** using published clinical formulae. Three CSV workflows (LabFlagr, RxInteract, VitalsTrend) use **zero LLM calls** — purely algorithmic.

### Why HAI-DEF Models Are the Right Choice

- **Privacy:** MedGemma and MedASR run *locally* — patient images, audio, and PHI never leave the host. This enables deployment in air-gapped clinical environments.
- **Medical calibration:** MedGemma's medical pre-training enables reliable clinical triage, structured imaging reports, and SOAP note generation where general-purpose models would be less effective.
- **Voice input:** MedASR's vocabulary (drug names, anatomical terms, abbreviations) outperforms general ASR on medical dictation, reducing transcription errors that propagate into clinical decisions.

---

## 3. Technical Details

### Architecture

The system is a single Flask application serving three interfaces:

| Page | Route | Purpose |
|------|-------|---------|
| Scenarios | `/` | 12 pre-built clinical scenarios |
| Free Chat | `/chat` | Open-ended multi-turn chat |
| Workflows | `/workflows` | 13-card grid with execution panel |

**Multi-agent pipeline:**
The `Orchestrator` routes messages between `PatientAgent` (intake phase) and `DoctorAgent` (scheduling phase) using keyword-scoring intent classification. Each agent extends `BaseAgent`, which implements a reusable **tool-calling loop**: the LLM receives tool schemas, invokes functions, receives results, and iterates up to 12 rounds until producing a final response.

**Workflow engine:**
All 13 workflows inherit from `ClinicalWorkflow` and implement `validate_input()` and `execute()`. Output is a standard `WorkflowResult` JSON schema (summary, metrics, protocol adherence, raw report).

### Model Deployment

| Model | Size | Device | Role |
|-------|------|--------|------|
| MedGemma 4B-IT | ~4 GB | Local GPU | Triage, vision, notes |
| MedASR | ~400 MB | CPU/GPU | Speech-to-text |
| OSS 120B (Groq) | API | Cloud | Orchestration, extraction |

MedGemma is loaded via `transformers.pipeline("image-text-to-text")` with **NF4 quantization** (`bitsandbytes`, double quantization, bf16 compute), fitting in ≤ 8 GB VRAM. A singleton pattern ensures the model is loaded once and shared across agents. MedASR loads lazily as a HuggingFace ASR pipeline; browser audio (WebM) is converted to 16 kHz mono WAV via `pydub`/`ffmpeg`.

### Key Design Decisions

- **Hybrid local/cloud:** Clinical reasoning runs on-premise (MedGemma, MedASR); orchestration runs via Groq's low-latency API. Patient data stays local.
- **Deterministic scoring:** LLMs extract variables; published formulae compute scores. This guarantees reproducibility — identical inputs always produce identical clinical scores.
- **23 specialty prompts:** `specialty.py` provides expert-level system prompts for every medical specialty, ensuring MedGemma operates with the appropriate clinical lens.
- **Multi-input support:** Text, image (drag-and-drop), PDF (server-side extraction via `pypdf`), CSV, and voice (**MedASR**) are all first-class input modalities.
- **SQLite scheduling:** 44 seeded doctors across 22 specialties with 14 days of time slots enable realistic appointment management within the demo.

### Feasibility & Deployment

The entire system installs with `pip install -r requirements.txt` and a single `GROQ_API_KEY` environment variable. Models auto-download from HuggingFace on first run. MedGemma inference averages < 5s on a consumer GPU; Groq API calls add ~1–2s per turn.

**Limitations & future work:**
MedGemma at 4B parameters occasionally produces malformed JSON (handled by fallback parsing). Future iterations would add FHIR/HL7 integration for real EMR connectivity, institution-specific fine-tuning, and role-based access control.

---

> **Source:** [github.com/YouvenZ/MedFlowAssit](https://github.com/YouvenZ/MedFlowAssit) — fully reproducible, documented, tested.
