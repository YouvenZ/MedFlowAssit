from __future__ import annotations

import json
import time
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Generator, Iterator
from specialty import Specialty, get_system_prompt

import requests
import torch
from PIL import Image
from transformers import pipeline, BitsAndBytesConfig, TextStreamer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Quantization — full menu
# ══════════════════════════════════════════════════════════════════════════════

class QuantMode(str, Enum):
    """All supported quantization modes."""
    NF4      = "nf4"       # 4-bit NF4  + double-quant  (best quality @ 4-bit)
    FP4      = "fp4"       # 4-bit FP4                  (faster, slightly lower)
    INT8     = "int8"      # 8-bit LLM.int8              (balanced)
    BF16     = "bf16"      # bfloat16  — no BnB          (good GPUs)
    FP16     = "fp16"      # float16   — no BnB          (CUDA standard)
    FP32     = "fp32"      # full precision              (CPU or large GPU)
    NONE     = "none"      # alias for fp32 / auto

    @classmethod
    def values(cls) -> list[str]:
        return [m.value for m in cls]


def get_bnb_config(mode: str | QuantMode = QuantMode.NF4) -> BitsAndBytesConfig | None:
    """
    Build a BitsAndBytesConfig for the given mode, or return None.

    Modes that do NOT use BnB (bf16 / fp16 / fp32 / none) return None;
    the caller should pass torch_dtype to the pipeline directly instead.

    Args:
        mode: One of QuantMode values or its string equivalent.

    Returns:
        BitsAndBytesConfig or None.
    """
    m = QuantMode(str(mode).lower())

    if m == QuantMode.NF4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    if m == QuantMode.FP4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype=torch.float16,
        )
    if m == QuantMode.INT8:
        return BitsAndBytesConfig(load_in_8bit=True)

    # bf16 / fp16 / fp32 / none — handled via torch_dtype, no BnB needed
    return None


def quant_mode_to_torch_dtype(mode: str | QuantMode) -> torch.dtype:
    """Return the appropriate torch dtype for non-BnB modes."""
    m = QuantMode(str(mode).lower())
    mapping = {
        QuantMode.NF4:  torch.bfloat16,   # BnB computes in bf16
        QuantMode.FP4:  torch.float16,
        QuantMode.INT8: torch.float16,
        QuantMode.BF16: torch.bfloat16,
        QuantMode.FP16: torch.float16,
        QuantMode.FP32: torch.float32,
        QuantMode.NONE: torch.float32,
    }
    return mapping.get(m, torch.bfloat16)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Image loading  (multi-image, metadata, retry)
# ══════════════════════════════════════════════════════════════════════════════

_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


@dataclass
class ImageInfo:
    """Metadata attached to every loaded image."""
    source: str
    width: int
    height: int
    original_mode: str
    image: Image.Image


def load_image(
    source: str | Path | Image.Image,
    retries: int = 3,
    backoff: float = 1.0,
) -> ImageInfo:
    """
    Load a single image from URL, local path, or PIL Image.

    Args:
        source:   URL, file-system path, or PIL Image.
        retries:  Number of HTTP retry attempts on transient failures.
        backoff:  Initial wait (seconds) between retries; doubles each attempt.

    Returns:
        ImageInfo with the RGB PIL Image and metadata.

    Raises:
        ValueError:        Permanent HTTP error or bad source type.
        FileNotFoundError: Local path does not exist.
    """
    if isinstance(source, Image.Image):
        return ImageInfo(
            source="<PIL Image>",
            width=source.width,
            height=source.height,
            original_mode=source.mode,
            image=source.convert("RGB"),
        )

    source_str = str(source)

    if source_str.startswith(("http://", "https://")):
        last_exc: Exception | None = None
        wait = backoff
        for attempt in range(1, retries + 1):
            try:
                resp = requests.get(source_str, headers=_HTTP_HEADERS, timeout=20)
                if resp.status_code == 200:
                    img = Image.open(BytesIO(resp.content))
                    info = ImageInfo(
                        source=source_str,
                        width=img.width,
                        height=img.height,
                        original_mode=img.mode,
                        image=img.convert("RGB"),
                    )
                    logger.info(
                        "Loaded URL image %dx%d  [%s]", img.width, img.height, img.mode
                    )
                    return info
                if resp.status_code in (400, 401, 403, 404):
                    raise ValueError(
                        f"Permanent HTTP {resp.status_code} for URL: {source_str}"
                    )
                last_exc = ValueError(f"HTTP {resp.status_code}")
            except requests.RequestException as e:
                last_exc = e
            logger.warning("Attempt %d/%d failed (%s). Retrying in %.1fs …",
                           attempt, retries, last_exc, wait)
            time.sleep(wait)
            wait *= 2
        raise ValueError(f"Failed to download image after {retries} attempts: {last_exc}")

    p = Path(source_str)
    if not p.exists():
        raise FileNotFoundError(f"Image file not found: {p}")
    img = Image.open(p)
    info = ImageInfo(
        source=str(p),
        width=img.width,
        height=img.height,
        original_mode=img.mode,
        image=img.convert("RGB"),
    )
    logger.info("Loaded local image %dx%d  [%s]  ← %s", img.width, img.height, img.mode, p.name)
    return info


def load_images(
    sources: str | Path | Image.Image | list,
) -> list[ImageInfo]:
    """
    Normalise any image input into a list of ImageInfo objects.

    Accepts:
        • A single URL / path / PIL Image  → [ImageInfo]
        • A list of any mix of the above   → [ImageInfo, ...]
        • None or []                       → []
    """
    if sources is None:
        return []
    if not isinstance(sources, list):
        sources = [sources]
    return [load_image(s) for s in sources]



# ══════════════════════════════════════════════════════════════════════════════
# 4.  Turn & session data classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Turn:
    role: str
    text: str
    image_sources: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    token_count: int = 0   # rough estimate


@dataclass
class Session:
    specialty: str
    turns: list[Turn] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, data: str) -> "Session":
        d = json.loads(data)
        d["turns"] = [Turn(**t) for t in d.get("turns", [])]
        return cls(**d)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  MedGemmaClient
# ══════════════════════════════════════════════════════════════════════════════

class MedGemmaClient:
    """
    Full-featured, stateful client for google/medgemma-1.5-4b-it.

    Key capabilities
    ----------------
    ask()          Single or multi-image turn, returns str reply.
    ask_batch()    Run many independent prompts without touching history.
    stream()       Yield reply tokens progressively (generator).
    reset()        Wipe history and optionally switch specialty.
    save_session() Persist conversation to JSON.
    load_session() Restore conversation from JSON.
    summarise()    Ask the model to summarise the conversation so far.

    Parameters
    ----------
    specialty:       Specialty preset or "custom"  (default: "general")
    custom_prompt:   Required when specialty="custom"
    quantization:    "nf4" | "fp4" | "int8" | "bf16" | "fp16" | "fp32" | "none"
    device_map:      HuggingFace device map  (default: "auto")
    max_new_tokens:  Default response token cap  (default: 512)
    max_history_turns: Truncate history beyond this many turns (default: 20)
    model_id:        Override the default model  (default: medgemma-1.5-4b-it)
    """

    DEFAULT_MODEL = "google/medgemma-1.5-4b-it"

    def __init__(
        self,
        specialty: str | Specialty = Specialty.general,
        custom_prompt: str = "",
        quantization: str | QuantMode = QuantMode.NF4,
        device_map: str = "auto",
        max_new_tokens: int = 512,
        max_history_turns: int = 20,
        model_id: str = DEFAULT_MODEL,
    ):
        self.specialty = Specialty(str(specialty))
        self.system_prompt = get_system_prompt(self.specialty, custom_prompt)
        self.max_new_tokens = max_new_tokens
        self.max_history_turns = max_history_turns
        self._session = Session(specialty=self.specialty.value)
        self.__init_pipeline_history()
        # Build pipeline kwargs
        bnb = get_bnb_config(quantization)
        dtype = quant_mode_to_torch_dtype(quantization)
        model_kwargs = {"quantization_config": bnb} if bnb is not None else {}

        logger.info(
            "Loading %s  [specialty=%s  quant=%s]",
            model_id, self.specialty.value, str(quantization)
        )
        self._pipe = pipeline(
            "image-text-to-text",
            model=model_id,
            torch_dtype=dtype,
            device_map=device_map,
            model_kwargs=model_kwargs,
        )
        logger.info("Model ready.")

    # ── internal history (pipeline format) ───────────────────────────────────

    @property
    def _pipeline_history(self) -> list[dict]:
        """
        Build the message list that the pipeline expects, including the
        system prompt prepended as the first user / system turn.
        History is truncated to max_history_turns if needed.
        """
        system_turn = {
            "role": "system",
            "content": [{"type": "text", "text": self.system_prompt}],
        }
        history = self._raw_pipeline_history

        # Warn and truncate if context is growing large
        if len(history) > self.max_history_turns * 2:
            logger.warning(
                "History has %d messages (limit=%d turns). "
                "Truncating oldest turns to stay within context window.",
                len(history),
                self.max_history_turns,
            )
            # Keep system turn + most recent max_history_turns*2 messages
            history = history[-(self.max_history_turns * 2):]

        return [system_turn] + history

    def __init_pipeline_history(self):
        self._raw_pipeline_history: list[dict] = []

    # ── public API ────────────────────────────────────────────────────────────

    def ask(
        self,
        text: str,
        images: str | Path | Image.Image | list | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        """
        Send a user message with zero or more images; return the reply.

        Args:
            text:           The prompt text.
            images:         Single image or list of images (URL / path / PIL).
                            Pass multiple images to compare scans, timepoints, etc.
            max_new_tokens: Per-call token override.

        Returns:
            Assistant reply as a plain string.
        """
        image_infos = load_images(images)
        content = self._build_content(text, image_infos)
        user_turn = {"role": "user", "content": content}
        self._raw_pipeline_history.append(user_turn)

        tokens = max_new_tokens or self.max_new_tokens
        raw = self._pipe(self._pipeline_history, max_new_tokens=tokens)
        reply = self._parse_reply(raw)

        # Store in raw pipeline history
        self._raw_pipeline_history.append(
            {"role": "assistant", "content": [{"type": "text", "text": reply}]}
        )

        # Store in session log
        self._session.turns.append(Turn(
            role="user",
            text=text,
            image_sources=[i.source for i in image_infos],
            token_count=self._rough_token_count(text),
        ))
        self._session.turns.append(Turn(
            role="assistant",
            text=reply,
            token_count=self._rough_token_count(reply),
        ))

        return reply

    def ask_batch(
        self,
        prompts: list[dict],
        max_new_tokens: int | None = None,
    ) -> list[str]:
        """
        Run multiple independent single-turn inferences in one call.
        Does NOT touch or update conversation history.

        Args:
            prompts: List of dicts, each with:
                     {"text": "...", "images": [...]}  (images key is optional)
            max_new_tokens: Token cap applied to every prompt.

        Returns:
            List of reply strings, one per prompt.

        Example:
            results = client.ask_batch([
                {"text": "Describe this X-ray.", "images": "chest1.png"},
                {"text": "Describe this X-ray.", "images": "chest2.png"},
                {"text": "What is the normal CBC range?"},
            ])
        """
        replies = []
        tokens = max_new_tokens or self.max_new_tokens
        for p in prompts:
            image_infos = load_images(p.get("images"))
            content = self._build_content(p["text"], image_infos)
            sys_turn = {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            }
            messages = [sys_turn, {"role": "user", "content": content}]
            raw = self._pipe(messages, max_new_tokens=tokens)
            replies.append(self._parse_reply(raw))
        return replies

    def stream(
        self,
        text: str,
        images: str | Path | Image.Image | list | None = None,
        max_new_tokens: int | None = None,
    ) -> Iterator[str]:
        """
        Streaming variant of ask() — yields token strings as they are generated.

        Note: Not all pipeline/model backends support streaming; falls back to
        returning the full reply in one chunk if unsupported.

        Example:
            for token in client.stream("Describe this MRI.", images=mri_url):
                print(token, end="", flush=True)
        """
        image_infos = load_images(images)
        content = self._build_content(text, image_infos)
        user_turn = {"role": "user", "content": content}
        self._raw_pipeline_history.append(user_turn)

        tokens = max_new_tokens or self.max_new_tokens
        try:
            streamer = TextStreamer(self._pipe.tokenizer, skip_prompt=True)
            raw = self._pipe(
                self._pipeline_history,
                max_new_tokens=tokens,
                streamer=streamer,
            )
            reply = self._parse_reply(raw)
        except (AttributeError, TypeError):
            # Tokenizer not accessible via pipeline — fall back to non-streaming
            logger.warning("Streaming not supported by this pipeline; returning full reply.")
            raw = self._pipe(self._pipeline_history, max_new_tokens=tokens)
            reply = self._parse_reply(raw)
            yield reply
            return

        self._raw_pipeline_history.append(
            {"role": "assistant", "content": [{"type": "text", "text": reply}]}
        )
        self._session.turns.append(Turn(role="user", text=text,
                                        image_sources=[i.source for i in image_infos]))
        self._session.turns.append(Turn(role="assistant", text=reply))
        yield reply

    def summarise(self, max_new_tokens: int = 300) -> str:
        """
        Ask the model to generate a concise summary of the conversation so far.
        Does NOT append the summary to history.
        """
        if not self._raw_pipeline_history:
            return "No conversation history to summarise."
        summary_request = (
            "Please provide a concise clinical summary of our conversation so far, "
            "including all key findings, differentials discussed, and any recommendations made."
        )
        sys_turn = {
            "role": "system",
            "content": [{"type": "text", "text": self.system_prompt}],
        }
        messages = self._raw_pipeline_history + [
            {"role": "user", "content": [{"type": "text", "text": summary_request}]}
        ]
        raw = self._pipe([sys_turn] + messages, max_new_tokens=max_new_tokens)
        return self._parse_reply(raw)

    def reset(
        self,
        specialty: str | Specialty | None = None,
        custom_prompt: str = "",
    ):
        """
        Clear conversation history.  Optionally switch specialty at the same time.

        Args:
            specialty:     New specialty preset, or None to keep current.
            custom_prompt: Required when switching to specialty='custom'.
        """
        if specialty is not None:
            self.specialty = Specialty(str(specialty))
            self.system_prompt = get_system_prompt(self.specialty, custom_prompt)
        self._raw_pipeline_history = []
        self._session = Session(specialty=self.specialty.value)
        logger.info("Session reset.  Specialty: %s", self.specialty.value)

    # ── session persistence ───────────────────────────────────────────────────

    def save_session(self, path: str | Path) -> None:
        """Save the current session (text only) to a JSON file."""
        p = Path(path)
        p.write_text(self._session.to_json(), encoding="utf-8")
        logger.info("Session saved → %s", p)

    def load_session(self, path: str | Path) -> None:
        """
        Restore a previously saved session from JSON.

        Note: PIL Image objects cannot be serialised; images are stored as
        source references only.  Restored turns are text-only in pipeline
        history — the model will have text context but not the raw pixels.
        """
        p = Path(path)
        self._session = Session.from_json(p.read_text(encoding="utf-8"))
        self.specialty = Specialty(self._session.specialty)
        self.system_prompt = get_system_prompt(self.specialty)

        # Rebuild pipeline history from saved turns (text only)
        self._raw_pipeline_history = []
        for turn in self._session.turns:
            self._raw_pipeline_history.append({
                "role": turn.role,
                "content": [{"type": "text", "text": turn.text}],
            })
        logger.info("Session loaded ← %s  (%d turns)", p, len(self._session.turns))

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def history(self) -> list[Turn]:
        """Read-only list of Turn objects (text + metadata)."""
        return list(self._session.turns)

    @property
    def token_usage(self) -> dict[str, int]:
        """Rough token-usage breakdown across the session."""
        user_tokens  = sum(t.token_count for t in self._session.turns if t.role == "user")
        model_tokens = sum(t.token_count for t in self._session.turns if t.role == "assistant")
        return {"user": user_tokens, "assistant": model_tokens, "total": user_tokens + model_tokens}

    @property
    def turn_count(self) -> int:
        """Number of complete user→assistant exchanges."""
        return len(self._session.turns) // 2

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _build_content(text: str, image_infos: list[ImageInfo]) -> list[dict]:
        """Construct a pipeline-compatible content block list."""
        content = []
        for info in image_infos:
            content.append({"type": "image", "image": info.image})
        content.append({"type": "text", "text": text})
        return content

    @staticmethod
    def _parse_reply(output) -> str:
        """
        Extract plain-text reply from any transformers pipeline output shape.

        Handles:
          1. List of message-dicts  (modern image-text-to-text)
          2. Plain string           (legacy / text-generation pipelines)
          3. Fallback stringify
        """
        result = output[0] if isinstance(output, list) else output
        generated = result.get("generated_text", "")

        if isinstance(generated, list):
            for turn in reversed(generated):
                if turn.get("role") == "assistant":
                    content = turn.get("content", "")
                    if isinstance(content, list):
                        return "\n".join(
                            b["text"] for b in content if b.get("type") == "text"
                        ).strip()
                    return str(content).strip()

        if isinstance(generated, str):
            return generated.strip()

        return str(generated).strip()

    @staticmethod
    def _rough_token_count(text: str) -> int:
        """~4 chars per token heuristic — good enough for budget tracking."""
        return max(1, len(text) // 4)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MedicalAssistant — specialty-aware high-level facade
# ══════════════════════════════════════════════════════════════════════════════

class MedicalAssistant:
    """
    High-level, user-facing facade over MedGemmaClient.

    Provides named helper methods for the most common clinical workflows,
    removing the need to craft prompts manually.

    Examples
    --------
    assistant = MedicalAssistant(specialty="radiology")
    report = assistant.analyze_scan("chest.png", modality="CXR")

    assistant = MedicalAssistant(specialty="dermatology")
    ddx = assistant.differential_diagnosis("erythematous plaque on the arm", image="skin.jpg")

    assistant = MedicalAssistant(specialty="general")
    plan = assistant.management_plan("T2DM, HbA1c 9.2%, on metformin 1g BD")
    """

    def __init__(self, specialty: str = "general", quantization: str = "nf4", **kwargs):
        self.client = MedGemmaClient(specialty=specialty, quantization=quantization, **kwargs)

    # ── imaging ───────────────────────────────────────────────────────────────

    def analyze_scan(
        self,
        images: str | Path | Image.Image | list,
        modality: str = "",
        clinical_context: str = "",
        **kwargs,
    ) -> str:
        """
        Structured analysis of one or more imaging studies.

        Args:
            images:           Single image or list of images.
            modality:         e.g. "CXR", "CT Abdomen", "MRI Brain", "Echo".
            clinical_context: Relevant history / indication.
        """
        mod_str = f"Modality: {modality}." if modality else ""
        ctx_str = f"Clinical context: {clinical_context}." if clinical_context else ""
        prompt = (
            f"{mod_str} {ctx_str}\n"
            "Please provide a structured radiological report:\n"
            "1. Technical adequacy\n"
            "2. Key findings (with measurements where visible)\n"
            "3. Differential diagnosis (ranked)\n"
            "4. Impression and recommendations"
        ).strip()
        return self.client.ask(prompt, images=images, **kwargs)

    def compare_scans(
        self,
        images: list,
        timepoints: list[str] | None = None,
        **kwargs,
    ) -> str:
        """
        Compare two or more imaging studies (e.g. pre/post treatment).

        Args:
            images:     List of images in chronological order.
            timepoints: Optional labels e.g. ["Baseline", "3-month follow-up"].
        """
        if timepoints:
            tp_str = ", ".join(
                f"Image {i+1} = {t}" for i, t in enumerate(timepoints)
            )
            prefix = f"Timepoints: {tp_str}.\n"
        else:
            prefix = f"{len(images)} images provided in chronological order.\n"

        prompt = (
            prefix
            + "Please compare these imaging studies and describe:\n"
            "1. Changes in existing findings (progression / regression)\n"
            "2. New findings\n"
            "3. Overall treatment response / disease course\n"
            "4. Recommendations"
        )
        return self.client.ask(prompt, images=images, **kwargs)

    # ── clinical reasoning ────────────────────────────────────────────────────

    def differential_diagnosis(
        self,
        presentation: str,
        images: str | Path | Image.Image | list | None = None,
        **kwargs,
    ) -> str:
        """
        Generate a ranked differential diagnosis.

        Args:
            presentation: Free-text clinical scenario / findings.
            images:       Optional supporting image(s).
        """
        prompt = (
            f"Clinical presentation: {presentation}\n\n"
            "Generate a ranked differential diagnosis. For each entry provide:\n"
            "• Diagnosis\n"
            "• Supporting features\n"
            "• Against features\n"
            "• Key distinguishing investigation"
        )
        return self.client.ask(prompt, images=images, **kwargs)

    def management_plan(self, clinical_summary: str, **kwargs) -> str:
        """
        Suggest an evidence-based management plan.

        Args:
            clinical_summary: Patient history, diagnosis, current medications, etc.
        """
        prompt = (
            f"Patient summary: {clinical_summary}\n\n"
            "Please suggest an evidence-based management plan covering:\n"
            "1. Immediate / acute management\n"
            "2. Investigations to order\n"
            "3. Long-term management\n"
            "4. Patient education and follow-up"
        )
        return self.client.ask(prompt, **kwargs)

    def interpret_labs(self, lab_text: str, clinical_context: str = "", **kwargs) -> str:
        """
        Interpret laboratory results in clinical context.

        Args:
            lab_text:         Raw lab values (plain text or structured string).
            clinical_context: Patient history for contextualisation.
        """
        ctx = f"Clinical context: {clinical_context}\n\n" if clinical_context else ""
        prompt = (
            f"{ctx}Laboratory results:\n{lab_text}\n\n"
            "Interpret these results, flagging:\n"
            "• Abnormal values and their clinical significance\n"
            "• Possible underlying causes\n"
            "• Recommended follow-up tests\n"
            "• Urgency level (routine / urgent / critical)"
        )
        return self.client.ask(prompt, **kwargs)

    def drug_interaction_check(self, medications: list[str], **kwargs) -> str:
        """
        Check for clinically significant drug interactions.

        Args:
            medications: List of drug names (and doses if available).
        """
        med_str = "\n".join(f"  • {m}" for m in medications)
        prompt = (
            f"Patient medications:\n{med_str}\n\n"
            "Identify any clinically significant drug interactions, "
            "contraindications, or monitoring requirements. "
            "Flag any that require immediate action."
        )
        return self.client.ask(prompt, **kwargs)

    def generate_referral(
        self,
        patient_summary: str,
        referring_to: str,
        urgency: str = "routine",
        **kwargs,
    ) -> str:
        """
        Draft a clinical referral letter.

        Args:
            patient_summary: Relevant history, diagnosis, and question.
            referring_to:    Specialty or named clinician.
            urgency:         "routine" | "urgent" | "emergency"
        """
        prompt = (
            f"Draft a professional clinical referral letter to {referring_to} "
            f"(urgency: {urgency}) for the following patient:\n\n"
            f"{patient_summary}\n\n"
            "Include: reason for referral, relevant history, examination findings, "
            "investigations to date, current management, and specific question."
        )
        return self.client.ask(prompt, **kwargs)

    # ── delegation ────────────────────────────────────────────────────────────

    def ask(self, *args, **kwargs) -> str:
        """Pass-through to client.ask() for free-form queries."""
        return self.client.ask(*args, **kwargs)

    def reset(self, *args, **kwargs):
        """Pass-through to client.reset()."""
        self.client.reset(*args, **kwargs)

    def save_session(self, path: str | Path):
        self.client.save_session(path)

    def load_session(self, path: str | Path):
        self.client.load_session(path)

    @property
    def summary(self) -> str:
        return self.client.summarise()
