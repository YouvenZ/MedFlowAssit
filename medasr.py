"""
medasr.py — MedASR: Medical Speech-to-Text via HuggingFace.

MedASR is a Conformer-based speech-to-text model pre-trained on ~5 000 h
of de-identified medical dictations across radiology, internal medicine,
and family medicine.

    Model  : google/medasr  (HuggingFace)
    Input  : mono-channel 16 kHz int16 waveform
    Output : plain-text transcription

This module provides:
  • transcribe_audio(audio_bytes)  → str
  • A lazy-loaded singleton so the model is only loaded once.
"""

from __future__ import annotations

import io
import logging
import os
import wave
import struct
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
MEDASR_MODEL_ID: str = os.getenv("MEDASR_MODEL", "google/medasr")
MEDASR_SAMPLE_RATE: int = 16_000  # 16 kHz required
MEDASR_DEVICE: str = os.getenv("MEDASR_DEVICE", "cpu")  # "cpu" | "cuda"

# ── Lazy-loaded pipeline singleton ────────────────────────────────────────────
_pipeline = None
_load_error: Optional[str] = None


def _get_pipeline():
    """Load the HuggingFace ASR pipeline once (lazy singleton)."""
    global _pipeline, _load_error
    if _pipeline is not None:
        return _pipeline
    if _load_error:
        raise RuntimeError(_load_error)

    try:
        from transformers import pipeline as hf_pipeline
        logger.info("Loading MedASR model '%s' on device '%s' …", MEDASR_MODEL_ID, MEDASR_DEVICE)

        _pipeline = hf_pipeline(
            task="automatic-speech-recognition",
            model=MEDASR_MODEL_ID,
            device=MEDASR_DEVICE,
            # Conformer-CTC models typically don't need a decoder config;
            # if the model ships with a processor, HF handles it automatically.
        )
        logger.info("MedASR model loaded successfully.")
        return _pipeline

    except Exception as exc:
        _load_error = (
            f"Failed to load MedASR model '{MEDASR_MODEL_ID}': {exc}. "
            "Set MEDASR_MODEL env var to override, or install the model weights."
        )
        logger.error(_load_error)
        raise RuntimeError(_load_error) from exc


# ── Audio helpers ─────────────────────────────────────────────────────────────

def _wav_bytes_to_numpy(audio_bytes: bytes) -> np.ndarray:
    """
    Convert raw WAV bytes (16 kHz, mono, int16) → float32 numpy array in [-1, 1].
    Handles the standard WebM/WAV formats browsers produce.
    """
    buf = io.BytesIO(audio_bytes)
    try:
        with wave.open(buf, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        # Convert to numpy
        if sampwidth == 2:
            dtype = np.int16
        elif sampwidth == 4:
            dtype = np.int32
        else:
            dtype = np.int16

        audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)

        # Mix to mono if stereo
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)

        # Normalise to [-1, 1]
        if dtype == np.int16:
            audio /= 32768.0
        elif dtype == np.int32:
            audio /= 2147483648.0

        return audio

    except Exception:
        # Fallback: try reading as raw int16 PCM @ 16 kHz
        logger.warning("WAV header parse failed; treating as raw int16 PCM")
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return audio


def _convert_webm_to_wav(audio_bytes: bytes) -> np.ndarray:
    """
    Convert WebM/OGG audio (from browser MediaRecorder) → float32 numpy array.
    Uses pydub if available; falls back to ffmpeg subprocess.
    """
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
        seg = seg.set_channels(1).set_frame_rate(MEDASR_SAMPLE_RATE).set_sample_width(2)
        raw = seg.raw_data
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return audio
    except ImportError:
        pass

    # Fallback: ffmpeg via subprocess
    import subprocess, tempfile
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_in:
        tmp_in.write(audio_bytes)
        tmp_in_path = tmp_in.name
    tmp_out_path = tmp_in_path.replace(".webm", ".wav")

    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", tmp_in_path,
            "-ar", str(MEDASR_SAMPLE_RATE), "-ac", "1", "-sample_fmt", "s16",
            tmp_out_path,
        ], capture_output=True, check=True, timeout=30)
        with open(tmp_out_path, "rb") as f:
            wav_bytes = f.read()
        return _wav_bytes_to_numpy(wav_bytes)
    finally:
        for p in (tmp_in_path, tmp_out_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def _bytes_to_array(audio_bytes: bytes, content_type: str = "") -> np.ndarray:
    """Route to the right decoder based on content type or magic bytes."""
    ct = content_type.lower()

    # WAV
    if ct.startswith("audio/wav") or ct.startswith("audio/x-wav") or audio_bytes[:4] == b"RIFF":
        return _wav_bytes_to_numpy(audio_bytes)

    # WebM / OGG (browser MediaRecorder default)
    if ("webm" in ct or "ogg" in ct or
            audio_bytes[:4] == b"\x1aE\xdf\xa3" or audio_bytes[:4] == b"OggS"):
        return _convert_webm_to_wav(audio_bytes)

    # MP3 / MPEG
    if "mpeg" in ct or "mp3" in ct or audio_bytes[:3] == b"ID3" or audio_bytes[:2] == b"\xff\xfb":
        return _convert_webm_to_wav(audio_bytes)  # pydub/ffmpeg handles mp3 too

    # Default: try WAV first, then webm converter
    try:
        return _wav_bytes_to_numpy(audio_bytes)
    except Exception:
        return _convert_webm_to_wav(audio_bytes)


# ── Public API ────────────────────────────────────────────────────────────────

def transcribe_audio(audio_bytes: bytes, content_type: str = "audio/wav") -> str:
    """
    Transcribe medical audio using MedASR.

    Args:
        audio_bytes:  Raw audio file bytes (WAV 16 kHz preferred; WebM/OGG/MP3 also accepted).
        content_type: MIME type hint (e.g. "audio/wav", "audio/webm;codecs=opus").

    Returns:
        Transcribed text string.
    """
    if not audio_bytes:
        raise ValueError("Empty audio data")

    logger.info("Transcribing %d bytes (type=%s)", len(audio_bytes), content_type)

    # Decode to numpy float32
    audio_array = _bytes_to_array(audio_bytes, content_type)
    logger.info("Audio decoded: %.1f s, %d samples", len(audio_array) / MEDASR_SAMPLE_RATE, len(audio_array))

    # Run ASR pipeline
    pipe = _get_pipeline()
    result = pipe(
        {"raw": audio_array, "sampling_rate": MEDASR_SAMPLE_RATE},
        return_timestamps=False,
    )

    text = result.get("text", "").strip() if isinstance(result, dict) else str(result).strip()
    logger.info("Transcription: %d chars", len(text))
    return text


def is_medasr_available() -> bool:
    """Check whether the MedASR model can be loaded (non-blocking check)."""
    if _pipeline is not None:
        return True
    if _load_error:
        return False
    # Don't actually load — just check if transformers is installed
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False
