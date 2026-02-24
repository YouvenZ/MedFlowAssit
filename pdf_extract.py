"""
pdf_extract.py — Extract text from uploaded PDF files.

Uses PyPDF2 (or pypdf) as primary parser. Falls back to pdfplumber
if PyPDF2 yields empty text (scanned PDFs, etc.).
"""

from __future__ import annotations

import io
import logging

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract text content from a PDF file's raw bytes.

    Tries PyPDF2/pypdf first, then pdfplumber as fallback.
    Returns the concatenated text of all pages.
    """
    if not pdf_bytes:
        raise ValueError("Empty PDF data")

    text = ""

    # ── Attempt 1: pypdf (PyPDF2 successor) ──────────────────────────────────
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            pages.append(page_text)
        text = "\n\n".join(pages).strip()
        if text:
            logger.info("PDF extracted via pypdf: %d chars, %d pages", len(text), len(pages))
            return text
    except ImportError:
        logger.debug("pypdf not installed, trying PyPDF2")
    except Exception as exc:
        logger.warning("pypdf extraction failed: %s", exc)

    # ── Attempt 2: PyPDF2 (legacy) ───────────────────────────────────────────
    try:
        from PyPDF2 import PdfReader as PdfReader2
        reader = PdfReader2(io.BytesIO(pdf_bytes))
        pages = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            pages.append(page_text)
        text = "\n\n".join(pages).strip()
        if text:
            logger.info("PDF extracted via PyPDF2: %d chars, %d pages", len(text), len(pages))
            return text
    except ImportError:
        logger.debug("PyPDF2 not installed, trying pdfplumber")
    except Exception as exc:
        logger.warning("PyPDF2 extraction failed: %s", exc)

    # ── Attempt 3: pdfplumber (handles more complex layouts) ─────────────────
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages = []
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                pages.append(page_text)
            text = "\n\n".join(pages).strip()
            if text:
                logger.info("PDF extracted via pdfplumber: %d chars, %d pages", len(text), len(pages))
                return text
    except ImportError:
        logger.debug("pdfplumber not installed")
    except Exception as exc:
        logger.warning("pdfplumber extraction failed: %s", exc)

    if not text:
        raise RuntimeError(
            "Could not extract text from PDF. "
            "Install at least one of: pypdf, PyPDF2, pdfplumber. "
            "For scanned PDFs, OCR support is needed."
        )
    return text
