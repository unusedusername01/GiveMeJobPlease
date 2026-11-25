"""Local CV text extraction for PDF, TXT, and Markdown uploads."""

from __future__ import annotations

import io
import re

from pypdf import PdfReader

from give_me_job_please.errors import ParseError

MAX_CV_STORED_CHARS = 20_000
MAX_CV_PAGES = 20


def parse_cv_bytes(raw_bytes: bytes, filename: str) -> str:
    """Extract clean plain text from an uploaded PDF, TXT, or MD resume."""
    lower_name = filename.lower()
    if lower_name.endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(raw_bytes))
            if len(reader.pages) > MAX_CV_PAGES:
                raise ParseError(
                    f"This PDF has {len(reader.pages)} pages; the maximum supported is "
                    f"{MAX_CV_PAGES}. Please upload a shorter CV.",
                    detail=f"CV page count {len(reader.pages)} exceeds limit {MAX_CV_PAGES}",
                )
            pages = [page.extract_text() or "" for page in reader.pages]
            raw_text = "\n\n".join(pages)
        except ParseError:
            raise
        except Exception as exc:
            raise ParseError(
                "Could not read this PDF file. Please ensure it is not password-protected "
                "or scanned as an image without selectable text.",
                detail=f"pypdf failed on {filename}: {exc}",
            ) from exc
    elif lower_name.endswith((".txt", ".md")):
        try:
            raw_text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raw_text = raw_bytes.decode("windows-1252", errors="replace")
    else:
        raise ParseError(
            "Unsupported CV format. Please upload a .pdf, .txt, or .md file.",
            detail=f"Unsupported CV extension for {filename}",
        )

    cleaned = re.sub(r"[ \t]+", " ", raw_text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if len(cleaned) < 20:
        raise ParseError(
            "Extracted CV text looks empty (under 20 characters). "
            "If your PDF is a scanned image, please upload a text-based PDF or .txt file.",
            detail=f"CV extracted length={len(cleaned)}",
        )
    return cleaned[:MAX_CV_STORED_CHARS]
