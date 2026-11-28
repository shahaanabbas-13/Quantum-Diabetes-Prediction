"""In-memory, local extraction of conservative candidates from report files."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re
from typing import Any

MAX_REPORT_BYTES = 15 * 1024 * 1024
MAX_PDF_PAGES = 10
MAX_IMAGE_PIXELS = 20_000_000


@dataclass(frozen=True)
class Candidate:
    feature: str
    value: float
    unit: str
    source_text: str
    page: int | None
    ocr_confidence: float | None


def _numeric_value(text: str) -> float:
    return float(text.replace(",", "."))


def extract_candidates_from_lines(
    lines: list[tuple[str, int | None, float | None]],
) -> list[Candidate]:
    """Map only explicit, relatively unambiguous labels; never infer missing values."""
    patterns = [
        (
            "Glucose",
            re.compile(
                r"\b(?:fasting(?:\s+(?:plasma|blood))?\s+glucose|glucose\s*,\s*fasting|FPG)\b"
                r"[^\d\n]{0,24}(\d+(?:\.\d+)?)\s*(mg\s*/\s*dL|mmol\s*/\s*L)?",
                re.IGNORECASE,
            ),
            "mg/dL",
        ),
        (
            "BloodPressure",
            re.compile(
                r"\b(?:blood\s+pressure|BP)\b[^\d\n]{0,16}"
                r"(\d{2,3})\s*/\s*(\d{2,3})\s*(mm\s*Hg|mmHg)?",
                re.IGNORECASE,
            ),
            "mmHg (diastolic component)",
        ),
        (
            "BMI",
            re.compile(
                # OCR commonly confuses capital I with lowercase l or digit 1.
                r"\b(?:BM[I1l]|body\s+mass\s+index)\b[^\d\n]{0,16}"
                r"(\d{1,2}(?:\.\d+)?)\s*(kg\s*/\s*m(?:2|²))?",
                re.IGNORECASE,
            ),
            "kg/m²",
        ),
        (
            "Age",
            re.compile(r"\bage\b[^\d\n]{0,12}(\d{1,3})\s*(years?|yrs?)?\b", re.IGNORECASE),
            "years",
        ),
        (
            "Pregnancies",
            re.compile(r"\b(?:pregnancies|gravida)\b[^\d\n]{0,12}(\d{1,2})\b", re.IGNORECASE),
            "count",
        ),
        (
            "SkinThickness",
            re.compile(
                r"\btriceps\s+(?:skin\s*)?(?:fold|thickness)\b[^\d\n]{0,16}"
                r"(\d{1,3}(?:\.\d+)?)\s*(mm)?",
                re.IGNORECASE,
            ),
            "mm",
        ),
        (
            "Insulin",
            re.compile(
                r"(?:\b(?:2\s*[- ]?\s*(?:hour|hr)|2h)\b[^\n]{0,24}\binsulin\b|"
                r"\binsulin\b[^\n]{0,24}\b(?:2\s*[- ]?\s*(?:hour|hr)|2h)\b)"
                r"[^\d\n]{0,12}(\d+(?:\.\d+)?)\s*(?:μ|u|µ)?IU\s*/\s*mL",
                re.IGNORECASE,
            ),
            "μU/mL (2-hour)" ,
        ),
    ]

    candidates: list[Candidate] = []
    for text, page, confidence in lines:
        for feature, pattern, expected_unit in patterns:
            match = pattern.search(text)
            if not match:
                continue
            if feature == "BloodPressure":
                value = _numeric_value(match.group(2))  # The dataset uses diastolic pressure.
            else:
                value = _numeric_value(match.group(1))
            source_unit = match.group(match.lastindex or 0)
            unit = expected_unit
            if feature == "Glucose" and match.group(2):
                source_unit = match.group(2)
                unit = re.sub(r"\s+", "", source_unit).replace("dL", "dL").replace("L", "L")
                if "mmol" in unit.lower():
                    # Keep the candidate visible, but the UI will leave the model field blank.
                    unit = f"{unit} — not mapped to mg/dL"
            elif feature == "Glucose":
                unit = "not stated; expected mg/dL"
            candidates.append(
                Candidate(
                    feature=feature,
                    value=value,
                    unit=unit,
                    source_text=text.strip()[:240],
                    page=page,
                    ocr_confidence=confidence,
                )
            )
    return candidates


def _ocr_image(image: Any, engine: Any) -> list[tuple[str, int | None, float | None]]:
    import numpy as np

    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ValueError("Image is too large for local OCR (over 20 megapixels).")
    result = engine(np.asarray(image.convert("RGB")))
    texts = getattr(result, "txts", None) or ()
    scores = getattr(result, "scores", None) or ()
    return [(str(text), None, float(scores[i]) if i < len(scores) else None) for i, text in enumerate(texts)]


def extract_report(file_name: str, content: bytes) -> tuple[str, list[Candidate]]:
    """Extract report text in memory and return conservative measurement candidates."""
    if not content:
        raise ValueError("The uploaded report is empty.")
    if len(content) > MAX_REPORT_BYTES:
        raise ValueError("Report exceeds the 15 MB upload limit.")

    extension = Path(file_name).suffix.lower()
    lines: list[tuple[str, int | None, float | None]] = []
    if extension == ".pdf":
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(content)
        try:
            if len(pdf) > MAX_PDF_PAGES:
                raise ValueError("PDF exceeds the 10-page processing limit.")
            engine = None
            for page_number, page in enumerate(pdf, start=1):
                page_text = page.get_textpage().get_text_range()
                text_lines = [line.strip() for line in page_text.splitlines() if line.strip()]
                if text_lines:
                    lines.extend((line, page_number, None) for line in text_lines)
                else:
                    if engine is None:
                        from rapidocr import RapidOCR

                        engine = RapidOCR()
                    image = page.render(scale=2.5).to_pil()
                    lines.extend(
                        (text, page_number, confidence)
                        for text, _, confidence in _ocr_image(image, engine)
                    )
        finally:
            pdf.close()
    elif extension in {".png", ".jpg", ".jpeg"}:
        from PIL import Image
        from rapidocr import RapidOCR

        with Image.open(BytesIO(content)) as image:
            lines = _ocr_image(image, RapidOCR())
    else:
        raise ValueError("Use a PDF, PNG, JPG, or JPEG medical report.")

    extracted_text = "\n".join(line for line, _, _ in lines)
    if not extracted_text.strip():
        raise ValueError("No readable text was extracted. Enter the values manually instead.")
    return extracted_text, extract_candidates_from_lines(lines)
