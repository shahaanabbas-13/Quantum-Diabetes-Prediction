"""Validate local image and scanned-PDF OCR using synthetic content only.

Run with the Python environment containing ``app/requirements-documents.txt``:
    python -m app.validate_ocr
"""

from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageDraw, ImageFont

from app.services.local_ocr_bridge import extract_candidates_locally


EXPECTED = {
    "Glucose": 130.0,
    "Age": 51.0,
    "BMI": 31.2,
    "BloodPressure": 80.0,
}


def synthetic_report_image() -> Image.Image:
    image = Image.new("RGB", (1800, 700), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 68)
    except OSError:
        font = ImageFont.load_default(size=68)
    lines = [
        "SYNTHETIC OCR VALIDATION ONLY",
        "Fasting glucose: 130 mg/dL",
        "Age: 51 years",
        "BMI: 31.2 kg/m2",
        "Blood pressure: 120/80 mmHg",
    ]
    for index, line in enumerate(lines):
        draw.text((70, 35 + index * 125), line, font=font, fill="black")
    return image


def validate(file_name: str, content: bytes) -> None:
    candidates = extract_candidates_locally(file_name, content)
    actual: dict[str, float] = {}
    for candidate in candidates:
        actual[candidate.feature] = candidate.value
    if not all(actual.get(feature) == value for feature, value in EXPECTED.items()):
        raise AssertionError(
            f"{file_name} candidate mismatch: expected {EXPECTED}, got {actual}; "
            "review the OCR runtime output for extraction details."
        )
    print(f"PASS {file_name}: {actual}")


def main() -> None:
    image = synthetic_report_image()
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    validate("synthetic_validation.png", image_bytes.getvalue())

    scanned_pdf = BytesIO()
    image.save(scanned_pdf, format="PDF", resolution=150)
    validate("synthetic_scanned_validation.pdf", scanned_pdf.getvalue())
    print("OCR VALIDATION PASSED (synthetic fixtures only; no files written).")


if __name__ == "__main__":
    main()
