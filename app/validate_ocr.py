"""Validate local image and scanned-PDF OCR using synthetic content only.

Run with the Python environment containing ``app/requirements-documents.txt``:
    python -m app.validate_ocr
"""

from __future__ import annotations

from io import BytesIO
import subprocess

from PIL import Image, ImageDraw, ImageFont

from app.services.local_ocr_bridge import (
    extract_candidates_locally,
    local_ocr_worker_python,
)


EXPECTED = {
    "Glucose": 130.0,
    "Age": 51.0,
    "BMI": 31.2,
    "BloodPressure": 80.0,
}


def synthetic_report_image(
    *,
    size: tuple[int, int] = (1800, 700),
    font_size: int = 68,
    lines: list[str] | None = None,
) -> Image.Image:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default(size=font_size)
    content = lines or [
        "SYNTHETIC OCR VALIDATION ONLY",
        "Fasting glucose: 130 mg/dL",
        "Age: 51 years",
        "BMI: 31.2 kg/m2",
        "Blood pressure: 120/80 mmHg",
    ]
    vertical_step = min(125, max(46, (size[1] - 55) // max(len(content), 1)))
    for index, line in enumerate(content):
        draw.text((35, 15 + index * vertical_step), line, font=font, fill="black")
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


def expect_rejected(file_name: str, content: bytes) -> None:
    try:
        extract_candidates_locally(file_name, content)
    except (RuntimeError, ValueError) as exc:
        print(f"PASS {file_name} rejected safely: {exc}")
        return
    raise AssertionError(f"{file_name} should have been rejected")


def main() -> None:
    image = synthetic_report_image()
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    validate("synthetic_validation.png", image_bytes.getvalue())

    # Lower resolution, smaller type, alternate layout, and slight skew.
    low_resolution = synthetic_report_image(size=(1000, 420), font_size=38)
    low_resolution = low_resolution.resize((650, 273), Image.Resampling.LANCZOS)
    low_bytes = BytesIO()
    low_resolution.save(low_bytes, format="JPEG", quality=92)
    validate("synthetic_low_resolution.jpg", low_bytes.getvalue())

    alternate_layout = synthetic_report_image(
        size=(1500, 500),
        font_size=46,
        lines=[
            "SYNTHETIC ALTERNATE LAYOUT",
            "Age 51 years       BMI 31.2 kg/m2",
            "Fasting glucose 130 mg/dL",
            "Blood pressure 120/80 mmHg",
        ],
    )
    rotated = alternate_layout.rotate(2.5, expand=True, fillcolor="white")
    rotated_bytes = BytesIO()
    rotated.save(rotated_bytes, format="PNG")
    validate("synthetic_rotated_alternate.png", rotated_bytes.getvalue())

    # Different glucose readings must remain separate candidates. The UI leaves
    # conflicting values blank for explicit user selection and confirmation.
    multiple_image = synthetic_report_image(
        size=(1800, 500),
        font_size=54,
        lines=[
            "SYNTHETIC MULTIPLE READINGS",
            "Fasting glucose: 130 mg/dL",
            "Fasting glucose: 142 mg/dL",
        ],
    )
    multiple_bytes = BytesIO()
    multiple_image.save(multiple_bytes, format="PNG")
    multiple_candidates = extract_candidates_locally(
        "synthetic_multiple.png", multiple_bytes.getvalue()
    )
    glucose_values = [
        candidate.value for candidate in multiple_candidates if candidate.feature == "Glucose"
    ]
    if glucose_values != [130.0, 142.0]:
        raise AssertionError(f"Multiple glucose readings were not preserved: {glucose_values}")
    print(f"PASS multiple readings kept distinct: {glucose_values}")

    missing_unit = synthetic_report_image(
        size=(1300, 260),
        font_size=60,
        lines=["SYNTHETIC UNIT REVIEW", "Fasting glucose: 130"],
    )
    missing_bytes = BytesIO()
    missing_unit.save(missing_bytes, format="PNG")
    missing_candidates = extract_candidates_locally("synthetic_missing_unit.png", missing_bytes.getvalue())
    glucose_without_unit = [c for c in missing_candidates if c.feature == "Glucose"]
    if len(glucose_without_unit) != 1 or "not stated" not in glucose_without_unit[0].unit.lower():
        raise AssertionError(f"Missing glucose unit was not flagged: {glucose_without_unit}")
    print("PASS missing glucose unit flagged for manual review")

    scanned_pdf = BytesIO()
    image.save(scanned_pdf, format="PDF", resolution=150)
    validate("synthetic_scanned_validation.pdf", scanned_pdf.getvalue())

    expect_rejected("synthetic_corrupt.pdf", b"not a valid PDF document")

    pdf_generator = subprocess.run(
        [
            str(local_ocr_worker_python()),
            "-c",
            (
                "from io import BytesIO; from pypdf import PdfWriter; import sys; "
                "w=PdfWriter(); w.add_blank_page(width=612,height=792); "
                "w.encrypt(user_password='synthetic-test-password'); "
                "b=BytesIO(); w.write(b); sys.stdout.buffer.write(b.getvalue())"
            ),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    expect_rejected("synthetic_password_protected.pdf", pdf_generator.stdout)

    print("OCR VALIDATION PASSED (synthetic fixtures only; no report files written).")


if __name__ == "__main__":
    main()
