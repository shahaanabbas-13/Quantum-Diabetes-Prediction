"""Synthetic parser fixtures; no patient reports or values are used."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from app.services.local_ocr_bridge import extract_candidates_locally
from app.services.report_extraction import extract_candidates_from_lines
from app.services.report_session import (
    confirmed_candidate_defaults,
    clear_sensitive_report_state,
    unconfirmed_or_missing_features,
)


class ReportCandidateParsingTests(unittest.TestCase):
    def test_fasting_glucose_maps_but_random_glucose_does_not(self) -> None:
        rows = extract_candidates_from_lines(
            [
                ("Fasting glucose: 120 mg/dL", 1, 0.99),
                ("Random glucose: 190 mg/dL", 1, 0.98),
            ]
        )
        glucose = [row for row in rows if row.feature == "Glucose"]
        self.assertEqual(len(glucose), 1)
        self.assertEqual(glucose[0].value, 120)
        self.assertEqual(glucose[0].unit, "mg/dL")

    def test_unconverted_mmol_glucose_is_retained_with_warning_unit(self) -> None:
        rows = extract_candidates_from_lines([("FPG: 6.8 mmol/L", 2, 0.91)])
        self.assertEqual(len(rows), 1)
        self.assertIn("not mapped", rows[0].unit.lower())

    def test_glucose_without_unit_is_flagged_for_manual_confirmation(self) -> None:
        rows = extract_candidates_from_lines([("Fasting glucose: 120", 2, 0.91)])
        self.assertEqual(len(rows), 1)
        self.assertIn("not stated", rows[0].unit.lower())

    def test_multiple_glucose_measurements_are_kept_separately(self) -> None:
        rows = extract_candidates_from_lines(
            [
                ("Fasting glucose: 120 mg/dL", 1, 0.99),
                ("Fasting glucose: 135 mg/dL", 2, 0.98),
            ]
        )
        self.assertEqual([row.value for row in rows], [120, 135])
        self.assertEqual([row.page for row in rows], [1, 2])

    def test_blood_pressure_maps_only_diastolic_component(self) -> None:
        rows = extract_candidates_from_lines([("Blood pressure 120/80 mmHg", 3, 0.95)])
        self.assertEqual([(row.feature, row.value) for row in rows], [("BloodPressure", 80)])

    def test_never_fabricates_pedigree_function(self) -> None:
        rows = extract_candidates_from_lines(
            [("Age: 35 years; BMI: 28.4 kg/m2", 1, 0.9)]
        )
        self.assertEqual({row.feature for row in rows}, {"Age", "BMI"})


class ReportFailureAndLifecycleTests(unittest.TestCase):
    def test_candidate_defaults_require_unique_values_and_compatible_units(self) -> None:
        candidates = [
            {"feature": "Glucose", "value": 130, "unit": "mg/dL"},
            {"feature": "Age", "value": 51, "unit": "years"},
            {"feature": "BMI", "value": 31.2, "unit": "kg/m²"},
        ]
        self.assertEqual(
            confirmed_candidate_defaults(candidates, ("Glucose", "Age", "BMI", "Insulin")),
            {"Glucose": 130, "Age": 51, "BMI": 31.2, "Insulin": None},
        )
        candidates.extend(
            [
                {"feature": "Glucose", "value": 135, "unit": "mg/dL"},
                {"feature": "Insulin", "value": 88, "unit": "not stated"},
            ]
        )
        self.assertEqual(
            confirmed_candidate_defaults(candidates, ("Glucose", "Insulin")),
            {"Glucose": None, "Insulin": None},
        )

    def test_every_feature_requires_value_and_explicit_confirmation(self) -> None:
        features = ("Age", "Glucose", "BMI")
        values = {"Age": 51, "Glucose": 130, "BMI": 31.2}
        confirmations = {"Age": True, "Glucose": False, "BMI": True}
        self.assertEqual(
            unconfirmed_or_missing_features(values, confirmations, features),
            ([], ["Glucose"]),
        )
        values["BMI"] = None
        self.assertEqual(
            unconfirmed_or_missing_features(values, confirmations, features),
            (["BMI"], ["Glucose"]),
        )

    def test_unavailable_worker_is_reported_without_starting_process(self) -> None:
        with patch.dict("os.environ", {"QD_OCR_PYTHON": r"Z:\missing\python.exe"}):
            with self.assertRaisesRegex(RuntimeError, "OCR runtime not found"):
                extract_candidates_locally("report.png", b"synthetic")

    def test_subprocess_timeout_is_reported_and_stopped(self) -> None:
        import subprocess

        with patch(
            "app.services.local_ocr_bridge.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="ocr", timeout=90),
        ) as run:
            with self.assertRaisesRegex(RuntimeError, "exceeded 90 seconds"):
                extract_candidates_locally("report.png", b"synthetic")
        self.assertEqual(run.call_args.kwargs["timeout"], 90)
        self.assertTrue(run.call_args.kwargs["close_fds"])

    def test_unsupported_and_malformed_uploads_fail_cleanly(self) -> None:
        cases = [
            ("report.docx", b"synthetic"),
            ("report.pdf", b"not a PDF"),
            ("report.png", b"not an image"),
        ]
        for name, content in cases:
            with self.subTest(name=name):
                with self.assertRaises(RuntimeError):
                    extract_candidates_locally(name, content)

    def test_oversized_upload_is_rejected_before_parsing(self) -> None:
        from app.services.report_extraction import MAX_REPORT_BYTES

        with self.assertRaisesRegex(RuntimeError, "15 MB upload limit"):
            extract_candidates_locally("oversized.pdf", b"x" * (MAX_REPORT_BYTES + 1))

    def test_clear_removes_report_values_and_predictions_only(self) -> None:
        state = {
            "medical_report_upload": "synthetic report bytes",
            "report_candidates": [{"feature": "Age", "value": 51}],
            "report_feature_Age": 51,
            "confirm_Age": True,
            "report_models_to_run": ["QSVM"],
            "last_live_model_predictions": [{"predicted_class": 1}],
            "unrelated_dashboard_preference": "dark",
        }
        clear_sensitive_report_state(state)
        self.assertEqual(state, {"unrelated_dashboard_preference": "dark"})


if __name__ == "__main__":
    unittest.main()
