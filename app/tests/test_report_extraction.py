"""Synthetic parser fixtures; no patient reports or values are used."""

from __future__ import annotations

import unittest

from app.services.report_extraction import extract_candidates_from_lines


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


if __name__ == "__main__":
    unittest.main()
