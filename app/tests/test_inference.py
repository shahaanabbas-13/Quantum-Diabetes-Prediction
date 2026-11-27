"""Inference contract checks; TEST_INPUT is synthetic test data only."""

from __future__ import annotations

import unittest

from app.services.classical_inference import (
    ARTIFACTS,
    load_classical_model,
    predict_classical,
)
from app.services.quantum_inference import load_qsvm, predict_qsvm
from app.services.random_forest_inference import FEATURES


# Synthetic software fixture only. It is never shown in the application or
# interpreted as a real patient's measurements.
TEST_INPUT = {
    "Pregnancies": 2,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 25,
    "Insulin": 90,
    "BMI": 28.0,
    "DiabetesPedigreeFunction": 0.35,
    "Age": 33,
}
MODEL_NAMES = [*ARTIFACTS, "QSVM"]


def run_model(model_name: str, values: dict[str, float]) -> dict:
    if model_name == "QSVM":
        return predict_qsvm(values)
    return predict_classical(model_name, values)


class InferenceContractTests(unittest.TestCase):
    def test_all_supported_artifacts_load(self) -> None:
        for model_name in ARTIFACTS:
            with self.subTest(model=model_name):
                artifact = load_classical_model(model_name)
                self.assertEqual(artifact["model_name"], model_name)
        self.assertEqual(load_qsvm()["model_name"], "QSVM")

    def test_missing_features_are_rejected(self) -> None:
        incomplete = dict(TEST_INPUT)
        incomplete.pop("Age")
        for model_name in MODEL_NAMES:
            with self.subTest(model=model_name):
                with self.assertRaises(ValueError):
                    run_model(model_name, incomplete)

    def test_training_invalid_zero_measurements_are_rejected(self) -> None:
        for feature in ("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"):
            invalid = dict(TEST_INPUT)
            invalid[feature] = 0
            for model_name in MODEL_NAMES:
                with self.subTest(model=model_name, feature=feature):
                    with self.assertRaises(ValueError):
                        run_model(model_name, invalid)

    def test_feature_order_and_model_output_labels(self) -> None:
        reversed_input = dict(reversed(list(TEST_INPUT.items())))
        self.assertEqual(tuple(TEST_INPUT), FEATURES)

        for model_name in MODEL_NAMES:
            with self.subTest(model=model_name):
                ordered = run_model(model_name, TEST_INPUT)
                reversed_result = run_model(model_name, reversed_input)
                self.assertEqual(ordered["predicted_class"], reversed_result["predicted_class"])
                self.assertEqual(ordered["predicted_label"], reversed_result["predicted_label"])
                expected_label = (
                    "Diabetes class (1)"
                    if ordered["predicted_class"] == 1
                    else "Non-diabetes class (0)"
                )
                self.assertEqual(ordered["predicted_label"], expected_label)

                if model_name == "QSVM":
                    self.assertIsInstance(ordered["quantum_decision_score"], float)
                    self.assertNotIn("positive_class_probability_estimate", ordered)
                else:
                    probability = ordered["positive_class_probability_estimate"]
                    self.assertGreaterEqual(probability, 0.0)
                    self.assertLessEqual(probability, 1.0)


if __name__ == "__main__":
    unittest.main()
