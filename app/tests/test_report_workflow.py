"""Streamlit end-to-end report workflow using synthetic report content only."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest

from app.services.local_ocr_bridge import local_ocr_worker_python
from app.services.random_forest_inference import FEATURES
from app.validate_ocr import synthetic_report_image


def _worker_available() -> bool:
    try:
        return local_ocr_worker_python().is_file()
    except RuntimeError:
        return False


@unittest.skipUnless(_worker_available(), "local OCR worker is not installed")
class ReportUploadWorkflowTests(unittest.TestCase):
    def test_upload_review_edit_confirm_and_predict_with_all_five_models(self) -> None:
        image_bytes = BytesIO()
        synthetic_report_image().save(image_bytes, format="PNG")
        page_path = Path(__file__).resolve().parents[1] / "pages" / "report_upload.py"
        app = AppTest.from_file(str(page_path), default_timeout=90).run()

        app.file_uploader[0].upload(
            "synthetic_e2e.png", image_bytes.getvalue(), "image/png"
        ).run(timeout=90)
        app.button[1].click().run(timeout=90)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state["report_candidates"]), 4)

        current_values = {widget.key: widget.value for widget in app.number_input}
        self.assertEqual(current_values["report_feature_Glucose"], 130)
        self.assertEqual(current_values["report_feature_BMI"], 31.2)
        self.assertEqual(current_values["report_feature_Age"], 51)

        # Missing values remain blank; unchecked fields block prediction.
        app.button[-1].click().run(timeout=90)
        self.assertTrue(any("Values not confirmed" in item.value for item in app.error))
        self.assertNotIn("last_live_model_predictions", app.session_state)

        manual_values = {
            "Pregnancies": 2,
            "SkinThickness": 25,
            "Insulin": 90,
            "DiabetesPedigreeFunction": 0.35,
        }
        for feature in FEATURES:
            value = manual_values.get(feature, current_values[f"report_feature_{feature}"])
            app.number_input(key=f"report_feature_{feature}").set_value(value)
            app.checkbox(key=f"confirm_{feature}").set_value(True)

        # Simulate an edit during review before committing the confirmed form.
        app.number_input(key="report_feature_Age").set_value(52)
        app.multiselect(key="report_models_to_run").set_value(
            ["Random Forest", "Logistic Regression", "RBF-SVM", "Gradient Boosting", "QSVM"]
        )
        app.button[-1].click().run(timeout=90)

        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        predictions = app.session_state["last_live_model_predictions"]
        self.assertEqual(
            {item["model"] for item in predictions},
            {"Random Forest", "Logistic Regression", "RBF-SVM", "Gradient Boosting", "QSVM"},
        )
        self.assertTrue(all("predicted_label" in item for item in predictions))

        app.button[0].click().run(timeout=90)
        remaining_keys = list(app.session_state)
        self.assertNotIn("report_candidates", remaining_keys)
        self.assertNotIn("last_live_model_predictions", remaining_keys)
        for key in remaining_keys:
            if key.startswith("report_feature_"):
                self.assertIsNone(app.session_state[key])
            if key.startswith("confirm_"):
                self.assertFalse(app.session_state[key])


if __name__ == "__main__":
    unittest.main()
