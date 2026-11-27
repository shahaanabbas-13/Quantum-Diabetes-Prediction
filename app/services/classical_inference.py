"""Independent inference entry points for exported classical model pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd

from app.services.random_forest_inference import (
    DEFAULT_ARTIFACT_PATH,
    FEATURES,
    INVALID_ZERO_FEATURES,
    _as_feature_frame,
)


ARTIFACTS = {
    "Random Forest": DEFAULT_ARTIFACT_PATH,
    "Logistic Regression": (
        Path(__file__).resolve().parents[2]
        / "models"
        / "classical"
        / "logistic_regression_pipeline.joblib"
    ),
    "RBF-SVM": (
        Path(__file__).resolve().parents[2]
        / "models"
        / "classical"
        / "rbf_svm_pipeline.joblib"
    ),
    "Gradient Boosting": (
        Path(__file__).resolve().parents[2]
        / "models"
        / "classical"
        / "gradient_boosting_pipeline.joblib"
    ),
}


def load_classical_model(model_name: str) -> dict[str, Any]:
    if model_name not in ARTIFACTS:
        raise ValueError(f"Unsupported or not-yet-exported classical model: {model_name}")
    artifact_path = ARTIFACTS[model_name]
    if not artifact_path.is_file():
        raise FileNotFoundError(f"{model_name} artifact not found at {artifact_path}")
    artifact = joblib.load(artifact_path)
    if artifact.get("model_name") != model_name or tuple(artifact.get("features", ())) != FEATURES:
        raise ValueError(f"Artifact metadata does not match {model_name}'s expected schema.")
    return artifact


def predict_classical(
    model_name: str,
    values: Mapping[str, Any] | pd.DataFrame,
    artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Predict one input independently with the named model's saved pipeline."""
    loaded = artifact if artifact is not None else load_classical_model(model_name)
    if loaded.get("model_name") != model_name:
        raise ValueError("The supplied artifact belongs to a different model.")
    frame = _as_feature_frame(values)
    invalid_zero_values = [
        name for name in INVALID_ZERO_FEATURES if float(frame.iloc[0][name]) == 0.0
    ]
    if invalid_zero_values:
        raise ValueError(
            "Enter measured non-zero values for: "
            + ", ".join(invalid_zero_values)
            + ". These zero values were treated as missing during training."
        )

    estimator = loaded["pipeline"]
    predicted_class = int(estimator.predict(frame)[0])
    probability = None
    if hasattr(estimator, "predict_proba"):
        class_labels = list(estimator.classes_)
        if 1 in class_labels:
            probability = float(estimator.predict_proba(frame)[0, class_labels.index(1)])
    return {
        "model": model_name,
        "family": "Classical ML",
        "predicted_class": predicted_class,
        "predicted_label": "Diabetes class (1)" if predicted_class == 1 else "Non-diabetes class (0)",
        "positive_class_probability_estimate": probability,
        "probability_note": "Uncalibrated model probability estimate; not a clinical risk score.",
    }
