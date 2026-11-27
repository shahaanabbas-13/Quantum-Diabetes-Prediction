"""Load and run the verified Random Forest inference pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


FEATURES = (
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
)
INVALID_ZERO_FEATURES = (
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
)
DEFAULT_ARTIFACT_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "classical"
    / "random_forest_pipeline.joblib"
)


class InvalidZeroToMissing(BaseEstimator, TransformerMixin):
    """Apply the research preprocessing rule for invalid zero measurements."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> "InvalidZeroToMissing":
        self.feature_names_in_ = np.asarray(FEATURES, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = _as_feature_frame(X, single_row=False)
        return frame.replace({name: {0: np.nan} for name in INVALID_ZERO_FEATURES})


def _as_feature_frame(
    values: Mapping[str, Any] | pd.DataFrame,
    *,
    single_row: bool = True,
) -> pd.DataFrame:
    if isinstance(values, pd.DataFrame):
        if set(values.columns) != set(FEATURES):
            raise ValueError("Input must contain all eight named features and no extras.")
        frame = values.loc[:, list(FEATURES)].copy()
    else:
        if not single_row:
            raise TypeError("The preprocessing transformer expects a pandas DataFrame.")
        if set(values) != set(FEATURES):
            missing = [name for name in FEATURES if name not in values]
            extra = [name for name in values if name not in FEATURES]
            raise ValueError(f"Feature mismatch. Missing: {missing}; unexpected: {extra}.")
        frame = pd.DataFrame([[values[name] for name in FEATURES]], columns=FEATURES)

    if single_row and len(frame) != 1:
        raise ValueError("Prediction input must contain exactly one row.")

    for name in FEATURES:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    array = frame.to_numpy(dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("All eight input values must be finite numbers.")
    return frame


def load_random_forest(artifact_path: Path | str = DEFAULT_ARTIFACT_PATH) -> dict[str, Any]:
    """Load the exported artifact; never fit or create model files here."""
    path = Path(artifact_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Random Forest artifact not found at {path}. "
            "Review and run app/export_random_forest.py after approval."
        )
    artifact = joblib.load(path)
    if artifact.get("model_name") != "Random Forest" or tuple(artifact.get("features", ())) != FEATURES:
        raise ValueError(f"Artifact metadata does not match the expected Random Forest schema: {path}")
    return artifact


def predict_random_forest(
    values: Mapping[str, Any] | pd.DataFrame,
    artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the model class and its positive-class probability estimate."""
    loaded = artifact if artifact is not None else load_random_forest()
    frame = _as_feature_frame(values)
    invalid_zero_values = [
        name for name in INVALID_ZERO_FEATURES if float(frame.iloc[0][name]) == 0.0
    ]
    if invalid_zero_values:
        raise ValueError(
            "Enter a measured non-zero value for: "
            + ", ".join(invalid_zero_values)
            + ". These zero values were treated as missing in training and cannot be used as patient measurements."
        )
    estimator = loaded["pipeline"]
    predicted_class = int(estimator.predict(frame)[0])
    probability = None
    if hasattr(estimator, "predict_proba"):
        class_labels = list(estimator.classes_)
        if 1 in class_labels:
            probability = float(estimator.predict_proba(frame)[0, class_labels.index(1)])
    return {
        "model": "Random Forest",
        "family": "Classical ML",
        "predicted_class": predicted_class,
        "predicted_label": "Diabetes class (1)" if predicted_class == 1 else "Non-diabetes class (0)",
        "positive_class_probability_estimate": probability,
        "probability_note": "Model probability estimate; not calibrated or a medical risk estimate.",
    }
