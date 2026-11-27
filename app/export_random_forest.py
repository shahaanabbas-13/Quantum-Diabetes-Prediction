"""One-time export of the notebook-configured Random Forest inference pipeline.

This script fits the deterministic held-out Random Forest once on the same
stratified training split/configuration as Notebook 07. It does not change or
write to notebooks/ or results/. Run only after reviewing and approving it.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.services.random_forest_inference import (
    FEATURES,
    InvalidZeroToMissing,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "dataset" / "diabetes.csv"
ARTIFACT_PATH = ROOT / "models" / "classical" / "random_forest_pipeline.joblib"


def main() -> None:
    if ARTIFACT_PATH.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing model artifact: {ARTIFACT_PATH}"
        )
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Expected the research dataset at {DATASET_PATH}")

    data = pd.read_csv(DATASET_PATH)
    expected_columns = [*FEATURES, "Outcome"]
    if list(data.columns) != expected_columns or len(data) != 768:
        raise ValueError(
            "Dataset schema/row count differs from the inspected research dataset. "
            f"Expected columns {expected_columns} and 768 rows; got "
            f"{list(data.columns)} and {len(data)} rows."
        )

    X = data.loc[:, list(FEATURES)]
    y = data["Outcome"].astype(int)
    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ("invalid_zero_to_missing", InvalidZeroToMissing()),
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    artifact = {
        "model_name": "Random Forest",
        "family": "Classical ML",
        "features": FEATURES,
        "invalid_zero_features": (
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
        ),
        "training_samples": int(len(X_train)),
        "training_split": "train_test_split(test_size=0.20, random_state=42, stratify=y)",
        "source_configuration": "notebooks/07_Experimental_Analysis.ipynb",
        "pipeline": pipeline,
    }

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, ARTIFACT_PATH)
    print(f"Saved Random Forest inference artifact: {ARTIFACT_PATH}")
    print(f"Training samples: {len(X_train)}")


if __name__ == "__main__":
    main()
