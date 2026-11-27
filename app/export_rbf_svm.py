"""Export the notebook-configured RBF-SVM pipeline without changing research files."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from app.services.random_forest_inference import FEATURES, InvalidZeroToMissing


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "dataset" / "diabetes.csv"
ARTIFACT_PATH = ROOT / "models" / "classical" / "rbf_svm_pipeline.joblib"


def main() -> None:
    if ARTIFACT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {ARTIFACT_PATH}")
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Expected the research dataset at {DATASET_PATH}")

    data = pd.read_csv(DATASET_PATH)
    expected_columns = [*FEATURES, "Outcome"]
    if list(data.columns) != expected_columns or len(data) != 768:
        raise ValueError(
            "Dataset schema/row count differs from the inspected research dataset. "
            f"Expected {expected_columns} and 768 rows; got {list(data.columns)} and {len(data)}."
        )
    X = data.loc[:, list(FEATURES)]
    y = data["Outcome"].astype(int)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    pipeline = Pipeline(
        steps=[
            ("invalid_zero_to_missing", InvalidZeroToMissing()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    artifact = {
        "model_name": "RBF-SVM",
        "family": "Classical ML",
        "features": FEATURES,
        "training_samples": int(len(X_train)),
        "training_split": "train_test_split(test_size=0.20, random_state=42, stratify=y)",
        "source_configuration": "notebooks/07_Experimental_Analysis.ipynb",
        "pipeline": pipeline,
    }
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, ARTIFACT_PATH)
    print(f"Saved RBF-SVM inference artifact: {ARTIFACT_PATH}")
    print(f"Training samples: {len(X_train)}")


if __name__ == "__main__":
    main()
