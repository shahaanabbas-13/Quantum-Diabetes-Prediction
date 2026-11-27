"""Recalculate QSVM held-out metrics without fitting or writing artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from app.services.quantum_inference import QSVM_ARTIFACT_PATH, load_qsvm, _predict_quantum_angles
from app.services.random_forest_inference import FEATURES


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "dataset" / "diabetes.csv"
RESULTS_PATH = ROOT / "results" / "qsvm_results.csv"


def main() -> None:
    artifact = load_qsvm(QSVM_ARTIFACT_PATH)
    data = pd.read_csv(DATASET_PATH)
    saved = pd.read_csv(RESULTS_PATH).set_index("model").loc["QSVM"]
    X = data.drop(columns="Outcome").replace(
        {name: {0: np.nan} for name in artifact["invalid_zero_features"]}
    )
    y = data["Outcome"].astype(int)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # Reuse fitted training preprocessors from the artifact; do not fit anything.
    imputed = artifact["imputer"].transform(X_test)
    scaled = artifact["scaler"].transform(imputed)
    selected = scaled[:, list(artifact["selected_indices"])]
    angles = artifact["angle_scaler"].transform(selected)
    predictions, scores = _predict_quantum_angles(angles, artifact)
    precision_curve, recall_curve, _ = precision_recall_curve(
        y_test, scores, pos_label=1
    )
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()

    calculated = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, pos_label=1, zero_division=0),
        "recall": recall_score(y_test, predictions, pos_label=1, zero_division=0),
        "f1_score": f1_score(y_test, predictions, pos_label=1, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "roc_auc_score(y_test, scores)": roc_auc_score(y_test, scores),
        "roc_auc_score(y_test, -scores)": roc_auc_score(y_test, -scores),
        "average_precision_score(y_test, scores, pos_label=1)": average_precision_score(
            y_test, scores, pos_label=1
        ),
        "trapezoidal_pr_auc": auc(recall_curve, precision_curve),
    }
    saved_columns = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1_score": "f1_score",
        "specificity": "specificity",
        "roc_auc_score(y_test, scores)": "roc_auc",
        "average_precision_score(y_test, scores, pos_label=1)": "pr_auc",
    }

    print(f"PennyLane runtime: {artifact['pennylane_version']}")
    print(f"SVC classes_: {artifact['classifier'].classes_.tolist()}")
    print("Notebook 05 calls: roc_auc_score(y_test, scores); average_precision_score(y_test, scores)")
    print("Current reconstruction vs saved results:")
    for metric, value in calculated.items():
        saved_column = saved_columns.get(metric)
        if saved_column:
            print(
                f"  {metric}: current={value:.12f}; "
                f"saved={float(saved[saved_column]):.12f}"
            )
        else:
            print(f"  {metric}: current={value:.12f}")
    print("No model fitting or file writes were performed.")


if __name__ == "__main__":
    main()
