"""Read-only comparison of exported QSVM inference state with Notebook 05.

Run explicitly from the repository root. This recreates the notebook's
reference fit and held-out predictions in memory, compares predictions and
metrics, and never saves a model or writes to results/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pennylane as qml
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from app.export_qsvm import ARTIFACT_PATH, DATASET_PATH, RESEARCH_RESULTS_PATH, N_QUBITS
from app.services.quantum_inference import load_qsvm, _predict_quantum_angles
from app.services.random_forest_inference import FEATURES


INVALID_ZERO_FEATURES = ("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI")


def reference_kernel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Notebook 05's fidelity-kernel calculation, restated for verification."""
    device = qml.device("default.qubit", wires=N_QUBITS)

    def feature_map(x):
        qml.AngleEmbedding(x, wires=range(N_QUBITS), rotation="Y")
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[N_QUBITS - 1, 0])

    @qml.qnode(device)
    def circuit(x1, x2):
        feature_map(x1)
        qml.adjoint(feature_map)(x2)
        return qml.probs(wires=range(N_QUBITS))

    matrix = np.zeros((len(A), len(B)), dtype=float)
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            matrix[i, j] = float(circuit(a, b)[0])
        if (i + 1) % 10 == 0 or i + 1 == len(A):
            print(f"Reference test kernel: completed {i + 1}/{len(A)} rows", flush=True)
    return matrix


def main() -> None:
    artifact = load_qsvm(ARTIFACT_PATH)
    data = pd.read_csv(DATASET_PATH)
    results = pd.read_csv(RESEARCH_RESULTS_PATH)
    expected_columns = [*FEATURES, "Outcome"]
    if list(data.columns) != expected_columns or len(data) != 768:
        raise ValueError("Dataset no longer matches the inspected 768-row research dataset.")

    selected = list(artifact["selected_features"])
    X = data.drop(columns="Outcome").replace(
        {name: {0: np.nan} for name in INVALID_ZERO_FEATURES}
    )
    y = data["Outcome"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # Independently reconstruct Notebook 05 preprocessing from the train split.
    reference_imputer = KNNImputer(n_neighbors=5, weights="distance")
    train_imputed = reference_imputer.fit_transform(X_train)
    test_imputed = reference_imputer.transform(X_test)
    reference_scaler = StandardScaler()
    train_scaled = reference_scaler.fit_transform(train_imputed)
    test_scaled = reference_scaler.transform(test_imputed)
    feature_indices = [list(FEATURES).index(name) for name in selected]
    reference_minmax = MinMaxScaler(feature_range=(-np.pi, np.pi))
    train_angles = reference_minmax.fit_transform(train_scaled[:, feature_indices])
    test_angles = reference_minmax.transform(test_scaled[:, feature_indices])
    _, reference_train_features, _, reference_train_labels = train_test_split(
        train_angles,
        y_train.to_numpy(),
        test_size=250,
        stratify=y_train.to_numpy(),
        random_state=42,
    )
    reference_train_labels = np.asarray(reference_train_labels, dtype=int)

    if not np.array_equal(
        artifact["quantum_training_features"], reference_train_features
    ):
        raise AssertionError("Exported QSVM training subset differs from Notebook 05 reconstruction.")
    print("Rebuilding reference training kernel (250 x 250)...", flush=True)
    K_train = reference_kernel(reference_train_features, reference_train_features)
    reference_classifier = SVC(kernel="precomputed", class_weight="balanced")
    reference_classifier.fit(K_train, reference_train_labels)

    if not np.array_equal(reference_classifier.support_, artifact["classifier"].support_):
        raise AssertionError("Exported QSVM support-vector indices differ from Notebook 05 reconstruction.")
    if not np.allclose(
        reference_classifier.dual_coef_, artifact["classifier"].dual_coef_, rtol=0.0, atol=1e-12
    ):
        raise AssertionError("Exported QSVM dual coefficients differ from Notebook 05 reconstruction.")

    print("Running held-out data through the dashboard QSVM inference path...", flush=True)
    app_predictions, app_scores = _predict_quantum_angles(test_angles, artifact)
    # The same held-out kernel rows are evaluated against both independently fitted
    # estimators so this compares the exported classifier state with the reference fit.
    K_test = reference_kernel(test_angles, reference_train_features)
    reference_predictions = reference_classifier.predict(K_test)
    reference_scores = reference_classifier.decision_function(K_test)

    if not np.array_equal(reference_predictions, app_predictions):
        mismatches = np.flatnonzero(reference_predictions != app_predictions)
        raise AssertionError(
            f"QSVM predictions differ for {len(mismatches)} held-out rows; "
            f"first mismatch indices: {mismatches[:10].tolist()}"
        )
    if not np.allclose(reference_scores, app_scores, rtol=0.0, atol=1e-10):
        raise AssertionError("Dashboard QSVM decision scores differ from the Notebook 05 reference.")

    tn, fp, fn, tp = confusion_matrix(y_test, app_predictions, labels=[0, 1]).ravel()
    actual_metrics = {
        "accuracy": accuracy_score(y_test, app_predictions),
        "precision": precision_score(y_test, app_predictions, zero_division=0),
        "recall": recall_score(y_test, app_predictions, zero_division=0),
        "f1_score": f1_score(y_test, app_predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, app_scores),
        "pr_auc": average_precision_score(y_test, app_scores),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
    }
    saved_row = results.loc[results["model"] == "QSVM"]
    if len(saved_row) != 1:
        raise ValueError("Expected exactly one QSVM row in the read-only results CSV.")
    ranking_metric_differences = {}
    for metric, actual_value in actual_metrics.items():
        saved_value = float(saved_row.iloc[0][metric])
        print(f"{metric}: reconstructed={actual_value:.12f}, saved={saved_value:.12f}")
        if not np.isclose(actual_value, saved_value, rtol=0.0, atol=1e-9):
            if metric in ("roc_auc", "pr_auc"):
                ranking_metric_differences[metric] = actual_value - saved_value
            else:
                raise AssertionError(f"Reconstructed {metric} differs from saved QSVM result.")

    print("PASS: all 154 dashboard QSVM predictions match the Notebook 05 reference.")
    print("PASS: dashboard decision scores match the Notebook 05 reference reconstruction.")
    print("PASS: accuracy, precision, recall, F1, and specificity match the existing QSVM CSV.")
    if ranking_metric_differences:
        print("NOTE: rank-based metric differences are reported without altering research results:")
        for metric, difference in ranking_metric_differences.items():
            print(f"  {metric} delta (reconstructed - saved): {difference:+.12f}")
    else:
        print("PASS: ROC-AUC and PR-AUC also match the existing QSVM CSV.")
    print("No model artifacts or research result files were written.")


if __name__ == "__main__":
    main()
