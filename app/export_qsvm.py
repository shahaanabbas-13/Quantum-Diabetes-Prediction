"""Recreate the notebook-configured QSVM inference artifact.

This performs one quantum-kernel training run using the saved experimental
feature list and the notebook's stratified 250-sample training subset. It
writes only models/ and never updates notebooks/ or results/.
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pennylane as qml
import sklearn
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from app.services.random_forest_inference import FEATURES


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "dataset" / "diabetes.csv"
RESEARCH_RESULTS_PATH = ROOT / "results" / "qsvm_results.csv"
ARTIFACT_PATH = ROOT / "models" / "quantum" / "qsvm_pipeline.joblib"
INVALID_ZERO_FEATURES = ("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI")
N_QUBITS = 4
TRAINING_SUBSET_SIZE = 250


def make_feature_map():
    def feature_map(x):
        qml.AngleEmbedding(x, wires=range(N_QUBITS), rotation="Y")
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[N_QUBITS - 1, 0])

    return feature_map


def make_kernel_circuit():
    device = qml.device("default.qubit", wires=N_QUBITS)
    feature_map = make_feature_map()

    @qml.qnode(device)
    def kernel_circuit(x1, x2):
        feature_map(x1)
        qml.adjoint(feature_map)(x2)
        return qml.probs(wires=range(N_QUBITS))

    return kernel_circuit


def quantum_kernel(A: np.ndarray, B: np.ndarray, label: str) -> np.ndarray:
    circuit = make_kernel_circuit()
    matrix = np.zeros((len(A), len(B)), dtype=float)
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            matrix[i, j] = float(circuit(a, b)[0])
        if (i + 1) % 10 == 0 or i + 1 == len(A):
            print(f"{label}: completed {i + 1}/{len(A)} rows", flush=True)
    return matrix


def main() -> None:
    if ARTIFACT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {ARTIFACT_PATH}")
    if not DATASET_PATH.is_file() or not RESEARCH_RESULTS_PATH.is_file():
        raise FileNotFoundError("The Pima dataset and existing QSVM result CSV are both required.")

    data = pd.read_csv(DATASET_PATH)
    expected_columns = [*FEATURES, "Outcome"]
    if list(data.columns) != expected_columns or len(data) != 768:
        raise ValueError(
            "Dataset schema/row count differs from the research dataset: "
            f"expected {expected_columns} and 768 rows."
        )
    results = pd.read_csv(RESEARCH_RESULTS_PATH)
    if len(results) != 1 or results.loc[0, "model"] != "QSVM":
        raise ValueError("Expected exactly one QSVM row in the existing result CSV.")
    selected_features = [
        value.strip() for value in str(results.loc[0, "selected_features"]).split(",")
    ]
    if len(selected_features) != N_QUBITS or any(name not in FEATURES for name in selected_features):
        raise ValueError(f"Unexpected recorded QSVM feature selection: {selected_features}")
    if int(results.loc[0, "quantum_training_samples"]) != TRAINING_SUBSET_SIZE:
        raise ValueError("Existing QSVM results do not record the expected 250-sample subset.")

    X = data.loc[:, list(FEATURES)]
    X = X.replace({name: {0: np.nan} for name in INVALID_ZERO_FEATURES})
    y = data["Outcome"].astype(int)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    imputer = KNNImputer(n_neighbors=5, weights="distance")
    imputed_train = imputer.fit_transform(X_train)
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(imputed_train)
    selected_indices = [list(FEATURES).index(name) for name in selected_features]
    selected_train = scaled_train[:, selected_indices]
    angle_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
    angle_train = angle_scaler.fit_transform(selected_train)
    _, X_q, _, y_q = train_test_split(
        angle_train,
        y_train.to_numpy(),
        test_size=TRAINING_SUBSET_SIZE,
        stratify=y_train.to_numpy(),
        random_state=42,
    )
    y_q = np.asarray(y_q, dtype=int)

    start = time.perf_counter()
    K_train = quantum_kernel(X_q, X_q, "QSVM training kernel")
    classifier = SVC(kernel="precomputed", class_weight="balanced")
    classifier.fit(K_train, y_q)
    elapsed = time.perf_counter() - start

    artifact = {
        "model_name": "QSVM",
        "family": "Quantum ML",
        "features": FEATURES,
        "selected_features": tuple(selected_features),
        "selected_indices": tuple(selected_indices),
        "invalid_zero_features": INVALID_ZERO_FEATURES,
        "training_samples": 614,
        "quantum_training_samples": int(len(X_q)),
        "number_of_qubits": N_QUBITS,
        "kernel_training_seconds": elapsed,
        "source_configuration": "notebooks/05_Quantum_Kernel_SVM.ipynb",
        "sklearn_version": sklearn.__version__,
        "pennylane_version": qml.__version__,
        "imputer": imputer,
        "scaler": scaler,
        "angle_scaler": angle_scaler,
        "quantum_training_features": X_q,
        "classifier": classifier,
    }
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, ARTIFACT_PATH)
    print(f"Saved QSVM inference artifact: {ARTIFACT_PATH}")
    print(f"Training subset: {len(X_q)} samples")
    print(f"Kernel training duration: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
