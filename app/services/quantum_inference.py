"""Inference service for the saved fidelity-kernel QSVM artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from app.services.random_forest_inference import _as_feature_frame


QSVM_ARTIFACT_PATH = (
    Path(__file__).resolve().parents[2] / "models" / "quantum" / "qsvm_pipeline.joblib"
)
INVALID_ZERO_FEATURES = ("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI")
N_QUBITS = 4


def load_qsvm(artifact_path: Path | str = QSVM_ARTIFACT_PATH) -> dict[str, Any]:
    path = Path(artifact_path)
    if not path.is_file():
        raise FileNotFoundError(f"QSVM inference artifact not found at {path}")
    artifact = joblib.load(path)
    if artifact.get("model_name") != "QSVM" or len(artifact.get("selected_features", ())) != N_QUBITS:
        raise ValueError(f"QSVM artifact metadata is invalid: {path}")
    return artifact


def _make_kernel_circuit():
    import pennylane as qml

    device = qml.device("default.qubit", wires=N_QUBITS)

    def feature_map(x):
        qml.AngleEmbedding(x, wires=range(N_QUBITS), rotation="Y")
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[N_QUBITS - 1, 0])

    @qml.qnode(device)
    def kernel_circuit(x1, x2):
        feature_map(x1)
        qml.adjoint(feature_map)(x2)
        return qml.probs(wires=range(N_QUBITS))

    return kernel_circuit


def _predict_quantum_angles(
    angles: np.ndarray,
    artifact: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Run already-preprocessed quantum angles through the stored QSVM."""
    training_features = artifact["quantum_training_features"]
    circuit = _make_kernel_circuit()
    kernel_matrix = np.asarray(
        [
            [float(circuit(row, training_point)[0]) for training_point in training_features]
            for row in angles
        ],
        dtype=float,
    )
    classifier = artifact["classifier"]
    return classifier.predict(kernel_matrix), classifier.decision_function(kernel_matrix)


def predict_qsvm(
    values: Mapping[str, Any] | pd.DataFrame,
    artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    loaded = artifact if artifact is not None else load_qsvm()
    frame = _as_feature_frame(values)
    invalid_zeros = [name for name in INVALID_ZERO_FEATURES if float(frame.iloc[0][name]) == 0.0]
    if invalid_zeros:
        raise ValueError(
            "Enter measured non-zero values for: " + ", ".join(invalid_zeros)
            + ". These zero values were treated as missing during training."
        )

    imputed = loaded["imputer"].transform(frame)
    scaled = loaded["scaler"].transform(imputed)
    selected = scaled[:, list(loaded["selected_indices"])]
    angles = loaded["angle_scaler"].transform(selected)
    predictions, decision_scores = _predict_quantum_angles(angles, loaded)
    predicted_class = int(predictions[0])
    decision_score = float(decision_scores[0])
    return {
        "model": "QSVM",
        "family": "Quantum ML",
        "predicted_class": predicted_class,
        "predicted_label": "Diabetes class (1)" if predicted_class == 1 else "Non-diabetes class (0)",
        "quantum_decision_score": decision_score,
        "score_note": "Uncalibrated QSVM decision score; not a probability or medical risk estimate.",
    }
