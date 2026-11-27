"""Read-only views of the project's existing experimental result files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

st.title("Research analytics")
st.caption("Every table and figure below is read from the existing results/ folder. The app does not regenerate experimental results.")
st.info("The completed evaluation used 768 samples and eight input features. Classical test models used 614 training samples; quantum experiments used a 250-sample subset and the same 154-sample held-out test set. This is not a perfectly controlled architecture-only comparison because training sizes and preprocessing differ.")


def show_table(filename: str, title: str) -> None:
    path = RESULTS / filename
    st.subheader(title)
    if path.is_file():
        st.dataframe(pd.read_csv(path), hide_index=True, width="stretch")
    else:
        st.warning(f"Source file not found: {path}")


def show_image(filename: str, caption: str) -> None:
    path = RESULTS / filename
    if path.is_file():
        st.image(str(path), caption=caption, width="stretch")


heldout_tab, classical_tab, feature_tab, quantum_tab = st.tabs(
    ["Held-out comparison", "Classical CV", "Feature selection", "Quantum analysis"]
)

with heldout_tab:
    show_table("classical_quantum_test_comparison.csv", "Archived classical and quantum test results")
    show_image("classical_quantum_test_comparison.png", "Original held-out comparison figure")
    show_image("all_models_confusion_matrices.png", "Original confusion matrices for all six research models")
    st.caption("VQC appears here only as a historical research result. No VQC live inference is available.")

with classical_tab:
    show_table("classical_cv.csv", "Five-fold stratified classical cross-validation")
    show_image("classical_cv_comparison.png", "Original classical CV comparison")

with feature_tab:
    show_table("feature_selection_cv.csv", "Feature-selection cross-validation")
    show_image("feature_selection_cv_comparison.png", "Original feature-selection comparison")

with quantum_tab:
    q_left, q_right = st.columns(2)
    with q_left:
        show_table("quantum_consolidated_comparison.csv", "Quantum held-out results")
        show_table("quantum_reliability_analysis.csv", "Quantum reliability metrics")
    with q_right:
        show_image("quantum_model_comparison.png", "Original quantum model comparison")
        show_image("quantum_execution_time.png", "Original quantum execution times")
        show_image("quantum_reliability_analysis.png", "Original quantum reliability analysis")
