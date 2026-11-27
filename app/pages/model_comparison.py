"""Separate current-session live predictions from archived research metrics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[2]
RESEARCH_COMPARISON = ROOT / "results" / "classical_quantum_test_comparison.csv"

st.title("Model comparison")
st.caption("Live model outputs and historical test-set results are shown in separate views.")

live_tab, research_tab = st.tabs(["Live predictions", "Research performance"])

with live_tab:
    live_results = st.session_state.get("last_live_model_predictions")
    if not live_results:
        st.info("Run a prediction first. The app keeps only the latest model outputs in this session; it does not keep the submitted feature values.")
        st.caption("Choose New Prediction from the sidebar to run the supported models.")
    else:
        rows = []
        for result in live_results:
            if result["model"] == "QSVM":
                score_label = "Decision score (uncalibrated)"
                score = result.get("quantum_decision_score")
            else:
                score_label = "Positive-class probability estimate (uncalibrated)"
                score = result.get("positive_class_probability_estimate")
            rows.append(
                {
                    "Model": result["model"],
                    "Family": result["family"],
                    "Predicted class": result["predicted_label"],
                    "Score type": score_label,
                    "Score": score,
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.warning("Outputs are independent. No consensus or ensemble result is calculated, and agreement is not proof of correctness.")
        st.caption("VQC is not included in live predictions because its trained weights are unavailable. Its archived test results remain visible in the research view.")

with research_tab:
    if not RESEARCH_COMPARISON.is_file():
        st.error(f"Archived comparison file is missing: {RESEARCH_COMPARISON}")
    else:
        historical = pd.read_csv(RESEARCH_COMPARISON)
        st.info("Historical results from the existing held-out test set. Classical models used 614 training samples; quantum models used a stratified 250-sample subset. These are research metrics, not confidence estimates for an individual prediction.")
        available_metrics = [
            metric
            for metric in ("accuracy", "recall", "f1_score", "roc_auc", "specificity", "execution_time_seconds")
            if metric in historical.columns and historical[metric].notna().any()
        ]
        if available_metrics:
            selected_metric = st.selectbox("Historical metric", available_metrics, key="research_metric")
            chart_data = historical.set_index("model")[[selected_metric]].dropna()
            st.bar_chart(chart_data, y=selected_metric, width="stretch")
        st.dataframe(historical, hide_index=True, width="stretch")

        confusion = ROOT / "results" / "all_models_confusion_matrices.png"
        if confusion.is_file():
            st.image(str(confusion), caption="Archived confusion matrices from the completed experiment", width="stretch")
