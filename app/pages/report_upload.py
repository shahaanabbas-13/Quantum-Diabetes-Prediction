"""Local-only medical-report extraction with explicit value confirmation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict

import pandas as pd
import streamlit as st

from app.services.classical_inference import ARTIFACTS, predict_classical
from app.services.random_forest_inference import FEATURES


st.title("Medical report upload")
st.caption("Reports are processed in memory on this computer. They are not written to disk or sent to an OCR service.")
st.warning("Extracted values are candidates only. Confirm every feature, its unit, and its source before prediction. Random glucose is not substituted for fasting glucose, missing features are never guessed, and QSVM/VQC outputs are not diagnostic.")


def clear_report_state() -> None:
    st.session_state.pop("medical_report_upload", None)
    st.session_state.pop("report_candidates", None)
    st.session_state.pop("last_live_model_predictions", None)


st.button("Clear report and extracted values", on_click=clear_report_state)
uploaded = st.file_uploader(
    "Upload a medical report",
    type=["pdf", "png", "jpg", "jpeg"],
    max_upload_size=15,
    key="medical_report_upload",
    help="Maximum 15 MB. Extraction runs locally; the app does not retain the uploaded file on disk.",
)

if uploaded is not None and st.button("Extract candidate values", type="primary"):
    try:
        from app.services.local_ocr_bridge import extract_candidates_locally

        # Bytes pass through a local process pipe; only candidate values/context
        # remain in session state, while extracted full text is discarded.
        candidates = extract_candidates_locally(uploaded.name, uploaded.getvalue())
        st.session_state["report_candidates"] = [asdict(candidate) for candidate in candidates]
        st.success(f"Extraction complete. Found {len(candidates)} candidate measurement(s).")
    except (ValueError, RuntimeError) as exc:
        st.error(str(exc))

candidate_rows = st.session_state.get("report_candidates", [])
candidate_map: dict[str, list[dict]] = defaultdict(list)
for candidate in candidate_rows:
    candidate_map[candidate["feature"]].append(candidate)

if candidate_rows:
    st.subheader("Extracted candidates for review")
    display_rows = [
        {
            "Feature": row["feature"],
            "Candidate value": row["value"],
            "Unit/context": row["unit"],
            "Source excerpt": row["source_text"],
            "Page": row["page"],
            "OCR score": row["ocr_confidence"],
        }
        for row in candidate_rows
    ]
    st.dataframe(pd.DataFrame(display_rows), hide_index=True, width="stretch")
else:
    st.info("Upload a report to extract candidates, or enter all eight measurements manually below.")

st.subheader("Confirm the eight model inputs")
st.caption("Fields with one consistent candidate are prefilled for review. Conflicting or missing values remain blank. Check each confirmation box after reviewing the source and units.")

with st.form("report_confirmed_prediction", clear_on_submit=False):
    col_a, col_b = st.columns(2, gap="large")
    values: dict[str, float | None] = {}
    confirmations: dict[str, bool] = {}
    field_spec = [
        ("Pregnancies", "count", 1.0),
        ("Glucose", "mg/dL", 1.0),
        ("BloodPressure", "mmHg (diastolic)", 1.0),
        ("SkinThickness", "mm", 1.0),
        ("Insulin", "μU/mL (2-hour)", 1.0),
        ("BMI", "kg/m²", 0.1),
        ("DiabetesPedigreeFunction", "score", 0.01),
        ("Age", "years", 1.0),
    ]
    for index, (feature, unit, step) in enumerate(field_spec):
        matches = candidate_map.get(feature, [])
        distinct_values = {float(item["value"]) for item in matches}
        unit_compatible = all(
            "not mapped" not in item["unit"].lower()
            and "not stated" not in item["unit"].lower()
            for item in matches
        )
        candidate_value = next(iter(distinct_values)) if len(distinct_values) == 1 and unit_compatible else None
        target_col = col_a if index % 2 == 0 else col_b
        with target_col:
            values[feature] = st.number_input(
                f"{feature} ({unit})",
                min_value=0.0,
                value=candidate_value,
                step=step,
                placeholder="Enter measured value",
                key=f"report_feature_{feature}",
            )
            confirmations[feature] = st.checkbox(
                f"I confirm {feature}",
                value=False,
                key=f"confirm_{feature}",
            )
            if matches:
                units = sorted({item["unit"] for item in matches})
                st.caption("Extracted unit/context: " + "; ".join(units))
                if not unit_compatible:
                    st.caption("This candidate was not placed in the model field because its unit is missing or does not match. Enter a measured value in the required unit after review.")

    selected_models = st.multiselect(
        "Models to run",
        options=[*ARTIFACTS, "QSVM"],
        default=[*ARTIFACTS, "QSVM"],
        key="report_models_to_run",
    )
    submitted = st.form_submit_button("Run confirmed predictions", type="primary", width="stretch")

if submitted:
    missing = [feature for feature in FEATURES if values[feature] is None]
    unconfirmed = [feature for feature in FEATURES if not confirmations[feature]]
    if missing or unconfirmed:
        details = []
        if missing:
            details.append("Missing measurements: " + ", ".join(missing))
        if unconfirmed:
            details.append("Values not confirmed: " + ", ".join(unconfirmed))
        st.error("Please complete the review. " + ". ".join(details))
    elif not selected_models:
        st.error("Select at least one model.")
    else:
        try:
            results = []
            for model_name in selected_models:
                if model_name == "QSVM":
                    from app.services.quantum_inference import predict_qsvm

                    results.append(predict_qsvm(values))
                else:
                    results.append(predict_classical(model_name, values))
            st.session_state["last_live_model_predictions"] = results
            st.subheader("Individual model outputs")
            for start in range(0, len(results), 2):
                result_columns = st.columns(min(2, len(results) - start))
                for column, result in zip(result_columns, results[start : start + 2]):
                    with column:
                        st.markdown(f"**{result['model']}** · {result['predicted_label']}")
                        if "quantum_decision_score" in result:
                            st.metric("QSVM decision score", f"{result['quantum_decision_score']:.4f}")
                            st.caption("Uncalibrated decision score, not a probability.")
                        else:
                            st.metric("Positive-class probability estimate", f"{result['positive_class_probability_estimate']:.1%}")
                            st.caption("Uncalibrated model probability estimate, not medical risk.")
            st.caption("No consensus or diagnosis is produced. Extracted report content is not saved to disk.")
        except (ValueError, FileNotFoundError) as exc:
            st.error(str(exc))

st.caption("Pima dataset features do not cover every value commonly found in medical reports. Family history is not a substitute for DiabetesPedigreeFunction; fasting and random glucose are not interchangeable.")
