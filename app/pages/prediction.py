"""Q-Diabetes manual-input prediction app (first inference milestone)."""

from __future__ import annotations

import streamlit as st

from app.services.random_forest_inference import FEATURES
from app.services.classical_inference import ARTIFACTS, predict_classical


MODEL_CHOICES = [*ARTIFACTS, "QSVM"]


st.markdown(
    """
    <style>
    :root { --ink:#122235; --muted:#66778a; --teal:#119b89; --blue:#5578df; }
    html, body, [class*="css"] { font-family:'Segoe UI',system-ui,sans-serif; }
    .stApp { background:linear-gradient(180deg,#f7fafc 0%,#f2f6f9 100%); }
    [data-testid="stSidebar"] { background:#102132; }
    [data-testid="stSidebar"] * { color:#eaf2f6; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#a9bdcb; }
    .brand { font:800 1.45rem Manrope,sans-serif; letter-spacing:-.04em; color:#f6fbfd; }
    .brand span { color:#39c3aa; }
    .side-tag { color:#a9bdcb; font-size:.85rem; margin-top:-.35rem; }
    .hero { background:linear-gradient(115deg,#12293c,#193e51 68%,#146e70); border-radius:24px; padding:2rem 2.2rem; color:white; margin:.35rem 0 1.4rem; box-shadow:0 16px 42px #16354b1c; }
    .eyebrow { color:#83e0ce; font-size:.75rem; letter-spacing:.14em; font-weight:700; text-transform:uppercase; }
    .hero h1 { font:800 clamp(1.8rem,3vw,2.65rem) 'Segoe UI',system-ui,sans-serif; letter-spacing:-.045em; margin:.45rem 0 .45rem; color:white; }
    .hero p { color:#d2e4eb; max-width:780px; margin:0; line-height:1.6; }
    .panel { background:white; border:1px solid #e3ebf0; border-radius:18px; padding:1.35rem 1.5rem; box-shadow:0 8px 24px #18354b08; }
    .panel-title { font:700 1.12rem 'Segoe UI',system-ui,sans-serif; color:var(--ink); margin:0 0 .2rem; }
    .panel-sub { color:var(--muted); font-size:.9rem; margin:0 0 1rem; }
    .result { border-radius:18px; padding:1.35rem 1.5rem; background:#ecf8f5; border:1px solid #c8e9e1; margin-top:1rem; }
    .result-label { color:#54756f; text-transform:uppercase; letter-spacing:.12em; font-size:.72rem; font-weight:700; }
    .result-class { font:800 1.7rem 'Segoe UI',system-ui,sans-serif; color:#123f3b; margin:.25rem 0; }
    .result-copy { color:#55716d; margin:0; }
    .disclaimer { background:#fff8e8; color:#775d2b; border:1px solid #f2e1b8; border-radius:12px; padding:.85rem 1rem; font-size:.84rem; line-height:1.5; }
    [data-testid="stFormSubmitButton"] button, button[data-testid="stBaseButton-primary"] { background:linear-gradient(90deg,#119b89,#16aa98)!important; border:0!important; border-radius:10px!important; color:white!important; font-weight:700!important; min-height:2.8rem; }
    [data-testid="stFormSubmitButton"] button:hover, button[data-testid="stBaseButton-primary"]:hover { background:#0c8577!important; border:0!important; }
    @media (max-width:700px) { .hero { padding:1.4rem; border-radius:18px; } }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="brand">Q<span>·</span>Diabetes</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-tag">Hybrid Quantum-Classical Intelligence</div>', unsafe_allow_html=True)
    st.divider()
    st.caption("Research prototype · Local inference")

st.markdown(
    """
    <section class="hero">
      <div class="eyebrow">Q-Diabetes · Research workspace</div>
      <h1>Diabetes prediction</h1>
      <p>Enter the eight Pima dataset features below. Each selected saved model applies its own verified training preprocessing and returns an individual output.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.7, 1], gap="large")
with left:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Medical parameters</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-sub">Enter values from a measured report. Fields start blank; no values are prefilled.</div>', unsafe_allow_html=True)

        with st.form("manual_prediction", clear_on_submit=False):
            col_a, col_b = st.columns(2, gap="large")
            values = {}
            field_spec = [
                ("Pregnancies", "count", 1.0, "Number of pregnancies"),
                ("Glucose", "mg/dL", 1.0, "Plasma glucose concentration"),
                ("BloodPressure", "mmHg", 1.0, "Diastolic blood pressure"),
                ("SkinThickness", "mm", 1.0, "Triceps skin fold thickness"),
                ("Insulin", "μU/mL", 1.0, "2-hour serum insulin"),
                ("BMI", "kg/m²", 0.1, "Body mass index"),
                ("DiabetesPedigreeFunction", "score", 0.01, "Diabetes pedigree function"),
                ("Age", "years", 1.0, "Age in years"),
            ]
            for index, (name, unit, step, help_text) in enumerate(field_spec):
                target_col = col_a if index % 2 == 0 else col_b
                with target_col:
                    values[name] = st.number_input(
                        f"{name} ({unit})",
                        min_value=0.0,
                        value=None,
                        step=step,
                        placeholder="Enter measured value",
                        help=help_text,
                        key=f"feature_{name}",
                    )
            selected_models = st.multiselect(
                "Models to run",
                options=MODEL_CHOICES,
                default=MODEL_CHOICES,
                help="Each selected model is evaluated independently using its own saved pipeline.",
            )
            submitted = st.form_submit_button("Run selected predictions", type="primary", width="stretch")

with right:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Inference details</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-sub">Separate outputs from each selected model</div>', unsafe_allow_html=True)
        st.markdown("**Classical models**  \nLogistic Regression · Random Forest · RBF-SVM · Gradient Boosting")
        st.markdown("**Quantum model available**  \nQSVM · PennyLane simulator")
        st.markdown("**VQC status**  \nUnavailable · trained weights were not saved")
        st.markdown("**Input schema**  \nEight Pima dataset features in training order")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="disclaimer"><strong>Research use only.</strong> This model is not clinically validated and does not provide a diagnosis. Its output is not a substitute for advice from a qualified healthcare professional.</div>', unsafe_allow_html=True)

st.info(
    "Training-set caveat: classical models used 614 training samples, while QSVM used a "
    "stratified 250-sample subset. Their preprocessing also differs, so live output "
    "agreement is not evidence of correctness or a controlled architecture comparison."
)
with st.expander("QSVM verification and research context"):
    st.write(
        "A separate Notebook 05 reconstruction matched the exported QSVM's predicted "
        "classes and decision scores on all 154 held-out rows. The original notebook "
        "did not save its row-level prediction vector, so verification uses an independent "
        "re-execution of its recorded configuration."
    )
    st.write(
        "The saved research CSV ROC-AUC is 0.797593. The current PennyLane 0.45.1 "
        "reconstruction yielded 0.797407. Saved PR-AUC is 0.660188; reconstructed PR-AUC "
        "is 0.659795. The archived research values remain unchanged. Live QSVM decision "
        "scores are not probabilities."
    )

if submitted:
    st.session_state.pop("last_live_model_predictions", None)
    missing = [name for name in FEATURES if values.get(name) is None]
    if missing:
        st.error("Please enter all eight measured values before prediction. Missing: " + ", ".join(missing))
    else:
        try:
            if not selected_models:
                st.warning("Select at least one model to run.")
            else:
                results = []
                for name in selected_models:
                    if name == "QSVM":
                        from app.services.quantum_inference import predict_qsvm

                        results.append(predict_qsvm(values))
                    else:
                        results.append(predict_classical(name, values))
                st.session_state["last_live_model_predictions"] = results
                st.subheader("Individual model predictions")
                for start in range(0, len(results), 2):
                    row_results = results[start : start + 2]
                    result_columns = st.columns(len(row_results), gap="medium")
                    for column, result in zip(result_columns, row_results):
                        with column:
                            st.markdown(
                                f"""
                                <section class="result">
                                  <div class="result-label">{result['model']} · predicted class</div>
                                  <div class="result-class">{result['predicted_label']}</div>
                                  <p class="result-copy">Independent model output. Not a diagnosis or consensus.</p>
                                </section>
                                """,
                                unsafe_allow_html=True,
                            )
                            probability = result.get("positive_class_probability_estimate")
                            if probability is not None:
                                st.metric("Positive-class probability estimate", f"{probability:.1%}")
                                st.caption("Uncalibrated model probability estimate; not a clinical risk score.")
                            elif "quantum_decision_score" in result:
                                st.metric("QSVM decision score", f"{result['quantum_decision_score']:.4f}")
                                st.caption("Uncalibrated decision score; not a probability or medical risk estimate.")
        except (ValueError, FileNotFoundError) as exc:
            st.error(str(exc))

st.caption("Model output reflects the saved research pipeline. Agreement between models, when added, will not be treated as proof of correctness.")
