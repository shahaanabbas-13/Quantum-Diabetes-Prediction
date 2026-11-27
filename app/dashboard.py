"""Streamlit page router for the Q-Diabetes application."""

from __future__ import annotations

import streamlit as st


st.set_page_config(
    page_title="Q-Diabetes | Research Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = [
    st.Page("pages/prediction.py", title="New Prediction", icon="🩺", default=True),
    st.Page("pages/report_upload.py", title="Report Upload", icon="📄"),
    st.Page("pages/model_comparison.py", title="Model Comparison", icon="⚖️"),
    st.Page("pages/research_analytics.py", title="Research Analytics", icon="📊"),
]

st.navigation(pages, position="sidebar").run()
