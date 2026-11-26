# Q-Diabetes application

The first application milestone is a manual-input Streamlit prediction page
using the exported Random Forest pipeline. It runs locally and does not upload
or retain submitted values.

## Run locally

From the repository root in PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
python -m streamlit run app\dashboard.py
```

The application expects the reviewed artifact at
`models/classical/random_forest_pipeline.joblib`. It does not train or create
that artifact at startup. If the artifact is missing, the page explains how to
export it.

The model is a research prototype, not a clinically validated diagnostic
service. Its positive-class probability estimate is not calibrated.
