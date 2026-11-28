# Q-Diabetes application

The manual-input Streamlit prediction page runs four classical pipelines:
Logistic Regression, Random Forest, RBF-SVM, and Gradient Boosting, plus QSVM.
Each model uses its own saved preprocessing steps. VQC is not available because
its original trained weights were not saved. The app runs locally and does not
upload or retain submitted values.

## Run locally

From the repository root in PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
pip install -r app\requirements.txt
python -m streamlit run app\dashboard.py
```

The dashboard can run in the existing app environment while document parsing
runs in a separate local Python 3.12 worker. The uploaded bytes are passed to
that worker over stdin; no report file is written. Set up the worker with:

```powershell
py -3.12 -m venv .venv-documents
.\.venv-documents\Scripts\Activate.ps1
pip install -r app\requirements-documents.txt
deactivate
```

From the normal application environment, validate the image and scanned-PDF
OCR paths using synthetic fixtures only:

```powershell
python -m app.validate_ocr
```

The worker defaults to `.venv-documents\Scripts\python.exe`. For a different
location, set `QD_OCR_PYTHON` to the Python 3.12 executable. On a deployment
machine, run the synthetic validation command before enabling report uploads.

The application expects these classical artifacts under `models/classical/`:

- `logistic_regression_pipeline.joblib`
- `random_forest_pipeline.joblib`
- `rbf_svm_pipeline.joblib`
- `gradient_boosting_pipeline.joblib`

The QSVM artifact is `models/quantum/qsvm_pipeline.joblib`.

Run `python -m app.verify_qsvm` to independently reconstruct Notebook 05 and
compare the 154 held-out outputs. The verifier is a manual command, not part
of app startup, and does not write any artifacts or research result files.
Run `python -m app.analyze_qsvm_metrics` to inspect score orientation and
compare current AUC/AP calculations with the archived QSVM metrics; it reuses
the saved pipeline and never refits or writes results.

Run the automated inference and report-field mapping checks with
`python -m unittest discover -s app/tests -v` from the repository root.

Export scripts are in `app/export_*.py`. The dashboard never trains or creates
model artifacts at startup.

This is a research prototype, not a clinically validated diagnostic service.
Positive-class probability estimates are not calibrated.
