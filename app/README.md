# Q-Diabetes application

The manual-input Streamlit prediction page runs four classical pipelines:
Logistic Regression, Random Forest, RBF-SVM, and Gradient Boosting, plus QSVM.
Each model uses its own saved preprocessing steps. VQC is not available because
its original trained weights were not saved. The app runs locally and does not
send submitted values to external services or persist them to application files.
Report-review values remain in active Streamlit session memory until cleared or
the session ends.

## Run locally

From the repository root in PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
pip install -r app\requirements.txt
python -m streamlit run app\dashboard.py --server.address 127.0.0.1
```

The dashboard can run in the existing app environment while document parsing
runs in a separate local Python 3.12 worker. The uploaded bytes are passed to
that worker over stdin; no report file is written. OCR is stopped after 90
seconds. Set up the worker with:

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

The validation covers reduced resolution, a 2.5-degree rotation, alternate
layout and type size, multiple glucose readings, missing glucose units, scanned
PDF extraction, and safe rejection of corrupt and password-protected PDFs.
These are software fixtures, not a substitute for validation on a diverse set
of real report formats or general medical-report extraction accuracy. **Every
extracted medical value requires explicit user confirmation regardless of OCR
confidence.**

The worker defaults to `.venv-documents\Scripts\python.exe`. For a different
location, set `QD_OCR_PYTHON` to the Python 3.12 executable. On a deployment
machine, run the synthetic validation command before enabling report uploads.

The application expects these classical artifacts under `models/classical/`:

- `logistic_regression_pipeline.joblib`
- `random_forest_pipeline.joblib`
- `rbf_svm_pipeline.joblib`
- `gradient_boosting_pipeline.joblib`

The QSVM artifact is `models/quantum/qsvm_pipeline.joblib`.
Model artifacts are local, ignored by Git, and must be present in the deployment
environment before prediction is available. The dashboard does not fit or
export models at startup. Use the existing `app/export_*.py` commands only when
you intentionally need to recreate an artifact from its preserved research
implementation.

## Report data handling

The report upload is held in Streamlit's in-memory upload/session handling and
sent to the local OCR worker through stdin. The worker writes no report or
extracted-value files and does not call a network OCR service. Candidate values,
editable fields, confirmations, and prediction outputs remain in the active
Streamlit session until **Clear report and extracted values** is used or that
session ends. The clear action removes these session keys. The subprocess has a
90-second timeout; its pipes and PDF document handles are closed on completion,
and timeout termination is handled by `subprocess.run`.

Run `python -m app.verify_qsvm` to independently reconstruct Notebook 05 and
compare the 154 held-out outputs. The verifier is a manual command, not part
of app startup, and does not write any artifacts or research result files.
Run `python -m app.analyze_qsvm_metrics` to inspect score orientation and
compare current AUC/AP calculations with the archived QSVM metrics; it reuses
the saved pipeline and never refits or writes results.

Run the automated inference, report-field mapping, error-handling, and
Streamlit end-to-end upload/review/confirmation checks with
`python -m unittest discover -s app/tests -v` from the repository root.

## Deployment checks

From a clean checkout, install the application and document-worker requirements
in their respective environments, ensure the local model artifacts above are
provisioned, then run:

```powershell
python -m pip check
python -m app.validate_ocr
python -m unittest discover -s app/tests -v
python -m streamlit run app/dashboard.py --server.address 127.0.0.1
```

The OCR worker path is resolved relative to the repository, not the current
working directory. Set `QD_OCR_PYTHON` if the Python 3.12 worker lives elsewhere.
This setup has been validated as a local Windows research prototype; remote
hosting needs a deployment-specific review of artifact provisioning, network
exposure, session isolation, and resource limits. For a local-only launch, bind
the server to loopback with `--server.address 127.0.0.1`; explicitly choose and
secure a broader bind address only when remote access is intended.

Export scripts are in `app/export_*.py`. The dashboard never trains or creates
model artifacts at startup.

This is a research prototype, not a clinically validated diagnostic service.
Positive-class probability estimates are not calibrated.
