# Intelligent ML Experiment Assistant

A full-stack application for running, tracking, evaluating, and explaining machine-learning experiments using TensorFlow, FastAPI, Streamlit, and SQLite.

## Features

- Run TensorFlow experiments in background workers.
- Track experiment lifecycle: queued, running, evaluating, completed, or failed.
- Store epoch-wise training and validation metrics.
- Visualize loss and performance curves in a Streamlit dashboard.
- Evaluate classification models using final metrics and confusion-matrix-based results.
- Upload custom CSV and ZIP datasets.
- Inspect CSV columns, data types, missing values, unique values, and sample data.
- Configure tabular classification or regression experiments.
- Detect overfitting, underfitting, validation degradation, class imbalance, and small test sets using deterministic rules.
- Compare experiment results and generate reports.
- Run the application with Docker Compose.
- Support optional grounded Gemini explanations using experiment metrics and diagnostics.

## Technology Stack

| Area | Technology |
| --- | --- |
| Frontend | Streamlit |
| Backend | FastAPI |
| Model Training | TensorFlow / Keras |
| Data Processing | pandas, scikit-learn |
| Database | SQLite |
| Background Jobs | Python subprocess workers |
| Containerization | Docker and Docker Compose |
| Optional LLM | Google Gemini API |

## Architecture

```text
Streamlit Dashboard
        |
        v
FastAPI Backend
        |
        +--> SQLite Database
        |    experiments, metrics, evaluations, diagnostics
        |
        +--> Background Training Worker
                    |
                    v
             TensorFlow Model Training
                    |
                    v
        Saved Metrics, Artifacts, and Evaluations
```

## Supported Workflows

### Image Classification

- Fashion-MNIST baseline experiments
- TensorFlow CNN training
- Model evaluation and diagnostics
- Experiment tracking and comparison

### Tabular CSV Data

- CSV dataset upload and inspection
- Target-column selection
- Classification and regression experiments
- Numeric and categorical feature preprocessing
- TensorFlow dense neural-network training
- Stored training metrics and final evaluation

## Run Locally

### 1. Create a virtual environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Run the FastAPI backend

```powershell
uvicorn app.main:app --reload
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

### 4. Run the Streamlit dashboard

Open a second terminal, activate the virtual environment, then run:

```powershell
streamlit run dashboard\app.py
```

Dashboard:

```text
http://localhost:8501
```

## Run with Docker

```powershell
docker compose up --build
```

- Dashboard: `http://localhost:8501`
- API documentation: `http://localhost:8000/docs`

## Optional Gemini Setup

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

Do not upload `.env` or your API key to GitHub.

## Planned Enhancements

- Grounded Gemini explanations based only on saved metrics and diagnostics.
- Hyperparameter recommendations.
- Custom image classification from ZIP datasets.
- Text-classification workflows.
- Time-series forecasting workflows.
- Unified experiment comparison across image, tabular, text, and time-series data.

## Author

**Kartikey**

- GitHub: https://github.com/Kartikey0507-tech
