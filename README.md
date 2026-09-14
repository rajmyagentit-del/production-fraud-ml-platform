# Production Fraud ML Platform

[![CI](https://github.com/rajmyagentit-del/production-fraud-ml-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/rajmyagentit-del/production-fraud-ml-platform/actions/workflows/ci.yml)

A production-style fraud detection ML platform built to demonstrate realistic machine-learning engineering challenges including temporal leakage, extreme class imbalance, behavioral feature engineering, threshold optimization, model serving, testing, CI, and public cloud deployment.

The system is built around PaySim, a synthetic financial transaction dataset containing more than 6.36 million transactions.

---

## Live Demo

**Fraud Risk Intelligence Dashboard**

https://production-fraud-ml-platform.onrender.com

**Interactive Swagger API**

https://production-fraud-ml-platform.onrender.com/docs

> The free Render service can require a short cold start after inactivity.

---

## Why This Project Exists

Fraud detection systems look simple when treated as ordinary classification problems, but production fraud ML introduces harder engineering problems:

- fraud is extremely rare compared with legitimate activity
- random data splitting can produce misleading evaluation results
- post-transaction information can accidentally leak the answer into the model
- user behavior must be reconstructed using only information available before the transaction being scored
- classification thresholds must balance false positives and false negatives
- a trained model is not useful until it can be tested, served, and deployed reproducibly

This project was built to address those problems rather than only train a classifier.

---

## What Is Implemented

- PaySim data ingestion and quality inspection
- validation for more than 6.36 million transactions
- strict chronological train/test evaluation
- investigation of leakage-prone balance-derived features
- point-in-time behavioral feature engineering
- DuckDB-backed feature generation for large data
- XGBoost fraud classification
- PR-AUC, ROC-AUC, precision, recall, F1, and confusion-matrix evaluation
- decision-threshold optimization
- MLflow experiment logging
- packaged XGBoost deployment artifact
- FastAPI inference service
- Fraud Risk Intelligence Dashboard
- interactive Swagger interface
- automated pytest test suite
- GitHub Actions continuous integration
- automatic Render deployment from GitHub
- public HTTPS inference
- deployment evidence and reproducibility documentation

---

## Dataset

The project uses PaySim, a synthetic mobile-money transaction dataset.

Dataset summary:

| Item | Value |
|---|---:|
| Transactions | 6,362,620 |
| Fraud transactions | 8,213 |
| Legitimate transactions | 6,354,407 |
| Fraud rate | approximately 0.129% |
| Time steps | 1-743 |

PaySim is synthetic and should not be described as real customer transaction data.

---

## Leakage-Aware Modeling

An early benchmark using post-transaction balance-derived information produced near-perfect metrics.

Those results are intentionally not treated as production performance because the balance fields can encode simulator-specific information that would not necessarily be available safely at real-time scoring.

The production modeling path therefore uses:

1. strict chronological evaluation
2. historical features generated from prior time steps only
3. exclusion of known leakage-prone signals from the production claim

This distinction is one of the most important design choices in the project.

### Leakage-Prone Benchmark

| Metric | Value |
|---|---:|
| ROC-AUC | 0.999977 |
| PR-AUC | 0.998693 |
| Precision | 0.973361 |
| Recall | 0.987776 |
| F1 | 0.980516 |

These values are documented for investigation purposes only.

---

## Behavioral Model Performance

The final behavioral XGBoost model uses current transaction information together with leakage-aware historical behavioral features.

### Evaluation at the Default 0.50 Threshold

| Metric | Value |
|---|---:|
| ROC-AUC | 0.999906 |
| PR-AUC | 0.974028 |
| Precision | 0.770368 |
| Recall | 0.979333 |
| F1 | 0.862372 |

### Deployed Threshold: 0.98

| Metric | Value |
|---|---:|
| Precision | 0.929336 |
| Recall | 0.877172 |
| F1 | 0.902501 |

The deployed threshold is 0.98 because it produced the best F1 score in the current threshold analysis.

PR-AUC is emphasized because fraud is extremely imbalanced and ranking quality on the positive class matters more than plain accuracy.

---

## System Architecture

```text
PaySim Transactions
        |
        v
Data Quality Validation
        |
        v
Strict Chronological Split
        |
        v
Point-in-Time Behavioral Features
        |
        v
DuckDB Feature Generation
        |
        v
XGBoost Fraud Model
        |
        v
Threshold Optimization
        |
        v
FastAPI Inference Service
        |
        +----> Fraud Risk Dashboard
        |
        +----> Swagger API
        |
        v
Render Public HTTPS Deployment
```

The historical features are generated using prior time steps so that future transaction information is not intentionally introduced into the behavioral history.

See [Architecture Documentation](docs/ARCHITECTURE.md) for the detailed design.

---

## Point-in-Time Behavioral Features

The behavioral pipeline reconstructs information that would have been available before the transaction being evaluated.

Examples include:

- prior transaction count for the origin account
- prior average transaction amount
- time since the previous transaction
- observed account age
- transaction amount relative to historical behavior
- previous origin-to-destination interaction count
- whether the destination has been seen before

The large feature-generation workload is executed with DuckDB and persisted as Parquet for downstream training.

---

## Verified Public Inference

A held-out fraud example was submitted to the deployed `/predict` endpoint.

The public service returned HTTP 200 with:

```json
{
  "fraud_probability": 0.987317,
  "threshold": 0.98,
  "prediction": "fraud",
  "risk_level": "HIGH"
}
```

A separate lower-risk transaction was also tested and returned a LOW risk classification.

Deployment and API screenshots are available in [Project Evidence](docs/PROJECT_EVIDENCE.md).

---

## Fraud Risk Intelligence Dashboard

The root application provides an interactive dashboard where a user can:

- enter transaction information
- load example transactions
- send the transaction to the deployed model
- view fraud probability
- view the configured decision threshold
- inspect the fraud/legitimate decision
- inspect the LOW, MEDIUM, or HIGH risk level

The dashboard calls the same FastAPI `/predict` endpoint exposed through the public API.

---

## API Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /` | Fraud Risk Intelligence Dashboard |
| `GET /health` | service and model availability |
| `GET /model-info` | model and threshold information |
| `POST /predict` | real-time fraud scoring |
| `GET /docs` | interactive Swagger API |

See [API Guide](docs/API_GUIDE.md) for request fields and examples.

---

## Automated Testing

The latest verified local test run completed with:

```text
21 passed, 1 warning
```

The warning is a non-blocking Starlette/TestClient deprecation warning observed during development.

Tests cover feature logic, validation, temporal splitting, threshold behavior, behavioral features, and API behavior.

---

## CI/CD

GitHub Actions automatically installs the project and runs the pytest suite for pushes and pull requests targeting `main`.

Render is connected to the repository and automatically rebuilds and deploys updates from the configured branch.

```text
Git push
   |
   +----> GitHub Actions ----> Automated Tests
   |
   +----> Render -----------> Build and Deploy
                                  |
                                  v
                           Public HTTPS App
```

---

## Reproduce the Project Locally

The repository is documented so another engineer can reproduce the core workflow rather than only inspect the finished model.

Basic setup:

```bash
git clone https://github.com/rajmyagentit-del/production-fraud-ml-platform.git
cd production-fraud-ml-platform
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,api]"
```

After obtaining PaySim and placing it at `data/raw/transactions.csv`, the core ML workflow is:

```bash
python -m fraud_ml.inspect_data
python -m fraud_ml.build_behavioral_features
python -m fraud_ml.train_behavioral
python -m fraud_ml.optimize_threshold
pytest -q
```

Run the application locally with:

```bash
uvicorn fraud_ml.api:app --host 0.0.0.0 --port 8000 --reload
```

For the complete reproduction walkthrough, see [Local Setup](docs/LOCAL_SETUP.md).

---

## Documentation

| Document | Purpose |
|---|---|
| [Local Setup](docs/LOCAL_SETUP.md) | reproduce the project from setup through inference |
| [Architecture](docs/ARCHITECTURE.md) | understand the current system design |
| [Model Card](docs/MODEL_CARD.md) | model performance, intended use, and limitations |
| [API Guide](docs/API_GUIDE.md) | test and integrate with the FastAPI service |
| [Deployment Guide](docs/DEPLOYMENT.md) | understand the Render deployment workflow |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | diagnose common setup and deployment problems |
| [Project Evidence](docs/PROJECT_EVIDENCE.md) | inspect screenshots and working evidence |
| [Roadmap](docs/ROADMAP.md) | current implementation status and next milestone |

---

## Current Limitations

- PaySim is synthetic; results are not evidence of performance on real customer transactions.
- behavioral history is currently supplied to the public inference API rather than retrieved from an online feature store
- the public deployment does not currently include production authentication or rate limiting
- model and data drift monitoring is implemented with chronological reference/current windows, feature PSI, transaction-type total variation, model-score PSI, label-shift tracking, and held-out performance reporting
- drift-triggered retraining orchestration and champion/challenger evaluation are implemented
- automatic production model promotion is intentionally disabled
- no formal model registry, approval workflow, or automated rollback mechanism is implemented yet

These limitations are stated explicitly to separate demonstrated capabilities from future engineering work.

---

## Next Major Milestone

The platform now includes drift-triggered retraining orchestration, chronological challenger training, champion/challenger evaluation on the same future window, and promotion governance.

The next major milestone is a controlled model registry and promotion workflow with explicit approval, model versioning, rollback protection, and auditable deployment decisions.
