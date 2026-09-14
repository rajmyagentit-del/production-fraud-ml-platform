# Production Fraud ML Platform

[![CI](https://github.com/rajmyagentit-del/production-fraud-ml-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/rajmyagentit-del/production-fraud-ml-platform/actions/workflows/ci.yml)


A production-style fraud detection ML platform focused on realistic ML engineering challenges including **temporal leakage, extreme class imbalance, behavioral feature engineering, model thresholding, API serving, testing, and public cloud deployment**.

## Live Demo

**Public API**  
https://production-fraud-ml-platform.onrender.com

**Interactive Swagger Demo**  
https://production-fraud-ml-platform.onrender.com/docs

> Render's free instance can require a short cold start after inactivity.

## Current Status

Implemented so far:

- PaySim pipeline with 6.36M+ synthetic transactions
- data-quality and schema validation
- strict chronological train/test split
- leakage-aware modeling
- point-in-time behavioral features
- DuckDB-backed feature generation
- XGBoost fraud classifier
- threshold optimization
- FastAPI inference service
- packaged model artifact
- public HTTPS deployment on Render
- 21 automated tests
- verified public fraud inference

## Verified Live Prediction

A held-out fraud transaction sent to the deployed API returned:

```json
{
  "fraud_probability": 0.987317,
  "threshold": 0.98,
  "prediction": "fraud",
  "risk_level": "HIGH"
}
```

## Model Performance

### Behavioral XGBoost Model

| Metric | Value |
|---|---:|
| ROC-AUC | 0.999906 |
| PR-AUC | 0.974028 |
| Precision @ 0.50 | 0.770368 |
| Recall @ 0.50 | 0.979333 |
| F1 @ 0.50 | 0.862372 |

### Deployed Threshold — 0.98

| Metric | Value |
|---|---:|
| Precision | 0.929336 |
| Recall | 0.877172 |
| F1 | 0.902501 |

## Architecture

```text
PaySim
   ↓
Data Validation
   ↓
Strict Temporal Split
   ↓
Point-in-Time Behavioral Features
   ↓
DuckDB Feature Store
   ↓
XGBoost
   ↓
Threshold Optimization
   ↓
FastAPI
   ↓
Render
   ↓
Public HTTPS Inference
```

## Leakage-Aware Evaluation

An early benchmark using post-transaction balance-derived information produced near-perfect results.

Those results are intentionally not presented as production performance because such fields can encode simulator-specific leakage.

The production modeling path instead uses strict chronological validation and historical behavioral features.

## Automated Tests

Latest verified test run:

```text
21 passed, 1 warning
```

The warning is a non-blocking Starlette/TestClient deprecation warning.

## Proof That the Project Works

Deployment screenshots, public API screenshots, the held-out fraud request, successful HTTP 200 response, architecture, and model metrics are documented here:

**[View Working Evidence](docs/PROJECT_EVIDENCE.md)**

## API Endpoints

| Endpoint | Purpose |
|---|---|
| `/` | service metadata |
| `/health` | deployment health |
| `/model-info` | model information |
| `/predict` | fraud inference |
| `/docs` | interactive Swagger interface |

## Dataset

This project uses **PaySim**, a synthetic financial transaction dataset.

Results should not be interpreted as performance on real customer financial transactions.

## Current Limitations

The current API performs real-time model scoring, but behavioral aggregates are supplied with the request rather than being generated from a production online feature store.

## Next Milestones

- recruiter-facing Fraud Risk Intelligence Dashboard
- GitHub Actions CI/CD
- architecture and metrics visualizations
- model/data drift monitoring
- automated retraining
- graph-based fraud detection
- additional production observability
