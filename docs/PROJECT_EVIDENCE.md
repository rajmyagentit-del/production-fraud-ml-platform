# Production Fraud ML Platform — Working Evidence

This document contains reproducible evidence that the fraud ML platform is trained, tested, deployed, and performing live public inference.

## 1. Automated Test Suite

Latest verified test run:

```text
21 passed, 1 warning
```

The warning is a non-blocking Starlette/TestClient deprecation warning.

## 2. Public Cloud Deployment

Live Service: https://production-fraud-ml-platform.onrender.com

Interactive API: https://production-fraud-ml-platform.onrender.com/docs

![Successful Render deployment](images/01-render-deploy-success.png)

## 3. Interactive API Documentation

The deployed FastAPI service exposes:

- `GET /`
- `GET /health`
- `GET /model-info`
- `POST /predict`

![Interactive Swagger API](images/05-swagger-interactive-api.png)

## 4. Public Service Verification

![Live public service](images/02-live-public-root.png)

## 5. Public Fraud Prediction Request

```json
{
  "step": 355,
  "transaction_type": "TRANSFER",
  "amount": 42483.97,
  "oldbalanceOrg": 42483.97,
  "oldbalanceDest": 0.0,
  "orig_prior_txn_count": 0.0,
  "orig_prior_amount_mean": 0.0,
  "orig_steps_since_prev_txn": -1,
  "orig_account_age_steps": 0,
  "amount_vs_prior_mean": 0.0,
  "orig_dest_prior_txn_count": 0.0,
  "dest_seen_before": 0
}
```

![Public prediction request](images/03-public-predict-request.png)

## 6. Public Model Prediction Result

```json
{
  "fraud_probability": 0.987317,
  "threshold": 0.98,
  "prediction": "fraud",
  "risk_level": "HIGH"
}
```

![Successful public fraud prediction](images/04-public-predict-response-200.png)

## 7. Verified Model Performance

| Metric | Behavioral Model |
|---|---:|
| ROC-AUC | 0.999906 |
| PR-AUC | 0.974028 |
| Precision @ 0.50 | 0.770368 |
| Recall @ 0.50 | 0.979333 |
| F1 @ 0.50 | 0.862372 |

### Deployed threshold 0.98

| Metric | Value |
|---|---:|
| Precision | 0.929336 |
| Recall | 0.877172 |
| F1 | 0.902501 |

## 8. Data and Leakage Controls

PaySim contains 6,362,620 synthetic transactions, including 8,213 fraud cases.

Production modeling uses:

- strict chronological splitting
- point-in-time behavioral history
- no train/test time overlap
- exclusion of `isFlaggedFraud`
- separation of leakage-prone benchmark results from production results

## 9. Current Architecture

```text
PaySim
  ↓
Validation
  ↓
Temporal Split
  ↓
Behavioral Features
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

## 10. Current Limitations

Planned next:

- recruiter-facing dashboard
- GitHub Actions CI/CD
- drift monitoring
- automated retraining
- graph-based fraud detection
