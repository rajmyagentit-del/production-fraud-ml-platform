# API Guide

This guide explains how to interact with the Production Fraud ML Platform inference API.

The API is built with FastAPI and serves the same behavioral XGBoost model used by the public Fraud Risk Intelligence Dashboard.

---

## Public API

Base URL:

```text
https://production-fraud-ml-platform.onrender.com
```

Interactive Swagger interface:

```text
https://production-fraud-ml-platform.onrender.com/docs
```

> The free Render instance may require a short cold start after inactivity.

---

## Available Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Fraud Risk Intelligence Dashboard |
| GET | `/health` | service and model health |
| GET | `/model-info` | model and threshold information |
| POST | `/predict` | fraud probability and risk prediction |
| GET | `/drift` | model and data drift monitoring report |
| GET | `/lifecycle` | retraining and champion/challenger lifecycle decision |
| GET | `/docs` | interactive Swagger API |

---

## 1. Health Check

Request:

```bash
curl https://production-fraud-ml-platform.onrender.com/health
```

The endpoint reports whether the service is running and whether the packaged model artifact is available.

---

## 2. Model Information

Request:

```bash
curl https://production-fraud-ml-platform.onrender.com/model-info
```

This endpoint exposes model metadata including the dataset description, configured threshold, and threshold-selection strategy.

The currently deployed decision threshold is:

```text
0.98
```

---

## 3. Prediction Endpoint

Endpoint:

```text
POST /predict
```

The request combines current transaction information with behavioral aggregates.

### Request Fields

| Field | Description |
|---|---|
| `step` | PaySim simulation step |
| `transaction_type` | CASH_IN, CASH_OUT, DEBIT, PAYMENT, or TRANSFER |
| `amount` | current transaction amount |
| `oldbalanceOrg` | origin balance before the transaction |
| `oldbalanceDest` | destination balance before the transaction |
| `orig_prior_txn_count` | number of prior origin transactions |
| `orig_prior_amount_mean` | mean amount of prior origin transactions |
| `orig_steps_since_prev_txn` | steps since the previous origin transaction |
| `orig_account_age_steps` | observed age of the origin account in steps |
| `amount_vs_prior_mean` | current amount relative to prior mean behavior |
| `orig_dest_prior_txn_count` | prior origin-to-destination transaction count |
| `dest_seen_before` | whether the destination was previously observed |

Behavioral values should represent information available before the transaction being scored.

---

## 4. Fraud Prediction Example

Example request body:

```json
{
  "step": 355,
  "transaction_type": "TRANSFER",
  "amount": 42483.97,
  "oldbalanceOrg": 42483.97,
  "oldbalanceDest": 0,
  "orig_prior_txn_count": 0,
  "orig_prior_amount_mean": 0,
  "orig_steps_since_prev_txn": -1,
  "orig_account_age_steps": 0,
  "amount_vs_prior_mean": 0,
  "orig_dest_prior_txn_count": 0,
  "dest_seen_before": 0
}
```

Verified deployed response:

```json
{
  "fraud_probability": 0.987317,
  "threshold": 0.98,
  "prediction": "fraud",
  "risk_level": "HIGH"
}
```

This example exceeds the deployed 0.98 threshold.

---

## 5. Lower-Risk Prediction Example

Example request body:

```json
{
  "step": 355,
  "transaction_type": "PAYMENT",
  "amount": 120,
  "oldbalanceOrg": 2500,
  "oldbalanceDest": 800,
  "orig_prior_txn_count": 8,
  "orig_prior_amount_mean": 135,
  "orig_steps_since_prev_txn": 2,
  "orig_account_age_steps": 50,
  "amount_vs_prior_mean": 0.89,
  "orig_dest_prior_txn_count": 3,
  "dest_seen_before": 1
}
```

This example produced a very low fraud probability and was classified as LOW risk.

---

## 6. Response Interpretation

The prediction response contains:

| Field | Meaning |
|---|---|
| `fraud_probability` | model-estimated fraud probability |
| `threshold` | probability cutoff used for the fraud decision |
| `prediction` | fraud or legitimate classification |
| `risk_level` | LOW, MEDIUM, or HIGH |

Current risk interpretation:

```text
HIGH   -> probability >= 0.98
MEDIUM -> probability >= 0.50 and < 0.98
LOW    -> probability < 0.50
```

---

## 7. Request Validation

Supported transaction types:

```text
CASH_IN
CASH_OUT
DEBIT
PAYMENT
TRANSFER
```

Unsupported transaction types are rejected by the API.

FastAPI also validates the JSON request structure before model inference.

---

## 8. Test with Swagger

Open:

https://production-fraud-ml-platform.onrender.com/docs

Then:

1. expand `POST /predict`
2. click **Try it out**
3. paste a JSON request body
4. click **Execute**
5. inspect the HTTP status and prediction response

Swagger is the easiest way to test the API without writing client code.

---

## 9. Run the API Locally

After installing the project dependencies, start the API with:

```bash
uvicorn fraud_ml.api:app --host 0.0.0.0 --port 8000 --reload
```

Local URLs:

```text
Dashboard:   http://127.0.0.1:8000
Swagger:     http://127.0.0.1:8000/docs
Health:      http://127.0.0.1:8000/health
Model info:  http://127.0.0.1:8000/model-info
```

---

## 10. Python Client Example

Example using `requests`:

Install the optional client library if needed:

```bash
pip install requests
```

```python
import requests

payload = {
    "step": 355,
    "transaction_type": "TRANSFER",
    "amount": 42483.97,
    "oldbalanceOrg": 42483.97,
    "oldbalanceDest": 0,
    "orig_prior_txn_count": 0,
    "orig_prior_amount_mean": 0,
    "orig_steps_since_prev_txn": -1,
    "orig_account_age_steps": 0,
    "amount_vs_prior_mean": 0,
    "orig_dest_prior_txn_count": 0,
    "dest_seen_before": 0,
}

response = requests.post(
    "https://production-fraud-ml-platform.onrender.com/predict",
    json=payload,
)

print(response.json())
```

The same request can also be sent to a locally running API by changing the URL to:

```text
http://127.0.0.1:8000/predict
```

---

## 11. Current API Limitation

The API performs real-time XGBoost inference, but it does not yet generate behavioral history from an online feature store.

The following behavioral values are currently supplied by the caller:

```text
orig_prior_txn_count
orig_prior_amount_mean
orig_steps_since_prev_txn
orig_account_age_steps
amount_vs_prior_mean
orig_dest_prior_txn_count
dest_seen_before
```

In a more complete production system, these features would be retrieved or computed automatically from recent transactional state before inference.

The current API therefore demonstrates production-style model serving without claiming a complete online feature platform.

---

## Related Documentation

- [Local Setup](LOCAL_SETUP.md)
- [Architecture](ARCHITECTURE.md)
- [Model Card](MODEL_CARD.md)
- [Project Evidence](PROJECT_EVIDENCE.md)
