# Local Setup Guide

This guide explains how to reproduce the Production Fraud ML Platform from a fresh machine.

The goal is not only to run the deployed API, but to reproduce the full workflow:

**dataset → validation → point-in-time behavioral features → model training → threshold optimization → tests → FastAPI → dashboard**

---

## 1. Prerequisites

You need:

- Git
- Python 3.11
- pip
- approximately 2+ GB of available memory
- several GB of free disk space for the PaySim dataset and generated features

> This project currently requires Python `>=3.11,<3.12`.

Check your Python version:

```bash
python --version
```

Expected:

```text
Python 3.11.x
```

---

## 2. Clone the Repository

```bash
git clone https://github.com/rajmyagentit-del/production-fraud-ml-platform.git
cd production-fraud-ml-platform
```

---

## 3. Create a Virtual Environment

Linux/macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

---

## 4. Install the Project

Install the ML package, developer tools, tests, and API dependencies:

```bash
pip install -e ".[dev,api]"
```

This installs the main stack including:

- pandas and NumPy
- scikit-learn
- XGBoost
- DuckDB
- PyArrow
- MLflow
- FastAPI and Uvicorn
- pytest

Verify the important packages:

```bash
python -c "import duckdb, fastapi, mlflow, pandas, sklearn, xgboost; print(\"Environment ready.\")"
```

---

## 5. Get the PaySim Dataset

This project uses the **PaySim synthetic financial transaction dataset**.

PaySim is synthetic data generated for fraud-detection research. Results from this project must not be interpreted as performance on real customer financial transactions.

The pipeline expects the dataset at exactly:

```text
data/raw/transactions.csv
```

Create the directory if needed:

```bash
mkdir -p data/raw
```

After obtaining the PaySim CSV, place or rename it as:

```text
production-fraud-ml-platform/
└── data/
    └── raw/
        └── transactions.csv
```

Verify the file:

```bash
ls -lh data/raw/transactions.csv
```

The dataset used to develop this project contained **6,362,620 transactions**, including **8,213 fraud transactions**.

> The raw PaySim CSV is intentionally excluded from Git because it is approximately 493 MB.

---

## 6. Inspect the Raw Dataset

Run:

```bash
python -m fraud_ml.inspect_data
```

This inspection step checks:

- row count
- schema and columns
- missing values
- duplicates
- transaction-type distribution
- fraud vs legitimate transaction counts
- overall fraud rate

The development dataset contained:

- 6,362,620 total transactions
- 8,213 fraud transactions
- 6,354,407 legitimate transactions
- fraud rate of approximately 0.129%

Because the classes are extremely imbalanced, accuracy alone is not a useful model metric.

---

## 7. Build Leakage-Safe Behavioral Features

Run:

```bash
python -m fraud_ml.build_behavioral_features
```

Expected output:

```text
data/processed/behavioral_features.parquet
```

This pipeline uses DuckDB to process the full multi-million-row dataset efficiently.

The most important design rule is:

**A transaction may use behavioral history only from PRIOR simulation steps.**

Transactions occurring in the same step are treated as simultaneous and are not allowed to become history for one another.

This helps prevent point-in-time leakage.

Generated behavioral features include:

- prior origin transaction count
- prior origin transaction amount sum
- prior origin mean transaction amount
- steps since previous origin transaction
- origin account age
- current amount relative to historical mean
- prior origin-to-destination transaction count
- whether the destination was previously seen

Verify the feature store:

```bash
ls -lh data/processed/behavioral_features.parquet
```

The generated feature store should contain the same 6,362,620 transactions as the raw dataset.

---

## 8. Train the Behavioral Fraud Model

Run:

```bash
python -m fraud_ml.train_behavioral
```

The training pipeline:

1. reads the behavioral feature store
2. performs a strict chronological split
3. trains an XGBoost classifier
4. handles severe class imbalance using `scale_pos_weight`
5. evaluates the held-out future period
6. logs metrics and parameters to MLflow
7. saves the trained model artifact

Expected model:

```text
models/behavioral_xgboost.json
```

Verify it:

```bash
ls -lh models/behavioral_xgboost.json
```

### Why a chronological split?

A random split can allow future patterns to leak into training.

This project instead evaluates the model in a production-like direction:

```text
past transactions → training
future transactions → testing
```

During development, the temporal cutoff was step 355.

Training used steps 1-354 and testing used steps 355-743.

---

## 9. Behavioral Model Performance

The leakage-aware behavioral XGBoost model achieved approximately:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.999906 |
| PR-AUC | 0.974028 |
| Precision @ 0.50 | 0.770368 |
| Recall @ 0.50 | 0.979333 |
| F1 @ 0.50 | 0.862372 |

Because fraud represents only about 0.129% of the dataset, **PR-AUC is one of the most important evaluation metrics in this project**.

ROC-AUC can appear extremely strong even when performance on the rare positive class is less useful, so it should not be interpreted alone.

---

## 10. Optimize the Fraud Decision Threshold

The model produces a fraud probability, but a production system must still decide what probability should trigger a fraud classification.

Run:

```bash
python -m fraud_ml.optimize_threshold
```

Expected report:

```text
reports/metrics/threshold_analysis.csv
```

The script evaluates thresholds from 0.01 through 0.99.

Development results:

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.50 | 0.770368 | 0.979333 | 0.862372 |
| 0.98 | 0.929336 | 0.877172 | 0.902501 |

The deployed API currently uses:

```text
threshold = 0.98
```

This threshold was selected because it produced the best F1 score in the evaluated range.

The repository also demonstrates cost-sensitive threshold analysis using illustrative relative cost units.

These cost values are not dollars or any other real currency.

---

## 11. Run the Automated Tests

Run:

```bash
pytest -q
```

The current project test suite covers areas including:

- feature engineering
- data validation
- temporal splitting
- behavioral features
- threshold evaluation
- FastAPI endpoints

The latest verified local run completed with **21 passing tests**.

GitHub Actions also runs the test suite automatically for pushes and pull requests to `main`.

---

## 12. Start the FastAPI Application

Run:

```bash
uvicorn fraud_ml.api:app --host 0.0.0.0 --port 8000 --reload
```

Open the dashboard locally:

```text
http://127.0.0.1:8000
```

Swagger API documentation:

```text
http://127.0.0.1:8000/docs
```

Health endpoint:

```text
http://127.0.0.1:8000/health
```

Model information:

```text
http://127.0.0.1:8000/model-info
```

---

## 13. Test a Verified Fraud Example

Use the dashboard or Swagger `/predict` endpoint with:

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

A verified deployed response was:

```json
{
  "fraud_probability": 0.987317,
  "threshold": 0.98,
  "prediction": "fraud",
  "risk_level": "HIGH"
}
```

---

## 14. Test a Lower-Risk Example

Example request:

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

This example produced a very low fraud probability in the deployed model.

---

## 15. View MLflow Experiments

Run:

```bash
mlflow ui
```

Then open:

```text
http://127.0.0.1:5000
```

Use MLflow to inspect training parameters and model metrics.

---

## 16. Important Generated Artifacts

After reproducing the full pipeline, you should have files similar to:

```text
data/raw/transactions.csv
data/processed/behavioral_features.parquet
models/behavioral_xgboost.json
reports/metrics/threshold_analysis.csv
mlflow.db
```

Large generated data files are intentionally excluded from Git.

Do not commit the raw PaySim CSV or generated feature store.

---

## 17. Current Production Limitation

The deployed API performs real-time model inference, but behavioral aggregates are currently supplied in the request.

Examples include:

```text
orig_prior_txn_count
orig_prior_amount_mean
orig_steps_since_prev_txn
orig_account_age_steps
```

A more advanced production system would calculate these features from an online feature store or streaming state service.

---

## 18. Public Demo

Live dashboard:

https://production-fraud-ml-platform.onrender.com

Interactive Swagger API:

https://production-fraud-ml-platform.onrender.com/docs

> The free Render service may require a short cold start after inactivity.

---

## 19. Troubleshooting

If the dataset is missing, verify:

```text
data/raw/transactions.csv
```

If the behavioral feature store is missing, run:

```bash
python -m fraud_ml.build_behavioral_features
```

If the trained model is missing, run:

```bash
python -m fraud_ml.train_behavioral
```

For more details, see [Troubleshooting Guide](TROUBLESHOOTING.md).

---

## 20. Next Reading

- [Architecture](ARCHITECTURE.md)
- [Model Card](MODEL_CARD.md)
- [API Guide](API_GUIDE.md)
- [Deployment Guide](DEPLOYMENT.md)
- [Project Evidence](PROJECT_EVIDENCE.md)
- [Roadmap](ROADMAP.md)

---

## Reproduction Flow Summary

```text
Clone Repository
      ↓
Install Python 3.11 Environment
      ↓
Install Dependencies
      ↓
Add PaySim transactions.csv
      ↓
Inspect Data
      ↓
Build Point-in-Time Behavioral Features
      ↓
Train Behavioral XGBoost
      ↓
Optimize Threshold
      ↓
Run Tests
      ↓
Start FastAPI
      ↓
Open Dashboard / Swagger
      ↓
Send Fraud Prediction
```

If all of these steps complete successfully, you have reproduced the core Production Fraud ML Platform locally.
