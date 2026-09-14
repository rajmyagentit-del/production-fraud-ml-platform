# Troubleshooting Guide

This guide covers common issues that may occur while reproducing, testing, or deploying the Production Fraud ML Platform.

---

## 1. Python Version Error

### Symptom

Installation or deployment fails because the active Python version is outside the supported range.

The project requires:

```text
>=3.11,<3.12
```

### Fix

Use Python 3.11.

Verify with:

```bash
python --version
```

Expected:

```text
Python 3.11.x
```

The repository also contains `.python-version` with `3.11` for deployment reproducibility.

---

## 2. PaySim Dataset Not Found

### Symptom

Training or feature-generation scripts fail because:

```text
data/raw/transactions.csv
```

does not exist.

### Fix

Obtain the PaySim dataset and place the CSV at:

```text
data/raw/transactions.csv
```

Do not commit the raw dataset to GitHub.

The repository `.gitignore` intentionally excludes the raw CSV.

---

## 3. Behavioral Feature File Missing

### Symptom

Behavioral model training fails because:

```text
data/processed/behavioral_features.parquet
```

is missing.

### Fix

Generate the point-in-time behavioral features first:

```bash
python -m fraud_ml.build_behavioral_features
```

Then verify the Parquet file exists before running behavioral-model training.

---

## 4. Model Artifact Missing

### Symptom

The API starts but reports that the model is unavailable, or `/predict` cannot score transactions.

### Expected Artifact

```text
models/behavioral_xgboost.json
```

### Fix

Train the behavioral model:

```bash
python -m fraud_ml.train_behavioral
```

The deployment repository intentionally tracks this model artifact so the public service can load it at startup.

---

## 5. Render Deployment Uses the Wrong Python Version

### Symptom

Render attempts to build with a newer Python version such as Python 3.14 and dependency installation fails.

### Fix

Confirm the repository contains:

```text
.python-version
```

with:

```text
3.11
```

Then redeploy the service.

This exact issue occurred during development and was resolved by pinning Python 3.11.

---

## 6. Render Cold Start

### Symptom

The public dashboard or API takes noticeably longer to respond after a period of inactivity.

### Cause

The free Render service can sleep when idle.

### Fix

Wait for the service to wake and retry the request.

The first request after inactivity can be slower than normal.

Do not interpret this cold-start delay as model inference latency.

---

## 7. API Request Validation Error

### Symptom

`POST /predict` returns a client error instead of a prediction.

### Common Causes

- a required field is missing
- a field has an invalid type
- the request body is not valid JSON
- an unsupported transaction type is supplied

### Supported Transaction Types

```text
CASH_IN
CASH_OUT
DEBIT
PAYMENT
TRANSFER
```

### Fix

Compare the request with the examples in [API Guide](API_GUIDE.md).

---

## 8. Behavioral Fields Missing from Prediction Request

### Symptom

The API cannot score a transaction because behavioral inputs are absent.

### Current Requirement

The caller currently supplies values such as:

```text
orig_prior_txn_count
orig_prior_amount_mean
orig_steps_since_prev_txn
orig_account_age_steps
amount_vs_prior_mean
orig_dest_prior_txn_count
dest_seen_before
```

### Why

The current public API does not yet have an online feature store that automatically reconstructs behavioral history.

### Fix

Provide the required behavioral fields in the request body.

---

## 9. Tests Pass with a Warning

### Symptom

Pytest reports all tests passing but also displays a Starlette or TestClient deprecation warning.

### Current Status

The warning observed during development is non-blocking.

The latest verified test run was:

```text
21 passed, 1 warning
```

### Fix

If the tests pass, the warning does not currently block the project.

Dependency modernization can be handled separately when the warning becomes actionable.

---

## 10. Behavioral Feature Generation Uses Significant Resources

### Symptom

Feature generation is slow or appears memory constrained.

### Context

The PaySim dataset contains more than 6.36 million transactions.

The point-in-time behavioral feature pipeline therefore uses DuckDB and writes the final result to Parquet.

The current script configures:

```text
memory_limit = 2GB
threads = 2
```

### Fix

Allow the process time to complete and make sure enough disk space is available for intermediate and output files.

Do not replace the point-in-time logic with a simpler cumulative calculation unless same-step leakage behavior is carefully preserved.

---

## 11. Terminal Shows a `>` Prompt

### Symptom

The shell displays:

```text
>
```

instead of the normal repository prompt.

### Cause

A pasted command usually contains an unfinished quote, escape sequence, or multiline shell expression.

### Fix

Press:

```text
Ctrl+C
```

Then confirm the terminal returns to the normal shell prompt before trying another command.

For long documentation edits in this project, short `printf` blocks are safer than large heredoc pastes.

---

## 12. Large Data or Generated Files Do Not Appear in Git

### Symptom

Files such as the raw PaySim CSV, generated Parquet features, MLflow data, or reports do not appear in `git status`.

### Cause

These files are intentionally ignored because they are large or reproducible artifacts.

Examples include:

```text
data/raw/
data/processed/
mlruns/
mlflow.db
reports/
```

The production model artifact is an intentional exception:

```text
models/behavioral_xgboost.json
```

### Fix

Do not force-add ignored datasets or generated files unless the repository design is intentionally changed.

---

## 13. Recovery Checklist

If the project is not working and the cause is unclear, check the system in this order:

1. confirm Python 3.11
2. confirm dependencies are installed
3. confirm `data/raw/transactions.csv` exists for training workflows
4. confirm behavioral features exist if retraining
5. confirm `models/behavioral_xgboost.json` exists for API inference
6. run `pytest -q`
7. start the API locally
8. test `/health`
9. test `/model-info`
10. test `/predict`
11. inspect Render logs if the local application works but cloud deployment fails

---

## Related Documentation

- [Local Setup](LOCAL_SETUP.md)
- [API Guide](API_GUIDE.md)
- [Deployment Guide](DEPLOYMENT.md)
- [Architecture](ARCHITECTURE.md)
- [Model Card](MODEL_CARD.md)
