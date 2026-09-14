# Architecture

This document explains how the Production Fraud ML Platform is structured from raw transaction data through deployed fraud scoring.

The platform is designed around four main engineering goals:

- prevent temporal and point-in-time leakage
- handle extreme class imbalance
- generate behavioral fraud signals efficiently at multi-million-row scale
- expose the trained model through a tested production-style API and dashboard

---

## High-Level Architecture

```text
PaySim Synthetic Transactions
          ↓
Data Inspection and Validation
          ↓
Strict Chronological Evaluation Design
          ↓
Point-in-Time Behavioral Feature Generation
          ↓
DuckDB-backed Parquet Feature Store
          ↓
Behavioral XGBoost Model
          ↓
Threshold Optimization
          ↓
FastAPI Inference Service
          ↓
Fraud Risk Intelligence Dashboard
          ↓
Render Public Deployment
```

---

## 1. Data Layer

The project uses the PaySim synthetic financial transaction dataset.

Development data characteristics:

- 6,362,620 transactions
- 8,213 fraud transactions
- approximately 0.129% fraud rate
- 11 raw columns

The raw file is expected at:

```text
data/raw/transactions.csv
```

The raw dataset is not committed to Git because it is approximately 493 MB.

The first pipeline stage performs data inspection and validation before feature generation or model training.

---

## 2. Temporal Leakage Controls

Fraud systems are highly vulnerable to data leakage because information created after a transaction can accidentally be used to predict that same transaction.

This project uses two important controls.

### Chronological Train/Test Split

Training uses earlier simulation steps and evaluation uses later steps.

```text
steps 1-354   → training
steps 355-743 → held-out future evaluation
```

This is more realistic than randomly mixing past and future transactions.

### Point-in-Time Feature Generation

Behavioral aggregates are calculated only from PRIOR simulation steps.

Transactions from the same step are treated as simultaneous.

Therefore, one transaction in a step cannot become historical information for another transaction in that same step.

---

## 3. Behavioral Feature Layer

Behavioral features are generated with DuckDB and written to Parquet.

Output:

```text
data/processed/behavioral_features.parquet
```

DuckDB was selected because it can efficiently process multi-million-row analytical workloads without requiring a separate database server.

Examples of generated behavioral signals include:

- prior transaction count for the origin account
- prior transaction amount sum
- prior mean transaction amount
- steps since the previous transaction
- account age in simulation steps
- current amount relative to historical mean
- prior origin-to-destination transaction count
- whether the destination was previously observed

These features try to represent transaction behavior instead of relying only on the current transaction itself.

The feature pipeline is intentionally disk-backed so the full dataset does not need to remain in application memory.

---

## 4. Model Layer

The production modeling path uses an XGBoost classifier trained on the behavioral feature store.

The training pipeline performs a strict chronological split so future transactions are not used to train the model that evaluates them.

The model is saved as:

```text
models/behavioral_xgboost.json
```

Key evaluation results from the behavioral model:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.999906 |
| PR-AUC | 0.974028 |
| Precision @ 0.50 | 0.770368 |
| Recall @ 0.50 | 0.979333 |
| F1 @ 0.50 | 0.862372 |

Because fraud is extremely rare in PaySim, PR-AUC is emphasized more heavily than accuracy.

An earlier benchmark using post-transaction balance-derived information produced even stronger results, but those results are treated as leakage-prone and are not used as the production performance claim.

---

## 5. Decision Threshold Layer

The classifier outputs a fraud probability rather than a final business decision.

Threshold analysis evaluates probability cutoffs from 0.01 to 0.99.

The deployed API currently uses:

```text
threshold = 0.98
```

At 0.98, the development evaluation produced approximately:

| Metric | Value |
|---|---:|
| Precision | 0.929336 |
| Recall | 0.877172 |
| F1 | 0.902501 |

The 0.98 threshold was selected because it produced the best F1 score in the evaluated range.

The project also contains illustrative cost-sensitive threshold analysis, but those costs are relative units rather than real currency.

---

## 6. Serving Layer

The trained model is exposed through FastAPI.

Main endpoints:

| Endpoint | Purpose |
|---|---|
| `/` | fraud risk dashboard |
| `/health` | service and model health |
| `/model-info` | model and threshold metadata |
| `/predict` | fraud probability and classification |
| `/docs` | interactive Swagger interface |

The API constructs the model feature vector from transaction fields and behavioral aggregates supplied in the request.

The model returns:

- fraud probability
- configured threshold
- fraud or legitimate prediction
- risk level

Risk levels are currently interpreted as:

```text
HIGH   → probability >= 0.98
MEDIUM → probability >= 0.50 and < 0.98
LOW    → probability < 0.50
```

---

## 7. Dashboard Layer

The root endpoint serves a recruiter-facing Fraud Risk Intelligence Dashboard.

The dashboard provides:

- interactive transaction inputs
- example fraud and lower-risk transactions
- real-time calls to the `/predict` endpoint
- fraud probability and risk classification
- model performance metrics
- direct links to Swagger, health, and model information

The dashboard is intentionally lightweight and uses the same FastAPI application as the inference service.

This means the public demo exercises the actual deployed model rather than displaying static example results.

---

## 8. Deployment Layer

The application is publicly deployed on Render.

```text
GitHub main branch
        ↓
Render build
        ↓
Install Python 3.11 dependencies
        ↓
Load behavioral_xgboost.json
        ↓
Start Uvicorn / FastAPI
        ↓
Public HTTPS application
```

The deployment uses `render.yaml` for service configuration and `.python-version` to pin Python 3.11.

Public application:

https://production-fraud-ml-platform.onrender.com

---

## 9. CI/CD

The repository uses GitHub Actions for continuous integration and Render for continuous deployment.

```text
Developer Push / Pull Request
             ↓
          GitHub
             ↓
      GitHub Actions CI
             ↓
Install Python 3.11 Environment
             ↓
        Run pytest
             ↓
        Tests Pass
             ↓
      main branch update
             ↓
     Render auto-deploy
             ↓
     Public application
```

GitHub Actions currently runs the automated test suite on pushes and pull requests to `main`.

Render watches the repository and automatically deploys updates from the configured branch.

This provides a basic production-style path from source-code change to automated validation and deployment.

---

## 10. Current Architecture Limitation

The current public system performs real model inference, but it does not yet contain a production online feature store.

At prediction time, behavioral aggregates such as prior transaction count and historical mean amount are supplied by the caller.

Current architecture:

```text
Transaction
     +
Precomputed Behavioral Aggregates
              ↓
           FastAPI
              ↓
           XGBoost
              ↓
        Fraud Decision
```

This limitation is documented explicitly so the demo does not imply capabilities that have not yet been implemented.

In a real high-throughput payment system, behavioral state would typically be calculated or retrieved automatically before model inference.

---

## 11. Target Production Architecture

A future production-grade evolution could use:

```text
Transaction Event Stream
          ↓
Streaming / Online Feature Computation
          ↓
Online Behavioral Feature Store
          ↓
Fraud Model Service
          ↓
Threshold / Decision Engine
          ↓
Approve / Review / Block Decision
          ↓
Prediction + Feature Logging
          ↓
Drift and Performance Monitoring
          ↓
Retraining Pipeline
          ↓
Model Validation and Deployment
```

Implemented lifecycle capabilities now include:

- feature and prediction drift monitoring
- drift-triggered retraining decisions
- chronological challenger training
- champion/challenger evaluation on the same untouched future window
- precision, recall, PR-AUC, and false-positive promotion gates
- explicit prevention of automatic model promotion

Remaining engineering extensions include:

- stronger online behavioral feature computation
- formal model registry and version management
- controlled production promotion and rollback
- graph-based fraud signals
- additional production observability

The lifecycle layer deliberately separates retraining from deployment. A challenger may improve fraud capture while still being rejected when operational tradeoffs such as precision loss or false-positive growth exceed configured limits.

---

## Architecture Principles

The project follows several core principles:

1. **Time matters** — evaluate future transactions using models trained only on the past.
2. **Features must be point-in-time correct** — future or simultaneous information must not leak into historical aggregates.
3. **Imbalance matters** — emphasize metrics such as PR-AUC, precision, and recall instead of accuracy alone.
4. **Model probability is not the final decision** — threshold selection is an explicit system component.
5. **Offline and online behavior should match** — production feature computation should eventually reproduce training-time feature semantics.
6. **Claims must match implementation** — current limitations are documented instead of hidden.
7. **A model is only one part of an ML system** — testing, APIs, deployment, monitoring, and retraining are part of the architecture.
