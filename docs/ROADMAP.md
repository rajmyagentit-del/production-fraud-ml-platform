# Project Roadmap

This document tracks the current implementation status of the Production Fraud ML Platform.

Detailed documentation for future capabilities will be added only after those capabilities are actually implemented, tested, and verified.

---

## Completed

- PaySim data ingestion and validation
- strict chronological train/test evaluation
- leakage investigation
- leakage-aware modeling path
- point-in-time behavioral feature engineering
- DuckDB-backed feature generation
- XGBoost behavioral fraud model
- model evaluation using PR-AUC, ROC-AUC, precision, recall, and F1
- threshold optimization
- MLflow experiment tracking
- FastAPI inference service
- packaged deployment model
- Fraud Risk Intelligence Dashboard
- public HTTPS deployment on Render
- Swagger API interface
- automated pytest test suite
- GitHub Actions CI
- Render automatic deployment from GitHub
- deployment verification and evidence
- local reproduction documentation
- architecture documentation
- model card
- API documentation
- deployment documentation
- troubleshooting documentation
- model and data drift monitoring
- model-score distribution monitoring
- label-shift monitoring
- held-out performance monitoring
- drift-triggered retraining policy
- chronological challenger model training
- champion/challenger evaluation on a shared future window
- promotion gating using PR-AUC, recall, precision, and false-positive controls
- explicit prevention of automatic production promotion

---

## Next Major Milestone

Controlled model registry, approval, and rollback workflow.

The next milestone will focus on:

- explicit model version management
- controlled human approval before production promotion
- auditable promotion decisions
- safe production model replacement
- rollback protection
- preserving previous champion versions

---

## Future Work

Future production-oriented extensions may include stronger online behavioral feature serving, graph-based fraud detection, production authentication, persistent observability infrastructure, and richer model registry integration.

These are roadmap items only and are not claimed as currently implemented capabilities.
