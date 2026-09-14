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

---

## Next Major Milestone

Model and data drift monitoring.

Implementation details, tests, metrics, screenshots, and documentation will be added after this milestone is built and verified.

---

## Future Work

Future production-oriented extensions may include automated retraining, stronger online behavioral feature serving, graph-based fraud detection, and additional observability.

These are roadmap items only and are not claimed as currently implemented capabilities.
