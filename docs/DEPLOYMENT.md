# Deployment Guide

This guide documents how the Production Fraud ML Platform is deployed as a public HTTPS service.

The current public deployment uses Render.

---

## Public Deployment

Application:

```text
https://production-fraud-ml-platform.onrender.com
```

Swagger:

```text
https://production-fraud-ml-platform.onrender.com/docs
```

The root URL serves the Fraud Risk Intelligence Dashboard.

---

## Deployment Platform

The application is deployed as a Render web service.

Render is responsible for:

- creating the Python runtime
- installing the project dependencies
- starting the FastAPI application
- exposing the service over HTTPS
- running the configured health check
- automatically redeploying changes from the connected GitHub branch

---

## Render Configuration

Deployment configuration is stored in:

```text
render.yaml
```

The service uses:

```text
Build command:
pip install -e ".[api]"

Start command:
uvicorn fraud_ml.api:app --host 0.0.0.0 --port $PORT

Health check:
/health
```

The application must bind to `0.0.0.0` and use the port provided by Render through the `PORT` environment variable.

---

## Python Version Pinning

The project requires Python 3.11.

The repository contains:

```text
.python-version
```

with:

```text
3.11
```

The project metadata also restricts Python to:

```text
>=3.11,<3.12
```

Python 3.11 is pinned so that the deployed environment matches the development environment and supported dependency range.

---

## Deployment Issue Encountered

An earlier Render deployment attempted to use Python 3.14.3.

The project metadata requires Python 3.11 and does not support Python 3.14.

This caused the deployment build to fail before the application could start.

The issue was fixed by adding the repository-level `.python-version` file with:

```text
3.11
```

After the Python runtime was pinned correctly, the Render deployment succeeded.

This is an important reproducibility lesson: cloud runtime versions should be explicitly controlled rather than relying on provider defaults.

---

## GitHub to Render Deployment Flow

The current delivery workflow is:

```text
Code change
    |
    v
Git commit and push to main
    |
    +----> GitHub Actions CI
    |          |
    |          v
    |      Automated tests
    |
    v
Render detects the updated main branch
    |
    v
Build application
    |
    v
Start FastAPI service
    |
    v
Run health check
    |
    v
Public HTTPS application
```

GitHub Actions and Render therefore perform different responsibilities.

GitHub Actions provides continuous integration by automatically running the test suite.

Render provides continuous deployment behavior by automatically rebuilding and deploying changes from the connected branch.

---

## Deployment Verification

A successful cloud build alone is not considered sufficient verification.

After deployment, verify the actual public service.

Recommended checks:

1. confirm the Render deployment reports success
2. open the public dashboard
3. call `/health`
4. call `/model-info`
5. open `/docs`
6. submit a known fraud example to `/predict`
7. confirm an HTTP 200 response
8. verify the returned probability, prediction, and risk level
9. submit a lower-risk example
10. confirm the dashboard remains functional

The repository contains screenshots from this verification process in `docs/images/`.

Additional evidence is documented in [Project Evidence](PROJECT_EVIDENCE.md).

---

## Health Check

Render uses:

```text
/health
```

The endpoint reports service status and whether the packaged model artifact is available.

This helps distinguish an available API process from a deployment where the required model artifact is missing.

---

## Free-Tier Cold Starts

The current deployment uses a free Render service.

After a period of inactivity, the service may sleep.

The first request after inactivity can therefore take longer while the service starts again.

This cold-start behavior is a hosting limitation and should not be interpreted as model inference latency.

---

## CI and Deployment Responsibilities

The project currently separates validation and deployment responsibilities:

| Component | Responsibility |
|---|---|
| GitHub Actions | install the project and run automated tests |
| Render | build and deploy the public application |
| FastAPI `/health` | expose runtime and model availability |
| Public API tests | verify the deployed application actually responds |

The GitHub Actions workflow is stored in:

```text
.github/workflows/ci.yml
```

---

## Deployment Recovery

If a deployment fails:

1. inspect the Render build and runtime logs
2. identify whether the failure occurred during dependency installation or application startup
3. confirm Python 3.11 is being used
4. verify the model artifact exists in `models/behavioral_xgboost.json`
5. verify the FastAPI module imports successfully
6. run the automated tests before pushing a fix
7. push the corrected commit
8. verify the replacement deployment through the public endpoints

Git history also provides known-good commits that can be used when diagnosing regressions.

---

## Current Deployment Limitations

The public deployment demonstrates a working ML inference application, but it is not a complete financial-production environment.

Current limitations include:

- free-tier hosting and cold starts
- no production authentication or authorization layer
- no production rate limiting or abuse controls
- no online behavioral feature store
- no production transaction database
- drift monitoring is available through the deployed `/drift` endpoint and public dashboard
- lifecycle and retraining decisions are available through the deployed `/lifecycle` endpoint and dashboard
- drift-triggered challenger training and champion/challenger evaluation are implemented
- automatic model promotion is intentionally disabled
- the evaluated challenger artifact is not part of the deployed production model bundle
- no formal model registry, approval workflow, or automated rollback mechanism is implemented yet

These limitations are documented deliberately so that implemented capabilities are clearly separated from roadmap work.

---

## Related Documentation

- [Local Setup](LOCAL_SETUP.md)
- [Architecture](ARCHITECTURE.md)
- [API Guide](API_GUIDE.md)
- [Model Card](MODEL_CARD.md)
- [Project Evidence](PROJECT_EVIDENCE.md)
