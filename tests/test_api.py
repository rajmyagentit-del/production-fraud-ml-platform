from fastapi.testclient import TestClient

from fraud_ml.api import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info():
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["decision_threshold"] == 0.98
    assert data["threshold_strategy"] == "best F1 threshold"


def test_invalid_transaction_type():
    payload = {
        "step": 100,
        "transaction_type": "INVALID",
        "amount": 1000.0,
        "oldbalanceOrg": 2000.0,
        "oldbalanceDest": 500.0,
        "orig_prior_txn_count": 2,
        "orig_prior_amount_mean": 400.0,
        "orig_steps_since_prev_txn": 3,
        "orig_account_age_steps": 50,
        "amount_vs_prior_mean": 2.5,
        "orig_dest_prior_txn_count": 0,
        "dest_seen_before": 0,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 400


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

    html = response.text

    assert "Fraud Risk Intelligence Dashboard" in html
    assert 'fetch("/predict"' in html
    assert "/docs" in html
    assert "/model-info" in html
    assert "/health" in html
    assert "/drift" in html
    assert "Model &amp; Data Drift Monitoring" in html
    assert 'fetch("/drift"' in html
    assert "/lifecycle" in html
    assert "Model Lifecycle &amp; Retraining" in html
    assert 'fetch("/lifecycle"' in html
    assert "Candidate Status" in html
    assert "Automatic Promotion" in html
    assert "Held-out PR-AUC" in html
    assert "Fraud Rate Shift" in html
    assert "Interpretation:" in html
    assert "35 Passed" in html


def test_drift_endpoint():
    response = client.get("/drift")

    assert response.status_code == 200
    payload = response.json()

    assert payload["overall_severity"] == "HIGH"

    metrics = {
        metric["feature"]: metric
        for metric in payload["metrics"]
    }

    assert "model_fraud_probability" in metrics
    assert (
        metrics["model_fraud_probability"]["severity"]
        == "HIGH"
    )

    assert "step" in metrics
    assert metrics["step"]["severity"] == "HIGH"

    assert "day" in metrics
    assert metrics["day"]["severity"] == "HIGH"

    assert payload["label_monitoring"]["fraud_rate_ratio"] > 4.0
    assert payload["reference_rows"] == 5069097
    assert payload["current_rows"] == 1293523
    assert payload["reference_window"] == "step < 355"
    assert payload["current_window"] == "step >= 355"
    assert isinstance(payload["metrics"], list)
    assert len(payload["metrics"]) >= 1


def test_lifecycle_endpoint_returns_model_lifecycle():
    response = client.get("/lifecycle")

    assert response.status_code == 200

    payload = response.json()

    assert payload["trigger"]["retraining_recommended"] is True
    assert payload["trigger"]["auto_promotion_allowed"] is False

    assert (
        payload["promotion"]["decision"]
        == "KEEP_PRODUCTION"
    )

    reasons = payload["promotion"]["reasons"]

    assert any(
        "precision degradation" in reason
        for reason in reasons
    )

    assert any(
        "false-positive increase" in reason
        for reason in reasons
    )

    assert payload["candidate_cutoff_step"] == 525
    assert payload["evaluation_window"] == "step >= 525"

    assert (
        payload["candidate_metrics"]["pr_auc"]
        > payload["champion_metrics"]["pr_auc"]
    )

    assert (
        payload["candidate_metrics"]["fn"]
        < payload["champion_metrics"]["fn"]
    )
