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
