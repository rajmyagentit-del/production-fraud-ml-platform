from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from xgboost import XGBClassifier


MODEL_PATH = Path("models/behavioral_xgboost.json")
DECISION_THRESHOLD = 0.98

app = FastAPI(
    title="Production Fraud ML API",
    version="0.1.0",
    description=(
        "Real-time fraud scoring API using a behavioral "
        "XGBoost model trained on PaySim."
    ),
)


class FraudRequest(BaseModel):
    step: int = Field(ge=1)
    transaction_type: str
    amount: float = Field(ge=0)
    oldbalanceOrg: float = Field(ge=0)
    oldbalanceDest: float = Field(ge=0)

    orig_prior_txn_count: float = Field(ge=0)
    orig_prior_amount_mean: float = Field(ge=0)

    orig_steps_since_prev_txn: float
    orig_account_age_steps: float = Field(ge=0)

    amount_vs_prior_mean: float = Field(ge=0)
    orig_dest_prior_txn_count: float = Field(ge=0)

    dest_seen_before: int = Field(ge=0, le=1)


class FraudResponse(BaseModel):
    fraud_probability: float
    threshold: float
    prediction: str
    risk_level: str


_model: XGBClassifier | None = None


def load_model() -> XGBClassifier:
    global _model

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}"
            )

        model = XGBClassifier()
        model.load_model(MODEL_PATH)
        _model = model

    return _model


def build_feature_frame(
    request: FraudRequest,
) -> pd.DataFrame:
    transaction_types = {
        "CASH_IN",
        "CASH_OUT",
        "DEBIT",
        "PAYMENT",
        "TRANSFER",
    }

    if request.transaction_type not in transaction_types:
        raise ValueError(
            "transaction_type must be one of: "
            "CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER"
        )

    amount_to_orig_balance = (
        request.amount / (request.oldbalanceOrg + 1.0)
    )

    data = {
        "step": float(request.step),
        "amount": float(request.amount),
        "oldbalanceOrg": float(request.oldbalanceOrg),
        "oldbalanceDest": float(request.oldbalanceDest),
        "hour": float(request.step % 24),
        "day": float(request.step // 24),
        "amount_to_orig_balance": float(
            amount_to_orig_balance
        ),
        "orig_prior_txn_count": float(
            request.orig_prior_txn_count
        ),
        "orig_prior_amount_mean": float(
            request.orig_prior_amount_mean
        ),
        "orig_steps_since_prev_txn": float(
            request.orig_steps_since_prev_txn
        ),
        "orig_account_age_steps": float(
            request.orig_account_age_steps
        ),
        "amount_vs_prior_mean": float(
            request.amount_vs_prior_mean
        ),
        "orig_dest_prior_txn_count": float(
            request.orig_dest_prior_txn_count
        ),
        "dest_seen_before": float(
            request.dest_seen_before
        ),
        "type_CASH_IN": int(
            request.transaction_type == "CASH_IN"
        ),
        "type_CASH_OUT": int(
            request.transaction_type == "CASH_OUT"
        ),
        "type_DEBIT": int(
            request.transaction_type == "DEBIT"
        ),
        "type_PAYMENT": int(
            request.transaction_type == "PAYMENT"
        ),
        "type_TRANSFER": int(
            request.transaction_type == "TRANSFER"
        ),
    }

    return pd.DataFrame([data])


def risk_level(probability: float) -> str:
    if probability >= DECISION_THRESHOLD:
        return "HIGH"

    if probability >= 0.50:
        return "MEDIUM"

    return "LOW"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL_PATH.exists(),
    }


@app.get("/model-info")
def model_info():
    return {
        "model": "XGBoost behavioral fraud classifier",
        "dataset": "PaySim synthetic fraud dataset",
        "decision_threshold": DECISION_THRESHOLD,
        "threshold_strategy": "best F1 threshold",
    }


@app.post(
    "/predict",
    response_model=FraudResponse,
)
def predict(request: FraudRequest):
    try:
        model = load_model()
        features = build_feature_frame(request)

        probability = float(
            model.predict_proba(features)[0, 1]
        )

        prediction = (
            "fraud"
            if probability >= DECISION_THRESHOLD
            else "legitimate"
        )

        return FraudResponse(
            fraud_probability=round(
                probability,
                6,
            ),
            threshold=DECISION_THRESHOLD,
            prediction=prediction,
            risk_level=risk_level(probability),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc
