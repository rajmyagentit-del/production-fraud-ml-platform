from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from fraud_ml.drift import (
    build_drift_report,
    calculate_prediction_score_drift,
    save_report,
)
from fraud_ml.train_behavioral import FEATURE_SQL


FEATURE_PATH = Path("data/processed/behavioral_features.parquet")
MODEL_PATH = Path("models/behavioral_xgboost.json")
REPORT_PATH = Path("reports/drift/drift_report.json")
TEMP_DIRECTORY = Path("data/processed/duckdb_tmp")

REFERENCE_END_STEP = 355

NUMERIC_FEATURES = [
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "orig_prior_txn_count",
    "orig_prior_amount_mean",
    "orig_steps_since_prev_txn",
    "orig_account_age_steps",
    "amount_vs_prior_mean",
    "orig_dest_prior_txn_count",
    "dest_seen_before",
]


def load_window(
    connection: duckdb.DuckDBPyConnection,
    where_clause: str,
):
    columns = ["step", "type", "isFraud", *NUMERIC_FEATURES]
    select_columns = ", ".join(columns)

    query = f"""
        SELECT {select_columns}
        FROM read_parquet(?)
        WHERE {where_clause}
        ORDER BY step
    """

    return connection.execute(
        query,
        [str(FEATURE_PATH)],
    ).fetch_df()


def load_scoring_model() -> XGBClassifier:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    return model


def build_model_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    features = pd.DataFrame(
        {
            "step": frame["step"].astype(float),
            "amount": frame["amount"].astype(float),
            "oldbalanceOrg": frame["oldbalanceOrg"].astype(float),
            "oldbalanceDest": frame["oldbalanceDest"].astype(float),
            "hour": (frame["step"] % 24).astype(float),
            "day": np.floor(frame["step"] / 24).astype(float),
            "amount_to_orig_balance": (
                frame["amount"]
                / (frame["oldbalanceOrg"] + 1.0)
            ).astype(float),
            "orig_prior_txn_count": frame[
                "orig_prior_txn_count"
            ].astype(float),
            "orig_prior_amount_mean": frame[
                "orig_prior_amount_mean"
            ].astype(float),
            "orig_steps_since_prev_txn": frame[
                "orig_steps_since_prev_txn"
            ].astype(float),
            "orig_account_age_steps": frame[
                "orig_account_age_steps"
            ].astype(float),
            "amount_vs_prior_mean": frame[
                "amount_vs_prior_mean"
            ].astype(float),
            "orig_dest_prior_txn_count": frame[
                "orig_dest_prior_txn_count"
            ].astype(float),
            "dest_seen_before": frame[
                "dest_seen_before"
            ].astype(float),
            "type_CASH_IN": (
                frame["type"] == "CASH_IN"
            ).astype(int),
            "type_CASH_OUT": (
                frame["type"] == "CASH_OUT"
            ).astype(int),
            "type_DEBIT": (
                frame["type"] == "DEBIT"
            ).astype(int),
            "type_PAYMENT": (
                frame["type"] == "PAYMENT"
            ).astype(int),
            "type_TRANSFER": (
                frame["type"] == "TRANSFER"
            ).astype(int),
        }
    )

    return features


def score_frame(
    model: XGBClassifier,
    frame: pd.DataFrame,
    batch_size: int = 250_000,
) -> pd.Series:
    chunks: list[np.ndarray] = []

    for start in range(0, len(frame), batch_size):
        batch = frame.iloc[
            start : start + batch_size
        ]

        features = build_model_features(batch)

        probabilities = model.predict_proba(
            features
        )[:, 1]

        chunks.append(
            probabilities.astype(np.float64)
        )

    return pd.Series(
        np.concatenate(chunks),
        index=frame.index,
        name="fraud_probability",
    )


def evaluate_heldout_performance(
    labels: pd.Series,
    scores: pd.Series,
    threshold: float = 0.98,
) -> dict:
    y_true = labels.to_numpy(dtype=int)
    y_score = scores.to_numpy(dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "evaluation_window": "held_out_current_window",
        "threshold": threshold,
        "rows": int(len(y_true)),
        "fraud_cases": int(y_true.sum()),
        "roc_auc": round(
            float(roc_auc_score(y_true, y_score)),
            6,
        ),
        "pr_auc": round(
            float(average_precision_score(y_true, y_score)),
            6,
        ),
        "precision": round(
            float(precision_score(y_true, y_pred)),
            6,
        ),
        "recall": round(
            float(recall_score(y_true, y_pred)),
            6,
        ),
        "f1": round(
            float(f1_score(y_true, y_pred)),
            6,
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def main() -> None:
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"Behavioral feature file not found: {FEATURE_PATH}"
        )

    TEMP_DIRECTORY.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect()
    connection.execute("SET memory_limit='2GB'")
    connection.execute("SET threads=2")
    connection.execute(
        f"SET temp_directory='{TEMP_DIRECTORY}'"
    )

    print("Loading reference window: steps < 355")
    reference = load_window(
        connection,
        f"step < {REFERENCE_END_STEP}",
    )

    print("Loading current window: steps >= 355")
    current = load_window(
        connection,
        f"step >= {REFERENCE_END_STEP}",
    )

    for frame in (reference, current):
        frame["hour"] = (
            frame["step"] % 24
        ).astype(float)

        frame["day"] = np.floor(
            frame["step"] / 24
        ).astype(float)

        frame["amount_to_orig_balance"] = (
            frame["amount"]
            / (frame["oldbalanceOrg"] + 1.0)
        )

    print("Loading trained XGBoost model...")
    model = load_scoring_model()

    print("Scoring reference window...")
    reference_scores = score_frame(
        model,
        reference,
    )

    print("Scoring current window...")
    current_scores = score_frame(
        model,
        current,
    )

    connection.close()

    print(f"Reference rows: {len(reference):,}")
    print(f"Current rows: {len(current):,}")

    report = build_drift_report(
        reference=reference,
        current=current,
        numeric_features=[
            *NUMERIC_FEATURES,
            "step",
            "hour",
            "day",
            "amount_to_orig_balance",
        ],
        categorical_feature="type",
    )

    reference_fraud_rate = float(
        reference["isFraud"].mean()
    )
    current_fraud_rate = float(
        current["isFraud"].mean()
    )

    report["label_monitoring"] = {
        "reference_fraud_rate": round(
            reference_fraud_rate,
            8,
        ),
        "current_fraud_rate": round(
            current_fraud_rate,
            8,
        ),
        "fraud_rate_ratio": round(
            current_fraud_rate
            / max(reference_fraud_rate, 1e-12),
            4,
        ),
    }

    report["heldout_performance"] = (
        evaluate_heldout_performance(
            current["isFraud"],
            current_scores,
            threshold=0.98,
        )
    )

    score_metric = calculate_prediction_score_drift(
        reference_scores,
        current_scores,
    )

    report["metrics"].append(
        asdict(score_metric)
    )

    severity_rank = {
        "LOW": 0,
        "MEDIUM": 1,
        "HIGH": 2,
    }

    report["overall_severity"] = max(
        report["overall_severity"],
        score_metric.severity,
        key=lambda value: severity_rank[value],
    )

    report["model_score_reference_rows"] = int(
        len(reference_scores)
    )
    report["model_score_current_rows"] = int(
        len(current_scores)
    )

    report["reference_window"] = "step < 355"
    report["current_window"] = "step >= 355"
    report["reference_step_max"] = 354
    report["current_step_min"] = 355

    save_report(report, REPORT_PATH)

    print("")
    print("Drift analysis complete")
    print(f"Overall severity: {report['overall_severity']}")
    print(f"Report: {REPORT_PATH}")
    print("")

    for metric in report["metrics"]:
        print(
            f"{metric['feature']:<30} "
            f"{metric['metric']:<18} "
            f"{metric['value']:<10} "
            f"{metric['severity']}"
        )


if __name__ == "__main__":
    main()
