from __future__ import annotations

import gc
import json
from pathlib import Path

import duckdb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from fraud_ml.retraining import (
    evaluate_candidate_promotion,
    evaluate_retraining_policy,
)
from fraud_ml.train_behavioral import load_partition


DRIFT_REPORT_PATH = Path(
    "reports/drift/drift_report.json"
)
PRODUCTION_MODEL_PATH = Path(
    "models/behavioral_xgboost.json"
)
CANDIDATE_MODEL_PATH = Path(
    "models/candidates/behavioral_xgboost_step_525.json"
)
LIFECYCLE_REPORT_PATH = Path(
    "reports/lifecycle/lifecycle_report.json"
)

CANDIDATE_CUTOFF_STEP = 525
DECISION_THRESHOLD = 0.98


def evaluate_model(model, X, y) -> dict:
    scores = model.predict_proba(X)[:, 1]
    predictions = (
        scores >= DECISION_THRESHOLD
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "roc_auc": float(
            roc_auc_score(y, scores)
        ),
        "pr_auc": float(
            average_precision_score(y, scores)
        ),
        "precision": float(
            precision_score(
                y, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y, predictions, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y, predictions, zero_division=0
            )
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main() -> None:
    drift_report = json.loads(
        DRIFT_REPORT_PATH.read_text()
    )

    trigger = evaluate_retraining_policy(
        drift_report
    )

    print("Retraining recommended:",
          trigger.retraining_recommended)

    if not trigger.retraining_recommended:
        return

    con = duckdb.connect()
    con.execute("SET memory_limit = '2GB'")
    con.execute("SET threads = 2")

    print("Loading challenger training data...")

    train = load_partition(
        con,
        f"step < {CANDIDATE_CUTOFF_STEP}",
    )

    y_train = train.pop("isFraud")
    X_train = train

    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)

    candidate = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(
            negatives / max(positives, 1)
        ),
        eval_metric="aucpr",
        tree_method="hist",
        random_state=42,
        n_jobs=2,
    )

    print("Training challenger...")
    candidate.fit(X_train, y_train)

    del train, X_train, y_train
    gc.collect()

    print("Loading untouched future window...")

    future = load_partition(
        con,
        f"step >= {CANDIDATE_CUTOFF_STEP}",
    )

    y_future = future.pop("isFraud")
    X_future = future

    champion = XGBClassifier()
    champion.load_model(PRODUCTION_MODEL_PATH)

    print("Evaluating champion...")
    champion_metrics = evaluate_model(
        champion,
        X_future,
        y_future,
    )

    print("Evaluating challenger...")
    candidate_metrics = evaluate_model(
        candidate,
        X_future,
        y_future,
    )

    promotion = evaluate_candidate_promotion(
        champion_metrics,
        candidate_metrics,
    )

    CANDIDATE_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    candidate.save_model(CANDIDATE_MODEL_PATH)

    report = {
        "trigger": trigger.to_dict(),
        "candidate_cutoff_step":
            CANDIDATE_CUTOFF_STEP,
        "training_window":
            f"step < {CANDIDATE_CUTOFF_STEP}",
        "evaluation_window":
            f"step >= {CANDIDATE_CUTOFF_STEP}",
        "decision_threshold":
            DECISION_THRESHOLD,
        "champion_metrics":
            champion_metrics,
        "candidate_metrics":
            candidate_metrics,
        "promotion":
            promotion.to_dict(),
        "production_model":
            str(PRODUCTION_MODEL_PATH),
        "candidate_model":
            str(CANDIDATE_MODEL_PATH),
    }

    LIFECYCLE_REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    LIFECYCLE_REPORT_PATH.write_text(
        json.dumps(report, indent=2)
    )

    print()
    print(json.dumps(report, indent=2))

    con.close()


if __name__ == "__main__":
    main()

