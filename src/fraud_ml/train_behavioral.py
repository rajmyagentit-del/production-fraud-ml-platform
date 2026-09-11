from pathlib import Path
import gc

import duckdb
import mlflow
import mlflow.xgboost
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


FEATURE_PATH = Path(
    "data/processed/behavioral_features.parquet"
)

MODEL_PATH = Path(
    "models/behavioral_xgboost.json"
)


FEATURE_SQL = """
    CAST(step AS FLOAT) AS step,
    CAST(amount AS FLOAT) AS amount,
    CAST(oldbalanceOrg AS FLOAT) AS oldbalanceOrg,
    CAST(oldbalanceDest AS FLOAT) AS oldbalanceDest,

    CAST(step % 24 AS FLOAT) AS hour,
    CAST(FLOOR(step / 24) AS FLOAT) AS day,

    CAST(
        amount / (oldbalanceOrg + 1.0)
        AS FLOAT
    ) AS amount_to_orig_balance,

    CAST(
        orig_prior_txn_count
        AS FLOAT
    ) AS orig_prior_txn_count,

    CAST(
        orig_prior_amount_mean
        AS FLOAT
    ) AS orig_prior_amount_mean,

    CAST(
        orig_steps_since_prev_txn
        AS FLOAT
    ) AS orig_steps_since_prev_txn,

    CAST(
        orig_account_age_steps
        AS FLOAT
    ) AS orig_account_age_steps,

    CAST(
        amount_vs_prior_mean
        AS FLOAT
    ) AS amount_vs_prior_mean,

    CAST(
        orig_dest_prior_txn_count
        AS FLOAT
    ) AS orig_dest_prior_txn_count,

    CAST(
        dest_seen_before
        AS FLOAT
    ) AS dest_seen_before,

    CAST(type = 'CASH_IN' AS INTEGER)
        AS type_CASH_IN,

    CAST(type = 'CASH_OUT' AS INTEGER)
        AS type_CASH_OUT,

    CAST(type = 'DEBIT' AS INTEGER)
        AS type_DEBIT,

    CAST(type = 'PAYMENT' AS INTEGER)
        AS type_PAYMENT,

    CAST(type = 'TRANSFER' AS INTEGER)
        AS type_TRANSFER
"""


def get_cutoff_step(con: duckdb.DuckDBPyConnection) -> int:
    row_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{FEATURE_PATH.as_posix()}')
        """
    ).fetchone()[0]

    split_index = int(row_count * 0.8)

    cutoff_step = con.execute(
        f"""
        SELECT step
        FROM read_parquet('{FEATURE_PATH.as_posix()}')
        ORDER BY step
        LIMIT 1
        OFFSET {split_index}
        """
    ).fetchone()[0]

    return int(cutoff_step)


def load_partition(
    con: duckdb.DuckDBPyConnection,
    where_clause: str,
):
    return con.execute(
        f"""
        SELECT
            isFraud,
            {FEATURE_SQL}
        FROM read_parquet(
            '{FEATURE_PATH.as_posix()}'
        )
        WHERE {where_clause}
        """
    ).fetchdf()


def main() -> None:
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            "Behavioral feature store not found. "
            "Run build_behavioral_features first."
        )

    con = duckdb.connect()

    con.execute("SET memory_limit = '2GB'")
    con.execute("SET threads = 2")

    cutoff_step = get_cutoff_step(con)

    print("=" * 60)
    print("BEHAVIORAL FRAUD MODEL")
    print("=" * 60)
    print(f"Temporal cutoff step: {cutoff_step}")
    print()

    print("Loading training partition...")

    train = load_partition(
        con,
        f"step < {cutoff_step}",
    )

    y_train = train.pop("isFraud")
    X_train = train

    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)

    scale_pos_weight = (
        negatives / max(positives, 1)
    )

    print(f"Training rows:  {len(X_train):,}")
    print(f"Training fraud: {positives:,}")
    print(
        f"scale_pos_weight: "
        f"{scale_pos_weight:.2f}"
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        tree_method="hist",
        random_state=42,
        n_jobs=2,
    )

    print("\nTraining behavioral XGBoost...")

    model.fit(X_train, y_train)

    del train
    del X_train
    del y_train
    gc.collect()

    print("Training complete.")
    print("Loading test partition...")

    test = load_partition(
        con,
        f"step >= {cutoff_step}",
    )

    y_test = test.pop("isFraud")
    X_test = test

    print(f"Testing rows:   {len(X_test):,}")
    print(f"Testing fraud:  {int(y_test.sum()):,}")

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    false_positive_rate = (
        fp / max(fp + tn, 1)
    )

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(
        f"ROC-AUC:             "
        f"{roc_auc:.6f}"
    )
    print(
        f"PR-AUC:              "
        f"{pr_auc:.6f}"
    )
    print(
        f"Precision:           "
        f"{precision:.6f}"
    )
    print(
        f"Recall:              "
        f"{recall:.6f}"
    )
    print(
        f"F1:                  "
        f"{f1:.6f}"
    )
    print(
        f"False Positive Rate: "
        f"{false_positive_rate:.6f}"
    )

    print()
    print("Confusion Matrix")
    print(f"TN: {tn:,}")
    print(f"FP: {fp:,}")
    print(f"FN: {fn:,}")
    print(f"TP: {tp:,}")

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_model(MODEL_PATH)

    mlflow.set_experiment(
        "production-fraud-ml-platform"
    )

    with mlflow.start_run(
        run_name="xgboost_behavioral_disk_backed"
    ):
        mlflow.log_params(
            {
                "model": "XGBClassifier",
                "behavioral_features": True,
                "point_in_time_safe": True,
                "same_step_history_used": False,
                "post_transaction_balances": False,
                "isFlaggedFraud_used": False,
                "split_strategy":
                    "strict_chronological",
                "cutoff_step": cutoff_step,
                "threshold": 0.5,
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
                "scale_pos_weight":
                    scale_pos_weight,
            }
        )

        mlflow.log_metrics(
            {
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_positive_rate":
                    false_positive_rate,
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
                "true_negatives": int(tn),
            }
        )

        mlflow.xgboost.log_model(
            model,
            name="behavioral_fraud_model",
        )

    con.close()

    print()
    print(
        f"Model saved: {MODEL_PATH}"
    )
    print(
        "Behavioral MLflow run logged."
    )


if __name__ == "__main__":
    main()
