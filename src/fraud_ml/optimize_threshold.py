from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from fraud_ml.thresholding import evaluate_threshold


FEATURE_PATH = Path(
    "data/processed/behavioral_features.parquet"
)

MODEL_PATH = Path(
    "models/behavioral_xgboost.json"
)

REPORT_PATH = Path(
    "reports/metrics/threshold_analysis.csv"
)

FALSE_POSITIVE_COST = 1.0
FALSE_NEGATIVE_COST = 100.0


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


def main() -> None:
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"Missing feature store: {FEATURE_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing trained model: {MODEL_PATH}"
        )

    con = duckdb.connect()

    con.execute("SET memory_limit = '2GB'")
    con.execute("SET threads = 2")

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

    print(f"Temporal cutoff step: {cutoff_step}")
    print("Loading test partition...")

    test = con.execute(
        f"""
        SELECT
            isFraud,
            {FEATURE_SQL}
        FROM read_parquet(
            '{FEATURE_PATH.as_posix()}'
        )
        WHERE step >= {cutoff_step}
        """
    ).fetchdf()

    y_test = test.pop("isFraud").to_numpy()
    X_test = test

    model = XGBClassifier()
    model.load_model(MODEL_PATH)

    print("Generating fraud probabilities...")

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    print("Evaluating thresholds...")

    thresholds = np.arange(
        0.01,
        1.00,
        0.01,
    )

    results = []

    for threshold in thresholds:
        results.append(
            evaluate_threshold(
                y_true=y_test,
                probabilities=probabilities,
                threshold=float(threshold),
                false_positive_cost=FALSE_POSITIVE_COST,
                false_negative_cost=FALSE_NEGATIVE_COST,
            )
        )

    results_df = pd.DataFrame(results)

    best_f1 = results_df.loc[
        results_df["f1"].idxmax()
    ]

    best_cost = results_df.loc[
        results_df[
            "estimated_business_cost"
        ].idxmin()
    ]

    baseline = results_df.iloc[
        (
            results_df["threshold"] - 0.50
        ).abs().argsort()[:1]
    ].iloc[0]

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        REPORT_PATH,
        index=False,
    )

    print()
    print("=" * 70)
    print("DEFAULT THRESHOLD 0.50")
    print("=" * 70)
    print(f"Precision: {baseline['precision']:.6f}")
    print(f"Recall:    {baseline['recall']:.6f}")
    print(f"F1:        {baseline['f1']:.6f}")
    print(f"FP:        {int(baseline['fp']):,}")
    print(f"FN:        {int(baseline['fn']):,}")

    print()
    print("=" * 70)
    print("BEST F1 THRESHOLD")
    print("=" * 70)
    print(f"Threshold: {best_f1['threshold']:.2f}")
    print(f"Precision: {best_f1['precision']:.6f}")
    print(f"Recall:    {best_f1['recall']:.6f}")
    print(f"F1:        {best_f1['f1']:.6f}")
    print(f"FP:        {int(best_f1['fp']):,}")
    print(f"FN:        {int(best_f1['fn']):,}")

    print()
    print("=" * 70)
    print("LOWEST RELATIVE BUSINESS COST")
    print("=" * 70)
    print(f"Threshold: {best_cost['threshold']:.2f}")
    print(f"Precision: {best_cost['precision']:.6f}")
    print(f"Recall:    {best_cost['recall']:.6f}")
    print(f"F1:        {best_cost['f1']:.6f}")
    print(f"FP:        {int(best_cost['fp']):,}")
    print(f"FN:        {int(best_cost['fn']):,}")
    print(
        "False-positive cost: "
        f"{best_cost['false_positive_cost']:,.0f} units"
    )
    print(
        "False-negative cost: "
        f"{best_cost['false_negative_cost']:,.0f} units"
    )
    print(
        "Estimated total cost: "
        f"{best_cost['estimated_business_cost']:,.0f} units"
    )

    print()
    print(
        f"Full threshold report saved to: "
        f"{REPORT_PATH}"
    )

    print()
    print(
        "NOTE: Business-cost values are illustrative "
        "relative cost units, not currency."
    )
    print(
        "Assumption: false negative = 100 units, "
        "false positive = 1 unit."
    )

    con.close()


if __name__ == "__main__":
    main()
