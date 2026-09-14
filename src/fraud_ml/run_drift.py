from __future__ import annotations

import json
from pathlib import Path

import duckdb

from fraud_ml.drift import build_drift_report, save_report


FEATURE_PATH = Path("data/processed/behavioral_features.parquet")
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
    columns = ["step", "type", *NUMERIC_FEATURES]
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

    connection.close()

    print(f"Reference rows: {len(reference):,}")
    print(f"Current rows: {len(current):,}")

    report = build_drift_report(
        reference=reference,
        current=current,
        numeric_features=NUMERIC_FEATURES,
        categorical_feature="type",
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
