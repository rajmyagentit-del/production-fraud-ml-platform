from pathlib import Path

import duckdb


INPUT_PATH = Path("data/raw/transactions.csv")
OUTPUT_PATH = Path("data/processed/behavioral_features.parquet")
TEMP_PATH = Path("data/processed/duckdb_tmp")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"PaySim dataset not found at {INPUT_PATH}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEMP_PATH.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    con.execute("SET memory_limit = '2GB'")
    con.execute(
        f"SET temp_directory = '{TEMP_PATH.as_posix()}'"
    )
    con.execute("SET threads = 2")

    print("Building point-in-time behavioral feature store...")
    print("This may take several minutes.")

    query = f"""
    COPY (
        WITH transactions AS (
            SELECT
                step::INTEGER AS step,
                type,
                amount::DOUBLE AS amount,
                nameOrig,
                oldbalanceOrg::DOUBLE AS oldbalanceOrg,
                nameDest,
                oldbalanceDest::DOUBLE AS oldbalanceDest,
                isFraud::INTEGER AS isFraud
            FROM read_csv_auto(
                '{INPUT_PATH.as_posix()}',
                header = true
            )
        ),

        origin_step AS (
            SELECT
                step,
                nameOrig,
                COUNT(*) AS step_txn_count,
                SUM(amount) AS step_amount_sum
            FROM transactions
            GROUP BY step, nameOrig
        ),

        origin_history AS (
            SELECT
                step,
                nameOrig,

                COALESCE(
                    SUM(step_txn_count) OVER (
                        PARTITION BY nameOrig
                        ORDER BY step
                        ROWS BETWEEN UNBOUNDED PRECEDING
                        AND 1 PRECEDING
                    ),
                    0
                ) AS orig_prior_txn_count,

                COALESCE(
                    SUM(step_amount_sum) OVER (
                        PARTITION BY nameOrig
                        ORDER BY step
                        ROWS BETWEEN UNBOUNDED PRECEDING
                        AND 1 PRECEDING
                    ),
                    0.0
                ) AS orig_prior_amount_sum,

                LAG(step) OVER (
                    PARTITION BY nameOrig
                    ORDER BY step
                ) AS orig_previous_step,

                MIN(step) OVER (
                    PARTITION BY nameOrig
                    ORDER BY step
                    ROWS BETWEEN UNBOUNDED PRECEDING
                    AND CURRENT ROW
                ) AS orig_first_step

            FROM origin_step
        ),

        origin_dest_step AS (
            SELECT
                step,
                nameOrig,
                nameDest,
                COUNT(*) AS pair_step_txn_count
            FROM transactions
            GROUP BY step, nameOrig, nameDest
        ),

        origin_dest_history AS (
            SELECT
                step,
                nameOrig,
                nameDest,

                COALESCE(
                    SUM(pair_step_txn_count) OVER (
                        PARTITION BY nameOrig, nameDest
                        ORDER BY step
                        ROWS BETWEEN UNBOUNDED PRECEDING
                        AND 1 PRECEDING
                    ),
                    0
                ) AS orig_dest_prior_txn_count

            FROM origin_dest_step
        )

        SELECT
            t.step,
            t.type,
            t.amount,
            t.oldbalanceOrg,
            t.oldbalanceDest,
            t.isFraud,

            h.orig_prior_txn_count,
            h.orig_prior_amount_sum,

            CASE
                WHEN h.orig_prior_txn_count > 0
                THEN
                    h.orig_prior_amount_sum
                    / h.orig_prior_txn_count
                ELSE 0.0
            END AS orig_prior_amount_mean,

            CASE
                WHEN h.orig_previous_step IS NULL
                THEN -1
                ELSE t.step - h.orig_previous_step
            END AS orig_steps_since_prev_txn,

            t.step - h.orig_first_step
                AS orig_account_age_steps,

            CASE
                WHEN h.orig_prior_txn_count > 0
                     AND h.orig_prior_amount_sum > 0
                THEN
                    t.amount
                    /
                    (
                        h.orig_prior_amount_sum
                        / h.orig_prior_txn_count
                    )
                ELSE 0.0
            END AS amount_vs_prior_mean,

            p.orig_dest_prior_txn_count,

            CASE
                WHEN p.orig_dest_prior_txn_count > 0
                THEN 1
                ELSE 0
            END AS dest_seen_before

        FROM transactions t

        LEFT JOIN origin_history h
            ON t.step = h.step
            AND t.nameOrig = h.nameOrig

        LEFT JOIN origin_dest_history p
            ON t.step = p.step
            AND t.nameOrig = p.nameOrig
            AND t.nameDest = p.nameDest

        ORDER BY t.step
    )
    TO '{OUTPUT_PATH.as_posix()}'
    (
        FORMAT PARQUET,
        COMPRESSION ZSTD
    )
    """

    con.execute(query)

    row_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{OUTPUT_PATH.as_posix()}')
        """
    ).fetchone()[0]

    min_step, max_step = con.execute(
        f"""
        SELECT MIN(step), MAX(step)
        FROM read_parquet('{OUTPUT_PATH.as_posix()}')
        """
    ).fetchone()

    print()
    print("=" * 60)
    print("BEHAVIORAL FEATURE STORE COMPLETE")
    print("=" * 60)
    print(f"Rows:       {row_count:,}")
    print(f"Step range: {min_step} -> {max_step}")
    print(f"Output:     {OUTPUT_PATH}")
    print()
    print(
        "Features use only activity from PRIOR steps, "
        "not transactions from the same step."
    )

    con.close()


if __name__ == "__main__":
    main()
