import json
from pathlib import Path

from fraud_ml.data import load_transactions
from fraud_ml.validation import (
    count_duplicate_rows,
    validate_schema,
    validate_target,
)


DATA_PATH = Path("data/raw/transactions.csv")
REPORT_PATH = Path("reports/metrics/data_quality.json")


def main() -> None:
    print("Loading PaySim dataset...")
    df = load_transactions(DATA_PATH)

    print("Validating schema...")
    validate_schema(df)
    validate_target(df)

    fraud_count = int(df["isFraud"].sum())
    total_transactions = len(df)
    legitimate_count = total_transactions - fraud_count
    fraud_rate = fraud_count / total_transactions
    duplicates = count_duplicate_rows(df)

    report = {
        "rows": total_transactions,
        "columns": len(df.columns),
        "fraud_transactions": fraud_count,
        "legitimate_transactions": legitimate_count,
        "fraud_rate": fraud_rate,
        "duplicate_rows": duplicates,
        "missing_values": {
            column: int(value)
            for column, value in df.isna().sum().items()
        },
        "transaction_types": {
            str(key): int(value)
            for key, value in df["type"].value_counts().items()
        },
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w") as file:
        json.dump(report, file, indent=2)

    print("\n=== PAYSim DATA QUALITY REPORT ===")
    print(f"Rows:                  {total_transactions:,}")
    print(f"Columns:               {len(df.columns)}")
    print(f"Fraud transactions:    {fraud_count:,}")
    print(f"Legitimate transactions: {legitimate_count:,}")
    print(f"Fraud rate:            {fraud_rate:.6%}")
    print(f"Duplicate rows:        {duplicates:,}")
    print("\nTransaction types:")
    print(df["type"].value_counts())
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
