import pandas as pd


EXPECTED_COLUMNS = {
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
}


def validate_schema(df: pd.DataFrame) -> None:
    """Validate the expected PaySim schema."""
    missing = EXPECTED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )


def validate_target(df: pd.DataFrame) -> None:
    """Validate that the fraud target contains only 0/1 values."""
    values = set(df["isFraud"].dropna().unique())

    if not values.issubset({0, 1}):
        raise ValueError(
            f"isFraud contains invalid values: {sorted(values)}"
        )


def count_duplicate_rows(df: pd.DataFrame) -> int:
    """Return the number of duplicated rows."""
    return int(df.duplicated().sum())
