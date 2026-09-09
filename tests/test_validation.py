import pandas as pd
import pytest

from fraud_ml.validation import (
    EXPECTED_COLUMNS,
    count_duplicate_rows,
    validate_schema,
    validate_target,
)


def make_valid_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "step": [1],
            "type": ["TRANSFER"],
            "amount": [100.0],
            "nameOrig": ["C1"],
            "oldbalanceOrg": [1000.0],
            "newbalanceOrig": [900.0],
            "nameDest": ["C2"],
            "oldbalanceDest": [500.0],
            "newbalanceDest": [600.0],
            "isFraud": [0],
            "isFlaggedFraud": [0],
        }
    )


def test_schema_valid():
    df = make_valid_dataframe()
    validate_schema(df)


def test_schema_missing_column():
    df = make_valid_dataframe().drop(columns=["amount"])

    with pytest.raises(ValueError):
        validate_schema(df)


def test_target_is_binary():
    df = pd.DataFrame({"isFraud": [0, 1, 0]})
    validate_target(df)


def test_target_invalid_value():
    df = pd.DataFrame({"isFraud": [0, 1, 2]})

    with pytest.raises(ValueError):
        validate_target(df)


def test_duplicate_count():
    df = make_valid_dataframe()
    duplicated = pd.concat([df, df], ignore_index=True)

    assert count_duplicate_rows(duplicated) == 1


def test_expected_columns_are_defined():
    assert "isFraud" in EXPECTED_COLUMNS
