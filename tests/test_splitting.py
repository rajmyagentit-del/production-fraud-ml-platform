import pandas as pd
import pytest

from fraud_ml.splitting import strict_time_split


def test_strict_time_split_has_no_overlap():
    df = pd.DataFrame(
        {
            "step": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "value": range(10),
        }
    )

    train_df, test_df, cutoff = strict_time_split(
        df,
        train_fraction=0.6,
    )

    assert train_df["step"].max() < test_df["step"].min()
    assert cutoff == test_df["step"].min()


def test_strict_time_split_preserves_all_rows():
    df = pd.DataFrame(
        {
            "step": [1, 1, 2, 2, 3, 3, 4, 4],
        }
    )

    train_df, test_df, _ = strict_time_split(df)

    assert len(train_df) + len(test_df) == len(df)


def test_invalid_fraction():
    df = pd.DataFrame({"step": [1, 2, 3]})

    with pytest.raises(ValueError):
        strict_time_split(df, train_fraction=1.0)


def test_missing_time_column():
    df = pd.DataFrame({"amount": [10, 20]})

    with pytest.raises(ValueError):
        strict_time_split(df)
