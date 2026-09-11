import pandas as pd

from fraud_ml.behavioral_features import create_behavioral_features


def test_behavioral_features_use_only_prior_rows():
    df = pd.DataFrame(
        {
            "step": [1, 2, 3],
            "nameOrig": ["A", "A", "A"],
            "nameDest": ["X", "Y", "X"],
            "amount": [100.0, 200.0, 300.0],
        }
    )

    result = create_behavioral_features(df)

    assert result["orig_prior_txn_count"].tolist() == [0, 1, 2]
    assert result["orig_prior_amount_sum"].tolist() == [
        0.0,
        100.0,
        300.0,
    ]

    assert result["orig_prior_amount_mean"].tolist() == [
        0.0,
        100.0,
        150.0,
    ]

    assert result["orig_steps_since_prev_txn"].tolist() == [
        -1.0,
        1.0,
        1.0,
    ]


def test_destination_history():
    df = pd.DataFrame(
        {
            "step": [1, 2, 3],
            "nameOrig": ["A", "A", "A"],
            "nameDest": ["X", "Y", "X"],
            "amount": [10.0, 20.0, 30.0],
        }
    )

    result = create_behavioral_features(df)

    assert result["orig_dest_prior_txn_count"].tolist() == [
        0,
        0,
        1,
    ]

    assert result["dest_seen_before"].tolist() == [
        0,
        0,
        1,
    ]


def test_account_age():
    df = pd.DataFrame(
        {
            "step": [5, 7, 10],
            "nameOrig": ["A", "A", "A"],
            "nameDest": ["X", "Y", "Z"],
            "amount": [10.0, 20.0, 30.0],
        }
    )

    result = create_behavioral_features(df)

    assert result["orig_account_age_steps"].tolist() == [
        0,
        2,
        5,
    ]
