import pandas as pd

from fraud_ml.features import create_features


def test_create_features():
    df = pd.DataFrame(
        {
            "step": [25],
            "amount": [100.0],
            "oldbalanceOrg": [1000.0],
            "newbalanceOrig": [900.0],
            "oldbalanceDest": [500.0],
            "newbalanceDest": [600.0],
        }
    )

    result = create_features(df)

    expected_features = {
        "amount_log",
        "orig_balance_delta",
        "dest_balance_delta",
        "orig_balance_error",
        "dest_balance_error",
        "amount_to_orig_balance",
        "hour",
        "day",
    }

    assert expected_features.issubset(result.columns)
    assert result.loc[0, "hour"] == 1
    assert result.loc[0, "day"] == 1
