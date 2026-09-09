import numpy as np
import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create transaction-level fraud detection features.
    """
    data = df.copy()

    data["amount_log"] = np.log1p(data["amount"])

    data["orig_balance_delta"] = (
        data["oldbalanceOrg"] - data["newbalanceOrig"]
    )

    data["dest_balance_delta"] = (
        data["newbalanceDest"] - data["oldbalanceDest"]
    )

    data["orig_balance_error"] = (
        data["oldbalanceOrg"]
        - data["amount"]
        - data["newbalanceOrig"]
    )

    data["dest_balance_error"] = (
        data["oldbalanceDest"]
        + data["amount"]
        - data["newbalanceDest"]
    )

    data["amount_to_orig_balance"] = (
        data["amount"] / (data["oldbalanceOrg"] + 1.0)
    )

    data["hour"] = data["step"] % 24
    data["day"] = data["step"] // 24

    return data
