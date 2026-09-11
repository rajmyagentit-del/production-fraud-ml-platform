import numpy as np
import pandas as pd


def create_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create point-in-time behavioral features.

    Each row only uses information from earlier transactions.
    The dataframe is sorted by step using stable ordering.
    """
    data = df.sort_values("step", kind="stable").copy()

    origin_group = data.groupby("nameOrig", sort=False)

    data["orig_prior_txn_count"] = origin_group.cumcount()

    prior_amount_sum = (
        origin_group["amount"]
        .cumsum()
        - data["amount"]
    )

    data["orig_prior_amount_sum"] = prior_amount_sum

    data["orig_prior_amount_mean"] = np.where(
        data["orig_prior_txn_count"] > 0,
        data["orig_prior_amount_sum"]
        / data["orig_prior_txn_count"],
        0.0,
    )

    prior_step = origin_group["step"].shift(1)

    data["orig_steps_since_prev_txn"] = (
        data["step"] - prior_step
    ).fillna(-1)

    first_step = origin_group["step"].transform("min")

    data["orig_account_age_steps"] = (
        data["step"] - first_step
    )

    data["amount_vs_prior_mean"] = np.where(
        data["orig_prior_amount_mean"] > 0,
        data["amount"]
        / data["orig_prior_amount_mean"],
        0.0,
    )

    pair_group = data.groupby(
        ["nameOrig", "nameDest"],
        sort=False,
    )

    data["orig_dest_prior_txn_count"] = pair_group.cumcount()

    data["dest_seen_before"] = (
        data["orig_dest_prior_txn_count"] > 0
    ).astype(int)

    return data
