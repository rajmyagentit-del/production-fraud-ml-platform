import pandas as pd


def strict_time_split(
    df: pd.DataFrame,
    train_fraction: float = 0.8,
    time_column: str = "step",
):
    """
    Split transactions chronologically while preventing the same
    time value from appearing in both train and test.

    The cutoff is chosen near the requested fraction of rows.
    Every row at the cutoff time is assigned to the test set.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")

    if time_column not in df.columns:
        raise ValueError(f"Missing time column: {time_column}")

    if df.empty:
        raise ValueError("Cannot split an empty dataframe")

    ordered = df.sort_values(
        time_column,
        kind="stable",
    ).reset_index(drop=True)

    split_index = int(len(ordered) * train_fraction)

    if split_index <= 0 or split_index >= len(ordered):
        raise ValueError("Invalid split index")

    cutoff_time = ordered.iloc[split_index][time_column]

    train_df = ordered[
        ordered[time_column] < cutoff_time
    ].copy()

    test_df = ordered[
        ordered[time_column] >= cutoff_time
    ].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Temporal split produced an empty train or test set"
        )

    if (
        train_df[time_column].max()
        >= test_df[time_column].min()
    ):
        raise ValueError(
            "Temporal leakage detected between train and test"
        )

    return train_df, test_df, cutoff_time
