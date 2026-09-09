from pathlib import Path

import pandas as pd


DEFAULT_DATA_PATH = Path("data/raw/transactions.csv")


def load_transactions(
    path: str | Path = DEFAULT_DATA_PATH,
) -> pd.DataFrame:
    """
    Load PaySim transaction data from CSV.

    Parameters
    ----------
    path:
        Path to the PaySim transaction CSV.

    Returns
    -------
    pd.DataFrame
        Raw transaction dataframe.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Place the PaySim CSV at data/raw/transactions.csv."
        )

    return pd.read_csv(path)
