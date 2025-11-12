from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd


def format_metrics_table(list_of_metric_dicts: List[Dict]) -> pd.DataFrame:
    """
    Convert a list of metric dictionaries (one per model/run)
    into a pandas DataFrame.
    """

    return pd.DataFrame(list_of_metric_dicts)


def save_metrics(df: pd.DataFrame, path: str) -> None:
    """
    Save metrics DataFrame to CSV, creating parent directories if needed.
    """

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)


def print_metrics(df: pd.DataFrame, decimals: int = 3) -> None:
    """
    Print a nicely formatted version of the DataFrame.
    """

    rounded = df.copy()
    numeric_cols = rounded.select_dtypes(include=["number"]).columns
    rounded[numeric_cols] = rounded[numeric_cols].round(decimals)
    print(rounded)
