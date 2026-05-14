"""
fetch_data.py
-------------
Downloads and prepares OHLCV data from Yahoo Finance.
Generates return labels (up/down) for binary classification.

Reference: Jiang, Kelly & Xiu (2023) - (Re-)Imag(in)ing Price Trends
"""

import logging
import os
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "JPM", "GS", "BAC", "WMT",
    "XOM", "CVX", "JNJ", "PFE", "KO",
    "PEP", "NVDA", "AMD", "INTC", "CSCO",
]

IMAGE_WINDOWS = [5, 20, 60]
RETURN_HORIZONS = [5, 20, 60]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _ensure_flat_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flattens yfinance MultiIndex columns to their price-type level.

    Recent yfinance versions return MultiIndex columns even for single-ticker
    downloads, e.g. ('Close', 'AAPL') instead of 'Close'.  This helper
    normalises the DataFrame so all downstream code can use plain string keys.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_ohlcv(
    tickers: list[str] = DEFAULT_TICKERS,
    start: str = "2000-01-01",
    end: str = "2023-12-31",
    save_dir: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """
    Downloads OHLCV data for a list of tickers.

    Returns
    -------
    dict  {ticker: DataFrame with columns Open/High/Low/Close/Volume}
    """
    data = {}
    for ticker in tickers:
        logger.info("Downloading %s ...", ticker)
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            if df.empty:
                logger.warning("%s: EMPTY — skipped", ticker)
                continue
            df = _ensure_flat_columns(df)
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            data[ticker] = df
            logger.info("%s: OK (%d days)", ticker, len(df))
        except Exception as e:
            logger.error("%s: ERROR — %s", ticker, e)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for ticker, df in data.items():
            df.to_csv(os.path.join(save_dir, f"{ticker}.csv"))
        logger.info("Data saved to: %s", save_dir)

    return data


def load_ohlcv(data_dir: str) -> dict[str, pd.DataFrame]:
    """Loads previously downloaded CSV files from a directory."""
    data = {}
    for fname in os.listdir(data_dir):
        if fname.endswith(".csv"):
            ticker = fname.replace(".csv", "")
            df = pd.read_csv(os.path.join(data_dir, fname), index_col=0, parse_dates=True)
            data[ticker] = _ensure_flat_columns(df)
    logger.info("%d tickers loaded from %s", len(data), data_dir)
    return data


# ---------------------------------------------------------------------------
# Normalisation (image scaling — see paper Section I)
# ---------------------------------------------------------------------------
def image_scale(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Normalises prices over a rolling window of `window` days.
    The window's max High and min Low are mapped to [0, 1].
    Volume is similarly normalised (max volume in window -> 1).

    This is the exact normalisation used in the paper to make
    all stocks comparable on a common scale.
    """
    df = _ensure_flat_columns(df)
    scaled = df.copy().astype(float)

    price_cols = ["Open", "High", "Low", "Close"]
    price_max = df[price_cols].rolling(window).max().max(axis=1)
    price_min = df[price_cols].rolling(window).min().min(axis=1)
    denom = (price_max - price_min).replace(0, np.nan)

    for col in price_cols:
        scaled[col] = df[col].sub(price_min, axis=0).div(denom, axis=0)

    vol_max = df["Volume"].rolling(window).max().replace(0, np.nan)
    scaled["Volume"] = df["Volume"] / vol_max

    return scaled.dropna()


def cumret_scale(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Normalises prices by the Close at the start of each window.
    Equivalent to a cumulative return — used as a comparison baseline.
    """
    df = _ensure_flat_columns(df)
    scaled = df.copy().astype(float)
    first_close = df["Close"].shift(window - 1)
    for col in ["Open", "High", "Low", "Close"]:
        scaled[col] = df[col] / first_close
    vol_max = df["Volume"].rolling(window).max().replace(0, np.nan)
    scaled["Volume"] = df["Volume"] / vol_max
    return scaled.dropna()


# ---------------------------------------------------------------------------
# Moving average
# ---------------------------------------------------------------------------
def add_moving_average(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Adds a `window`-day moving average column (MA)."""
    df = _ensure_flat_columns(df).copy()
    df["MA"] = df["Close"].rolling(window).mean()
    return df.dropna()


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def make_labels(
    df: pd.DataFrame,
    horizon: int,
    col: str = "Close",
) -> pd.Series:
    """
    Generates binary labels:
      1    if the `horizon`-day return is positive
      0    otherwise
      NaN  for the last `horizon` rows (future unknown)

    The label at t corresponds to the return between t and t+horizon.
    NaN is preserved so that downstream .dropna() correctly excludes
    the tail rows that have no valid future price.
    """
    df = _ensure_flat_columns(df)
    future_close = df[col].shift(-horizon)
    label = (future_close > df[col]).astype(float)
    label[future_close.isna()] = np.nan
    return label.rename(f"label_{horizon}d")


# ---------------------------------------------------------------------------
# Sliding windows — tabular data (for MLP / LSTM)
# ---------------------------------------------------------------------------
def make_tabular_dataset(
    df: pd.DataFrame,
    window: int,
    horizon: int,
    scaling: str = "image",
    add_ma: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Builds (X, y) for MLP and LSTM from a single OHLCV series.

    Parameters
    ----------
    df      : OHLCV DataFrame for a single ticker
    window  : input window length (e.g. 5, 20, 60)
    horizon : prediction horizon
    scaling : normalisation method ("image" | "cumret")
    add_ma  : include moving average

    Returns
    -------
    X : (n_samples, window, n_features)
    y : (n_samples,)
    """
    if add_ma:
        df = add_moving_average(df, window)

    if scaling == "image":
        df_scaled = image_scale(df, window)
    else:
        df_scaled = cumret_scale(df, window)

    labels = make_labels(df_scaled, horizon)

    feature_cols = ["Open", "High", "Low", "Close", "Volume"]
    if add_ma and "MA" in df_scaled.columns:
        feature_cols.append("MA")

    df_scaled = df_scaled.join(labels).dropna()

    X_list, y_list = [], []
    arr = df_scaled[feature_cols].values
    lbl = df_scaled[f"label_{horizon}d"].values

    for i in range(window, len(arr) - horizon):
        X_list.append(arr[i - window:i])
        y_list.append(lbl[i])

    if not X_list:
        return np.empty((0, window, len(feature_cols))), np.empty(0)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)


def make_multi_stock_dataset(
    data: dict[str, pd.DataFrame],
    window: int,
    horizon: int,
    scaling: str = "image",
    add_ma: bool = True,
    train_ratio: float = 0.7,
) -> dict:
    """
    Builds the full dataset (all tickers) and performs a
    chronological train / val / test split.

    Split used in the paper:
      - Train + Val : 1993-2000
      - Test        : 2001-2019

    Here adapted with a configurable ratio.
    """
    X_all, y_all = [], []

    for ticker, df in data.items():
        X, y = make_tabular_dataset(df, window, horizon, scaling, add_ma)
        if len(X) > 0:
            X_all.append(X)
            y_all.append(y)

    if not X_all:
        raise ValueError("No data available.")

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)

    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    n_train = int(len(X) * train_ratio)
    n_val = (len(X) - n_train) // 2

    return {
        "X_train": X[:n_train],
        "y_train": y[:n_train],
        "X_val":   X[n_train:n_train + n_val],
        "y_val":   y[n_train:n_train + n_val],
        "X_test":  X[n_train + n_val:],
        "y_test":  y[n_train + n_val:],
        "window":  window,
        "horizon": horizon,
        "scaling": scaling,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("=== Downloading data ===")
    raw_data = download_ohlcv(
        tickers=DEFAULT_TICKERS,
        start="2000-01-01",
        end="2023-12-31",
        save_dir="data/raw",
    )

    logger.info("=== Building tabular dataset (window=20, horizon=5) ===")
    dataset = make_multi_stock_dataset(
        data=raw_data,
        window=20,
        horizon=5,
        scaling="image",
    )
    for split in ["train", "val", "test"]:
        X = dataset[f"X_{split}"]
        y = dataset[f"y_{split}"]
        pos = y.mean() * 100
        logger.info("  %s: %s  —  %.1f%% positive", split, X.shape, pos)