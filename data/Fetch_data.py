"""
fetch_data.py
-------------
Downloads and prepares OHLCV data from Yahoo Finance.
Generates binary return labels (up/down) for classification.

Key fixes over the initial version:
  1. Strict temporal split — no past/future mixing
  2. Window-level normalization — no rolling over the full series
  3. Shuffle applied only to the training set, never val/test
  4. Clean MA integration in the feature vector

Reference: Jiang, Kelly & Xiu (2023) - (Re-)Imag(in)ing Price Trends
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "JPM", "GS", "BAC", "WMT",
    "XOM", "CVX", "JNJ", "PFE", "KO",
    "PEP", "NVDA", "AMD", "INTC", "CSCO",
]

IMAGE_WINDOWS   = [5, 20, 60]   # image window lengths (days)
RETURN_HORIZONS = [5, 20, 60]   # prediction horizons (days)

# Default temporal splits (adjustable depending on data availability)
DEFAULT_TRAIN_END = "2014-12-31"
DEFAULT_VAL_END   = "2018-12-31"
# Test: everything after val_end


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_ohlcv(
    tickers:  list[str] = DEFAULT_TICKERS,
    start:    str = "2000-01-01",
    end:      str = "2023-12-31",
    save_dir: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """
    Downloads OHLCV data for a list of tickers from Yahoo Finance.

    Returns
    -------
    dict  {ticker: DataFrame with columns Open/High/Low/Close/Volume}
          Index is a DatetimeIndex sorted chronologically.
    """
    data = {}
    for ticker in tickers:
        print(f"  Downloading : {ticker} ...", end=" ")
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
            )
            if df.empty:
                print("EMPTY — skipped")
                continue

            # Handle MultiIndex returned by some yfinance versions
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            df = df.sort_index()
            data[ticker] = df
            print(f"OK ({len(df)} days)")
        except Exception as e:
            print(f"ERROR : {e}")

    if not data:
        raise ValueError("No ticker was successfully downloaded.")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for ticker, df in data.items():
            df.to_csv(os.path.join(save_dir, f"{ticker}.csv"))
        print(f"\nData saved to : {save_dir}")

    return data


def load_ohlcv(data_dir: str) -> dict[str, pd.DataFrame]:
    """Loads pre-downloaded CSV files from a directory."""
    data = {}
    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".csv"):
            ticker = fname.replace(".csv", "")
            df = pd.read_csv(
                os.path.join(data_dir, fname),
                index_col=0,
                parse_dates=True,
            )
            df = df.sort_index()
            data[ticker] = df
    print(f"{len(data)} tickers loaded from {data_dir}")
    return data


# ---------------------------------------------------------------------------
# Window-level normalization (cf. Section I of the paper)
# ---------------------------------------------------------------------------
def _normalize_window(
    window_data: np.ndarray,
    scaling:     str,
) -> np.ndarray:
    """
    Normalizes ONE window of data (array of shape (window, n_features)).

    Expected column order: [Open, High, Low, Close, Volume, (MA optional)]
    Indices               :   0     1    2     3       4          5

    Modes
    -----
    "image"  : max(High) -> 1, min(Low) -> 0 over the window.
               Volume normalized by its max over the window.
               This is the paper's normalization — the best performing one.

    "cumret" : all prices divided by the first day's Close.
               Volume normalized by its max over the window.
               Comparison baseline (cumulative returns).

    Parameters
    ----------
    window_data : ndarray of shape (window, n_features), raw values
    scaling     : "image" or "cumret"

    Returns
    -------
    Normalized ndarray of the same shape, values in [0, 1] for "image".
    """
    w = window_data.copy()

    price_cols = slice(0, 4)   # Open, High, Low, Close
    vol_col    = 4
    # MA (col 5) follows the same normalization as prices

    if scaling == "image":
        price_min = w[:, price_cols].min()
        price_max = w[:, price_cols].max()
        denom     = price_max - price_min

        if denom > 1e-8:
            w[:, price_cols] = (w[:, price_cols] - price_min) / denom
            if w.shape[1] > 5:          # MA present
                w[:, 5] = np.clip((w[:, 5] - price_min) / denom, 0, 1)
        else:
            # Flat window (constant price) — set everything to 0.5
            w[:, price_cols] = 0.5
            if w.shape[1] > 5:
                w[:, 5] = 0.5

    elif scaling == "cumret":
        first_close = w[0, 3]           # Close of the first day
        if abs(first_close) > 1e-8:
            w[:, price_cols] = w[:, price_cols] / first_close
            if w.shape[1] > 5:
                w[:, 5] = w[:, 5] / first_close
        # If first_close ~= 0, leave as-is (pathological case)

    else:
        raise ValueError(f"Unknown scaling: '{scaling}'. Choose 'image' or 'cumret'.")

    # Volume — normalized by window max in all cases
    vol_max = w[:, vol_col].max()
    if vol_max > 1e-8:
        w[:, vol_col] = w[:, vol_col] / vol_max
    else:
        w[:, vol_col] = 0.0

    return w


# ---------------------------------------------------------------------------
# Window construction for a single ticker
# ---------------------------------------------------------------------------
def make_tabular_dataset(
    df:      pd.DataFrame,
    window:  int,
    horizon: int,
    scaling: str  = "image",
    add_ma:  bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Builds (X, y) for MLP and LSTM from a single-ticker OHLCV DataFrame.

    Logic
    -----
    For each index i from `window` to `len(df) - horizon`:
      - X[i] = normalized data over days [i-window, i)
      - y[i] = 1 if Close[i+horizon] > Close[i], 0 otherwise

    Normalization is applied window by window to prevent any
    look-ahead bias from leaking future information.

    Parameters
    ----------
    df      : Chronologically sorted OHLCV DataFrame (single ticker)
    window  : Input window length (5, 20, or 60 days)
    horizon : Number of days ahead to predict
    scaling : "image" (paper) or "cumret" (baseline)
    add_ma  : If True, appends the moving average as a 6th feature

    Returns
    -------
    X : ndarray of shape (n_samples, window, n_features)
        n_features = 6 if add_ma=True, 5 otherwise
    y : ndarray of shape (n_samples,) — binary int64 labels
    """
    df = df.copy()

    # Compute MA over the full series.
    # No leakage: MA at time t only looks at the `window` days before t.
    if add_ma:
        df["MA"] = df["Close"].rolling(window).mean()
        df = df.dropna(subset=["MA"])

    feature_cols = ["Open", "High", "Low", "Close", "Volume"]
    if add_ma:
        feature_cols.append("MA")

    arr    = df[feature_cols].values.astype(np.float32)
    closes = df["Close"].values

    n     = len(arr)
    # We need `window` days for input + `horizon` days ahead for the label.
    # Last usable position: n - horizon - 1
    max_i = n - horizon

    X_list: list[np.ndarray] = []
    y_list: list[int]        = []

    for i in range(window, max_i):
        window_raw  = arr[i - window : i]          # shape (window, n_features)
        window_norm = _normalize_window(window_raw, scaling)

        # Label: positive return between Close[i] and Close[i + horizon]
        future_close  = closes[i + horizon]
        current_close = closes[i]
        label = int(future_close > current_close)

        X_list.append(window_norm)
        y_list.append(label)

    if not X_list:
        n_feat = len(feature_cols)
        return (
            np.empty((0, window, n_feat), dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )

    X = np.array(X_list, dtype=np.float32)   # (n_samples, window, n_features)
    y = np.array(y_list,  dtype=np.int64)

    return X, y


# ---------------------------------------------------------------------------
# Multi-stock dataset with strict temporal split
# ---------------------------------------------------------------------------
def make_multi_stock_dataset(
    data:      dict[str, pd.DataFrame],
    window:    int,
    horizon:   int,
    scaling:   str  = "image",
    add_ma:    bool = True,
    train_end: str  = DEFAULT_TRAIN_END,
    val_end:   str  = DEFAULT_VAL_END,
    seed:      int  = 42,
    verbose:   bool = True,
) -> dict:
    """
    Builds the full dataset (all tickers) with a strict temporal split.

    Split
    -----
    Train : start of data  -> train_end  (inclusive)
    Val   : train_end + 1d -> val_end    (inclusive)
    Test  : val_end + 1d   -> end of data

    Shuffle is applied ONLY to the training set (as in the paper).
    Val and Test remain in chronological order.

    Parameters
    ----------
    data      : dict of {ticker: OHLCV DataFrame}
    window    : input window length in days
    horizon   : prediction horizon in days
    scaling   : "image" or "cumret"
    add_ma    : whether to include the moving average feature
    train_end : last date (inclusive) of the training set
    val_end   : last date (inclusive) of the validation set
    seed      : random seed for the training shuffle
    verbose   : whether to print split statistics

    Returns
    -------
    dict with keys:
      X_train, y_train, X_val, y_val, X_test, y_test,
      window, horizon, scaling, feature_names
    """
    rng = np.random.default_rng(seed)

    # Buckets: {split_name: (list of X arrays, list of y arrays)}
    buckets: dict[str, tuple[list, list]] = {
        "train": ([], []),
        "val":   ([], []),
        "test":  ([], []),
    }

    skipped = 0
    for ticker, df in data.items():
        # Temporal split on dates — done BEFORE any feature engineering
        splits = {
            "train": df[df.index <= train_end],
            "val":   df[(df.index > train_end) & (df.index <= val_end)],
            "test":  df[df.index > val_end],
        }

        for split_name, df_split in splits.items():
            # Minimum length: MA warmup (window) + input (window) + label (horizon)
            min_len = 2 * window + horizon
            if len(df_split) < min_len:
                skipped += 1
                continue

            X, y = make_tabular_dataset(df_split, window, horizon, scaling, add_ma)
            if len(X) == 0:
                skipped += 1
                continue

            buckets[split_name][0].append(X)
            buckets[split_name][1].append(y)

    if verbose and skipped > 0:
        print(f"  {skipped} (ticker, split) pairs skipped (insufficient data)")

    # Concatenate all tickers for each split
    def concat_bucket(key: str) -> tuple[np.ndarray, np.ndarray]:
        X_list, y_list = buckets[key]
        if not X_list:
            raise ValueError(
                f"Split '{key}' is empty. "
                f"Check dates: train_end='{train_end}', val_end='{val_end}' "
                f"and the data time range."
            )
        return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)

    X_train, y_train = concat_bucket("train")
    X_val,   y_val   = concat_bucket("val")
    X_test,  y_test  = concat_bucket("test")

    # Shuffle training set only — chronological order preserved for val/test
    idx     = rng.permutation(len(X_train))
    X_train = X_train[idx]
    y_train = y_train[idx]

    # Feature names (useful for debugging and Grad-CAM)
    feature_names = ["Open", "High", "Low", "Close", "Volume"]
    if add_ma:
        feature_names.append("MA")

    if verbose:
        for split, X, y in [
            ("Train", X_train, y_train),
            ("Val",   X_val,   y_val),
            ("Test",  X_test,  y_test),
        ]:
            pos_pct = y.mean() * 100
            print(
                f"  {split:5s} : {X.shape}"
                f"  —  {pos_pct:.1f}% positive labels"
            )

    return {
        "X_train":       X_train,
        "y_train":       y_train,
        "X_val":         X_val,
        "y_val":         y_val,
        "X_test":        X_test,
        "y_test":        y_test,
        "window":        window,
        "horizon":       horizon,
        "scaling":       scaling,
        "feature_names": feature_names,
    }


# ---------------------------------------------------------------------------
# Diagnostic utilities
# ---------------------------------------------------------------------------
def dataset_summary(dataset: dict) -> None:
    """Prints a readable summary of the dataset."""
    print("\n=== Dataset Summary ===")
    print(f"  Window    : {dataset['window']} days")
    print(f"  Horizon   : {dataset['horizon']} days")
    print(f"  Scaling   : {dataset['scaling']}")
    print(f"  Features  : {dataset['feature_names']}")
    print()
    for split in ["train", "val", "test"]:
        X = dataset[f"X_{split}"]
        y = dataset[f"y_{split}"]
        if len(X) == 0:
            print(f"  {split:5s} : EMPTY")
            continue
        pos_pct = y.mean() * 100
        print(
            f"  {split:5s} : {X.shape}"
            f"  —  {pos_pct:.1f}% positive"
            f"  —  min={X.min():.3f}  max={X.max():.3f}"
        )


def check_no_leakage(dataset: dict) -> None:
    """
    Sanity checks to verify data integrity and absence of obvious leakage.
    Raises AssertionError if any issue is detected.

    Checks performed
    ----------------
    - X and y have matching lengths in each split
    - X arrays are 3-dimensional (n_samples, window, n_features)
    - Normalized values are in [0, 1] (image scaling)
    - Labels are strictly binary {0, 1}
    """
    for split in ["train", "val", "test"]:
        X = dataset[f"X_{split}"]
        y = dataset[f"y_{split}"]
        assert len(X) == len(y), \
            f"X/y length mismatch in split '{split}'"
        assert X.ndim == 3, \
            f"X_{split} must be 3D, got {X.ndim}D"
        assert X.min() >= -1e-6, \
            f"Negative values found in X_{split} (normalization issue?)"
        assert X.max() <= 1 + 1e-6, \
            f"Values > 1 found in X_{split} (normalization issue?)"
        assert set(np.unique(y)) <= {0, 1}, \
            f"Invalid labels in y_{split}: {np.unique(y)}"

    print("  check_no_leakage: OK")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Downloading data ===")
    raw_data = download_ohlcv(
        tickers  = DEFAULT_TICKERS,
        start    = "2000-01-01",
        end      = "2023-12-31",
        save_dir = "data/raw",
    )

    print("\n=== Building dataset (window=20, horizon=5) ===")
    dataset = make_multi_stock_dataset(
        data      = raw_data,
        window    = 20,
        horizon   = 5,
        scaling   = "image",
        train_end = DEFAULT_TRAIN_END,
        val_end   = DEFAULT_VAL_END,
        verbose   = True,
    )

    dataset_summary(dataset)

    print("\n=== Integrity check ===")
    check_no_leakage(dataset)

    print("\n=== Testing cumret scaling ===")
    dataset_cumret = make_multi_stock_dataset(
        data      = raw_data,
        window    = 20,
        horizon   = 5,
        scaling   = "cumret",
        train_end = DEFAULT_TRAIN_END,
        val_end   = DEFAULT_VAL_END,
        verbose   = True,
    )
    check_no_leakage(dataset_cumret)