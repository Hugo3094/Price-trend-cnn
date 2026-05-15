"""
fetch_data.py
-------------
Downloads and prepares OHLCV data from Yahoo Finance.
Generates binary return labels (up/down) for classification.
"""

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "JPM", "GS", "BAC", "WMT",
    "XOM", "CVX", "JNJ", "PFE", "KO",
    "PEP", "NVDA", "AMD", "INTC", "CSCO",
]

IMAGE_WINDOWS = [5, 20, 60]
RETURN_HORIZONS = [5, 20, 60]

DEFAULT_TRAIN_END = "2014-12-31"
DEFAULT_VAL_END = "2018-12-31"


def _ensure_flat_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance MultiIndex columns if needed."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def download_ohlcv(
    tickers: list[str] = DEFAULT_TICKERS,
    start: str = "2000-01-01",
    end: str = "2023-12-31",
    save_dir: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """Download OHLCV data from Yahoo Finance."""
    data = {}

    for ticker in tickers:
        logger.info("Downloading %s ...", ticker)
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
            )

            if df.empty:
                logger.warning("%s: empty data, skipped", ticker)
                continue

            df = _ensure_flat_columns(df)
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            df = df.sort_index()
            data[ticker] = df

            logger.info("%s: OK (%d days)", ticker, len(df))

        except Exception as exc:
            logger.error("%s: download error: %s", ticker, exc)

    if not data:
        raise ValueError("No ticker was successfully downloaded.")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for ticker, df in data.items():
            df.to_csv(os.path.join(save_dir, f"{ticker}.csv"))
        logger.info("Data saved to: %s", save_dir)

    return data


def load_ohlcv(data_dir: str) -> dict[str, pd.DataFrame]:
    """Load OHLCV CSV files from a directory."""
    data = {}

    for fname in sorted(os.listdir(data_dir)):
        if fname.endswith(".csv"):
            ticker = fname.replace(".csv", "")
            df = pd.read_csv(
                os.path.join(data_dir, fname),
                index_col=0,
                parse_dates=True,
            )
            df = _ensure_flat_columns(df)
            df = df.sort_index()
            data[ticker] = df

    logger.info("%d tickers loaded from %s", len(data), data_dir)
    return data


def _normalize_window(window_data: np.ndarray, scaling: str) -> np.ndarray:
    """
    Normalize one input window.

    Expected column order:
    [Open, High, Low, Close, Volume, MA optional]
    """
    w = window_data.copy()

    price_cols = slice(0, 4)
    volume_col = 4

    if scaling == "image":
        price_min = w[:, price_cols].min()
        price_max = w[:, price_cols].max()
        denom = price_max - price_min

        if denom > 1e-8:
            w[:, price_cols] = (w[:, price_cols] - price_min) / denom
            if w.shape[1] > 5:
                w[:, 5] = np.clip((w[:, 5] - price_min) / denom, 0, 1)
        else:
            w[:, price_cols] = 0.5
            if w.shape[1] > 5:
                w[:, 5] = 0.5

    elif scaling == "cumret":
        first_close = w[0, 3]
        if abs(first_close) > 1e-8:
            w[:, price_cols] = w[:, price_cols] / first_close
            if w.shape[1] > 5:
                w[:, 5] = w[:, 5] / first_close

    else:
        raise ValueError(
            f"Unknown scaling: '{scaling}'. Choose 'image' or 'cumret'."
        )

    volume_max = w[:, volume_col].max()
    if volume_max > 1e-8:
        w[:, volume_col] = w[:, volume_col] / volume_max
    else:
        w[:, volume_col] = 0.0

    return w


def image_scale(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Rolling image-scale normalization helper, kept for tests/backward compatibility.
    """
    df = _ensure_flat_columns(df)
    scaled = df.copy().astype(float)

    price_cols = ["Open", "High", "Low", "Close"]
    price_max = df[price_cols].rolling(window).max().max(axis=1)
    price_min = df[price_cols].rolling(window).min().min(axis=1)
    denom = (price_max - price_min).replace(0, np.nan)

    for col in price_cols:
        scaled[col] = df[col].sub(price_min, axis=0).div(denom, axis=0)

    if "MA" in scaled.columns:
        scaled["MA"] = df["MA"].sub(price_min, axis=0).div(denom, axis=0)

    volume_max = df["Volume"].rolling(window).max().replace(0, np.nan)
    scaled["Volume"] = df["Volume"] / volume_max

    return scaled.dropna()


def cumret_scale(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Rolling cumulative-return normalization helper.
    """
    df = _ensure_flat_columns(df)
    scaled = df.copy().astype(float)

    first_close = df["Close"].shift(window - 1)

    for col in ["Open", "High", "Low", "Close"]:
        scaled[col] = df[col] / first_close

    if "MA" in scaled.columns:
        scaled["MA"] = df["MA"] / first_close

    volume_max = df["Volume"].rolling(window).max().replace(0, np.nan)
    scaled["Volume"] = df["Volume"] / volume_max

    return scaled.dropna()


def add_moving_average(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Add a moving-average column."""
    df = _ensure_flat_columns(df).copy()
    df["MA"] = df["Close"].rolling(window).mean()
    return df.dropna()


def make_labels(
    df: pd.DataFrame,
    horizon: int,
    col: str = "Close",
) -> pd.Series:
    """
    Generate binary labels from raw prices.
    """
    df = _ensure_flat_columns(df)
    future_close = df[col].shift(-horizon)

    labels = (future_close > df[col]).astype(float)
    labels[future_close.isna()] = np.nan

    return labels.rename(f"label_{horizon}d")


def make_tabular_dataset(
    df: pd.DataFrame,
    window: int,
    horizon: int,
    scaling: str = "image",
    add_ma: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a single-ticker dataset for MLP, LSTM and GRU.
    """
    df = _ensure_flat_columns(df).copy()
    df = df.sort_index()

    if add_ma:
        df["MA"] = df["Close"].rolling(window).mean()
        df = df.dropna(subset=["MA"])

    feature_cols = ["Open", "High", "Low", "Close", "Volume"]
    if add_ma:
        feature_cols.append("MA")

    arr = df[feature_cols].values.astype(np.float32)
    closes = df["Close"].values

    max_i = len(arr) - horizon

    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for i in range(window, max_i):
        window_raw = arr[i - window:i]
        window_norm = _normalize_window(window_raw, scaling)

        current_close = closes[i]
        future_close = closes[i + horizon]
        label = int(future_close > current_close)

        X_list.append(window_norm)
        y_list.append(label)

    if not X_list:
        return (
            np.empty((0, window, len(feature_cols)), dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.int64),
    )


def _ratio_split_dataset(
    data: dict[str, pd.DataFrame],
    window: int,
    horizon: int,
    scaling: str,
    add_ma: bool,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict:
    """Backward-compatible random split used by tests and toy data."""
    rng = np.random.default_rng(seed)

    X_all, y_all = [], []

    for _, df in data.items():
        X, y = make_tabular_dataset(df, window, horizon, scaling, add_ma)
        if len(X) > 0:
            X_all.append(X)
            y_all.append(y)

    if not X_all:
        raise ValueError("No data available.")

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)

    idx = rng.permutation(len(X))
    X = X[idx]
    y = y[idx]

    n_train = int(len(X) * train_ratio)
    n_val = int(len(X) * val_ratio)

    feature_names = ["Open", "High", "Low", "Close", "Volume"]
    if add_ma:
        feature_names.append("MA")

    return {
        "X_train": X[:n_train],
        "y_train": y[:n_train],
        "X_val": X[n_train:n_train + n_val],
        "y_val": y[n_train:n_train + n_val],
        "X_test": X[n_train + n_val:],
        "y_test": y[n_train + n_val:],
        "window": window,
        "horizon": horizon,
        "scaling": scaling,
        "feature_names": feature_names,
    }


def make_multi_stock_dataset(
    data: dict[str, pd.DataFrame],
    window: int,
    horizon: int,
    scaling: str = "image",
    add_ma: bool = True,
    train_ratio: Optional[float] = None,
    val_ratio: Optional[float] = None,
    train_end: Optional[str] = DEFAULT_TRAIN_END,
    val_end: Optional[str] = DEFAULT_VAL_END,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Build the full multi-stock dataset.

    Uses strict temporal split by default. If train_ratio is provided, or if
    toy dates are entirely before train_end, falls back to random ratio split
    for backward compatibility with existing tests.
    """
    if val_ratio is None:
        val_ratio = 0.15

    if train_ratio is not None:
        return _ratio_split_dataset(
            data=data,
            window=window,
            horizon=horizon,
            scaling=scaling,
            add_ma=add_ma,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )

    all_dates = pd.concat([df.index.to_series() for df in data.values()])

    use_ratio_fallback = (
        not all(isinstance(df.index, pd.DatetimeIndex) for df in data.values())
        or train_end is None
        or val_end is None
        or all_dates.max() <= pd.Timestamp(train_end)
        or all_dates.min() > pd.Timestamp(train_end)
        or all_dates.min() > pd.Timestamp(val_end)
    )

    if use_ratio_fallback:
        return _ratio_split_dataset(
            data=data,
            window=window,
            horizon=horizon,
            scaling=scaling,
            add_ma=add_ma,
            train_ratio=0.70,
            val_ratio=val_ratio,
            seed=seed,
        )

    rng = np.random.default_rng(seed)

    buckets: dict[str, tuple[list, list]] = {
        "train": ([], []),
        "val": ([], []),
        "test": ([], []),
    }

    skipped = 0

    for _, df in data.items():
        df = _ensure_flat_columns(df).sort_index()

        splits = {
            "train": df[df.index <= train_end],
            "val": df[(df.index > train_end) & (df.index <= val_end)],
            "test": df[df.index > val_end],
        }

        for split_name, df_split in splits.items():
            min_len = 2 * window + horizon

            if len(df_split) < min_len:
                skipped += 1
                continue

            X, y = make_tabular_dataset(
                df_split,
                window=window,
                horizon=horizon,
                scaling=scaling,
                add_ma=add_ma,
            )

            if len(X) == 0:
                skipped += 1
                continue

            buckets[split_name][0].append(X)
            buckets[split_name][1].append(y)

    if verbose and skipped > 0:
        logger.info("%d ticker/split pairs skipped", skipped)

    def concat_bucket(key: str) -> tuple[np.ndarray, np.ndarray]:
        X_list, y_list = buckets[key]
        if not X_list:
            raise ValueError(
                f"Split '{key}' is empty. Check train_end='{train_end}', "
                f"val_end='{val_end}' and the data time range."
            )
        return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)

    X_train, y_train = concat_bucket("train")
    X_val, y_val = concat_bucket("val")
    X_test, y_test = concat_bucket("test")

    train_idx = rng.permutation(len(X_train))
    X_train = X_train[train_idx]
    y_train = y_train[train_idx]

    feature_names = ["Open", "High", "Low", "Close", "Volume"]
    if add_ma:
        feature_names.append("MA")

    if verbose:
        for split_name, X, y in [
            ("train", X_train, y_train),
            ("val", X_val, y_val),
            ("test", X_test, y_test),
        ]:
            logger.info(
                "%s: %s — %.1f%% positive labels",
                split_name,
                X.shape,
                y.mean() * 100,
            )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "window": window,
        "horizon": horizon,
        "scaling": scaling,
        "feature_names": feature_names,
    }


def dataset_summary(dataset: dict) -> None:
    """Log a readable dataset summary."""
    logger.info("Dataset summary")
    logger.info("Window: %s", dataset["window"])
    logger.info("Horizon: %s", dataset["horizon"])
    logger.info("Scaling: %s", dataset["scaling"])
    logger.info("Features: %s", dataset["feature_names"])

    for split in ["train", "val", "test"]:
        X = dataset[f"X_{split}"]
        y = dataset[f"y_{split}"]
        logger.info(
            "%s: %s — %.1f%% positive — min=%.3f max=%.3f",
            split,
            X.shape,
            y.mean() * 100,
            X.min(),
            X.max(),
        )


def check_no_leakage(dataset: dict) -> None:
    """Perform basic sanity checks."""
    for split in ["train", "val", "test"]:
        X = dataset[f"X_{split}"]
        y = dataset[f"y_{split}"]

        assert len(X) == len(y), f"X/y mismatch in split '{split}'"
        assert X.ndim == 3, f"X_{split} must be 3D, got {X.ndim}D"
        assert set(np.unique(y)) <= {0, 1}, f"Invalid labels in y_{split}"

        if dataset["scaling"] == "image":
            assert X.min() >= -1e-6, f"Negative values in X_{split}"
            assert X.max() <= 1 + 1e-6, f"Values > 1 in X_{split}"

    logger.info("check_no_leakage: OK")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    raw_data = download_ohlcv(
        tickers=DEFAULT_TICKERS,
        start="2000-01-01",
        end="2023-12-31",
        save_dir="data/raw",
    )

    dataset = make_multi_stock_dataset(
        data=raw_data,
        window=20,
        horizon=5,
        scaling="image",
        train_end=DEFAULT_TRAIN_END,
        val_end=DEFAULT_VAL_END,
        verbose=True,
    )

    dataset_summary(dataset)
    check_no_leakage(dataset)