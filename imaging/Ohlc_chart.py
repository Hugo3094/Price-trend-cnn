"""
ohlc_chart.py
-------------
Generates black and white OHLC images from price data,
following exactly the paper's specification:

  - Black background, white objects
  - Each day = 3 pixels wide (center bar + open mark + close mark)
  - Normalized image height: max High -> top, min Low -> bottom
  - Volume in the bottom 1/5 of the image
  - Moving average overlaid (1 pixel per day, connected with Bresenham)

Reference: Jiang, Kelly & Xiu (2023), Section I
"""

import os
import numpy as np
import pandas as pd
from typing import Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.fetch_data import DEFAULT_TRAIN_END, DEFAULT_VAL_END


# ---------------------------------------------------------------------------
# Image dimensions (paper: 32x15 for 5d, 64x60 for 20d, 96x180 for 60d)
# ---------------------------------------------------------------------------
IMAGE_SPECS = {
    5:  {"width": 32,  "height": 15},
    20: {"width": 64,  "height": 60},
    60: {"width": 96,  "height": 180},
}

VOLUME_FRACTION = 0.20   # fraction of image height reserved for volume


# ---------------------------------------------------------------------------
# Price normalization
# ---------------------------------------------------------------------------
def _normalize_prices(
    opens:  np.ndarray,
    highs:  np.ndarray,
    lows:   np.ndarray,
    closes: np.ndarray,
    ma:     Optional[np.ndarray] = None,
) -> tuple:
    """
    Normalizes prices to [0, 1] over the window:
      - 1 corresponds to the max of High (or MA if higher)
      - 0 corresponds to the min of Low

    Parameters
    ----------
    opens, highs, lows, closes : price arrays of shape (window,)
    ma                         : optional moving average array of shape (window,)

    Returns
    -------
    Tuple of normalized arrays (opens, highs, lows, closes, ma or None)
    """
    all_prices = np.concatenate([highs, lows])
    if ma is not None:
        valid_ma = ma[~np.isnan(ma)]
        if len(valid_ma) > 0:
            all_prices = np.concatenate([all_prices, valid_ma])

    p_max = all_prices.max()
    p_min = all_prices.min()
    denom = p_max - p_min if p_max != p_min else 1.0

    def norm(x):
        return (x - p_min) / denom

    return (
        norm(opens),
        norm(highs),
        norm(lows),
        norm(closes),
        norm(ma) if ma is not None else None,
    )


def _normalize_volume(volumes: np.ndarray) -> np.ndarray:
    """Normalizes volume to [0, 1] by the window maximum."""
    v_max = volumes.max()
    return volumes / v_max if v_max > 1e-8 else np.zeros_like(volumes)


# ---------------------------------------------------------------------------
# Bresenham line drawing (for the moving average)
# ---------------------------------------------------------------------------
def _draw_line(
    img: np.ndarray,
    x0: int, y0: int,
    x1: int, y1: int,
    h:  int, w:  int,
) -> None:
    """
    Draws a line between two points using Bresenham's algorithm.
    Used to connect consecutive moving average dots.

    Parameters
    ----------
    img     : 2D uint8 image array (modified in place)
    x0, y0  : start pixel coordinates
    x1, y1  : end pixel coordinates
    h, w    : image height and width (for boundary checks)
    """
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        if 0 <= x0 < w and 0 <= y0 < h:
            img[y0, x0] = 255
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0  += sx
        if e2 < dx:
            err += dx
            y0  += sy


# ---------------------------------------------------------------------------
# Single window image generation
# ---------------------------------------------------------------------------
def generate_ohlc_image(
    df:          pd.DataFrame,
    window:      int  = 20,
    include_vol: bool = True,
    include_ma:  bool = True,
) -> np.ndarray:
    """
    Generates a black and white OHLC numpy image for a single price window.

    Layout
    ------
    - Top 80%  : OHLC bars + moving average line
    - Bottom 20%: volume bars (if include_vol=True)

    Each day occupies 3 pixels wide:
      col 0 : open mark  (single white pixel at open price level)
      col 1 : high-low bar (white pixels from low to high)
      col 2 : close mark (single white pixel at close price level)

    Parameters
    ----------
    df          : DataFrame with Open/High/Low/Close/Volume columns,
                  exactly `window` rows, already containing MA column if needed
    window      : 5, 20, or 60 (must be in IMAGE_SPECS)
    include_vol : whether to draw volume bars in the bottom section
    include_ma  : whether to draw the moving average line

    Returns
    -------
    np.ndarray of shape (height, width), dtype=uint8 — 0=black, 255=white
    """
    assert len(df) == window, \
        f"DataFrame must contain exactly {window} rows, got {len(df)}."
    assert window in IMAGE_SPECS, \
        f"window must be one of {list(IMAGE_SPECS.keys())}."

    specs = IMAGE_SPECS[window]
    W     = specs["width"]
    H     = specs["height"]

    # Price zone and volume zone heights
    if include_vol:
        h_vol   = int(H * VOLUME_FRACTION)
        h_price = H - h_vol
    else:
        h_vol   = 0
        h_price = H

    # Extract raw series
    opens   = df["Open"].values.astype(float)
    highs   = df["High"].values.astype(float)
    lows    = df["Low"].values.astype(float)
    closes  = df["Close"].values.astype(float)
    volumes = df["Volume"].values.astype(float) if include_vol else None

    ma = None
    if include_ma and "MA" in df.columns:
        ma = df["MA"].values.astype(float)

    # Normalize prices
    opens_n, highs_n, lows_n, closes_n, ma_n = _normalize_prices(
        opens, highs, lows, closes, ma
    )

    # Initialize black image
    img = np.zeros((H, W), dtype=np.uint8)

    # Center the OHLC bars horizontally
    day_width = 3
    x_start   = (W - window * day_width) // 2

    def to_px(v: float, h: int) -> int:
        """Converts normalized price [0,1] to pixel row (y-axis inverted)."""
        return int((1.0 - np.clip(v, 0.0, 1.0)) * (h - 1))

    # Draw OHLC bars
    for i in range(window):
        x_open  = x_start + i * day_width        # open mark  (left pixel)
        x_bar   = x_start + i * day_width + 1    # center bar (middle pixel)
        x_close = x_start + i * day_width + 2    # close mark (right pixel)

        if x_close >= W:
            break

        y_high  = to_px(highs_n[i],  h_price)
        y_low   = to_px(lows_n[i],   h_price)
        y_open  = to_px(opens_n[i],  h_price)
        y_close = to_px(closes_n[i], h_price)

        # High-low vertical bar
        img[y_high : y_low + 1, x_bar] = 255

        # Open mark (single pixel, left column)
        if 0 <= y_open < h_price:
            img[y_open, x_open] = 255

        # Close mark (single pixel, right column)
        if 0 <= y_close < h_price:
            img[y_close, x_close] = 255

    # Draw moving average line
    if ma_n is not None:
        prev_px = None
        for i in range(window):
            if np.isnan(ma_n[i]):
                prev_px = None
                continue
            x = x_start + i * day_width + 1
            y = to_px(ma_n[i], h_price)
            if 0 <= x < W and 0 <= y < h_price:
                img[y, x] = 255
                if prev_px is not None:
                    x0, y0 = prev_px
                    _draw_line(img, x0, y0, x, y, h_price, W)
            prev_px = (x, y)

    # Draw volume bars in the bottom section
    if include_vol and volumes is not None:
        vol_n = _normalize_volume(volumes)
        for i in range(window):
            x_bar = x_start + i * day_width + 1
            if x_bar >= W:
                break
            bar_h = int(vol_n[i] * h_vol)
            if bar_h > 0:
                y_top = H - bar_h
                img[y_top : H, x_bar] = 255

    return img


# ---------------------------------------------------------------------------
# Full image dataset with strict temporal split
# ---------------------------------------------------------------------------
def make_image_dataset(
    data:        dict[str, pd.DataFrame],
    window:      int  = 20,
    horizon:     int  = 5,
    include_vol: bool = True,
    include_ma:  bool = True,
    train_end:   str  = DEFAULT_TRAIN_END,
    val_end:     str  = DEFAULT_VAL_END,
    seed:        int  = 42,
    verbose:     bool = True,
) -> dict:
    """
    Generates the full image dataset (all tickers) with a strict temporal split,
    consistent with make_multi_stock_dataset() in fetch_data.py.

    Split
    -----
    Train : start of data  -> train_end  (inclusive)
    Val   : train_end + 1d -> val_end    (inclusive)
    Test  : val_end + 1d   -> end of data

    Shuffle is applied ONLY to the training set.
    Val and Test remain in chronological order.

    Label definition (consistent with fetch_data.py)
    -------------------------------------------------
    y = 1 if Close[i + horizon] > Close[i], else 0
    where i is the index of the last day in the input window.

    Parameters
    ----------
    data        : dict of {ticker: OHLCV DataFrame}
    window      : input window length in days (5, 20, or 60)
    horizon     : prediction horizon in days
    include_vol : whether to draw volume bars
    include_ma  : whether to draw the moving average line
    train_end   : last date (inclusive) of the training set
    val_end     : last date (inclusive) of the validation set
    seed        : random seed for training shuffle
    verbose     : whether to print split statistics

    Returns
    -------
    dict with keys:
      X_train, y_train, X_val, y_val, X_test, y_test,
      image_shape, window, horizon
    """
    assert window in IMAGE_SPECS, \
        f"window must be one of {list(IMAGE_SPECS.keys())}."

    rng   = np.random.default_rng(seed)
    specs = IMAGE_SPECS[window]
    H, W  = specs["height"], specs["width"]

    # Buckets: {split_name: (list of images, list of labels)}
    buckets: dict[str, tuple[list, list]] = {
        "train": ([], []),
        "val":   ([], []),
        "test":  ([], []),
    }

    skipped = 0
    for ticker, df in data.items():
        df = df.copy()

        # Compute MA on the full series before splitting.
        # No leakage: MA at time t only looks at the `window` days before t.
        if include_ma:
            df["MA"] = df["Close"].rolling(window).mean()
            df = df.dropna(subset=["MA"])

        # Temporal split on dates — done BEFORE building windows
        splits = {
            "train": df[df.index <= train_end],
            "val":   df[(df.index > train_end) & (df.index <= val_end)],
            "test":  df[df.index > val_end],
        }

        for split_name, df_split in splits.items():
            # Need at least window days of input + horizon days for the label
            min_len = window + horizon
            if len(df_split) < min_len:
                skipped += 1
                continue

            closes = df_split["Close"].values
            n      = len(df_split)

            for i in range(window, n - horizon):
                window_df = df_split.iloc[i - window : i]

                # Label consistent with fetch_data.py
                current_close = closes[i]
                future_close  = closes[i + horizon]
                label = int(future_close > current_close)

                try:
                    img = generate_ohlc_image(
                        window_df,
                        window      = window,
                        include_vol = include_vol,
                        include_ma  = include_ma,
                    )
                    buckets[split_name][0].append(img)
                    buckets[split_name][1].append(label)
                except Exception:
                    skipped += 1
                    continue

    if verbose and skipped > 0:
        print(f"  {skipped} windows skipped (insufficient data or generation error)")

    # Concatenate all tickers for each split
    def concat_bucket(key: str) -> tuple[np.ndarray, np.ndarray]:
        X_list, y_list = buckets[key]
        if not X_list:
            raise ValueError(
                f"Split '{key}' is empty. "
                f"Check dates: train_end='{train_end}', val_end='{val_end}'."
            )
        X = np.array(X_list, dtype=np.float32) / 255.0   # normalize to [0, 1]
        X = X[:, :, :, np.newaxis]                         # add channel: (N, H, W, 1)
        y = np.array(y_list, dtype=np.int64)
        return X, y

    X_train, y_train = concat_bucket("train")
    X_val,   y_val   = concat_bucket("val")
    X_test,  y_test  = concat_bucket("test")

    # Shuffle training set only — chronological order preserved for val/test
    idx     = rng.permutation(len(X_train))
    X_train = X_train[idx]
    y_train = y_train[idx]

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
        "X_train":     X_train,
        "y_train":     y_train,
        "X_val":       X_val,
        "y_val":       y_val,
        "X_test":      X_test,
        "y_test":      y_test,
        "image_shape": (H, W, 1),
        "window":      window,
        "horizon":     horizon,
    }


# ---------------------------------------------------------------------------
# Visualization utilities
# ---------------------------------------------------------------------------
def visualize_sample(
    img:       np.ndarray,
    label:     int,
    title:     Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """Displays a single OHLC image with its label."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(img.squeeze(), cmap="gray", aspect="auto", interpolation="nearest")
    color = "green" if label == 1 else "red"
    lbl   = "UP ↑"  if label == 1 else "DOWN ↓"
    ax.set_title(title or f"Label: {lbl}", color=color, fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def visualize_grid(
    X:         np.ndarray,
    y:         np.ndarray,
    n:         int = 16,
    save_path: Optional[str] = None,
) -> None:
    """Displays a grid of n OHLC images with their labels."""
    n    = min(n, len(X))
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5))
    axes = axes.flatten()

    for i in range(n):
        ax    = axes[i]
        color = "green" if y[i] == 1 else "red"
        ax.imshow(X[i].squeeze(), cmap="gray", aspect="auto", interpolation="nearest")
        ax.set_title("UP ↑" if y[i] == 1 else "DOWN ↓", color=color, fontsize=9)
        ax.axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("OHLC Image Samples", fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from data.fetch_data import download_ohlcv, DEFAULT_TRAIN_END, DEFAULT_VAL_END

    print("=== OHLC image generation test ===")

    # Single ticker test
    raw = download_ohlcv(["AAPL"], start="2010-01-01", end="2022-12-31")
    df  = raw["AAPL"].copy()
    df["MA"] = df["Close"].rolling(20).mean()
    df = df.dropna()

    window_df = df.iloc[20:40]
    img = generate_ohlc_image(window_df, window=20, include_vol=True, include_ma=True)
    print(f"Generated image : shape={img.shape}, min={img.min()}, max={img.max()}")
    visualize_sample(img, label=1, title="Test image 20 days — AAPL")

    # Full multi-stock dataset
    print("\n=== Building image dataset ===")
    raw_multi = download_ohlcv(
        ["AAPL", "MSFT", "GOOGL"],
        start="2010-01-01",
        end="2022-12-31",
    )
    dataset = make_image_dataset(
        raw_multi,
        window    = 20,
        horizon   = 5,
        train_end = DEFAULT_TRAIN_END,
        val_end   = DEFAULT_VAL_END,
        verbose   = True,
    )

    for split in ["train", "val", "test"]:
        X = dataset[f"X_{split}"]
        y = dataset[f"y_{split}"]
        print(
            f"  {split:5s} : {X.shape}"
            f"  —  {y.mean() * 100:.1f}% positive"
            f"  —  min={X.min():.3f}  max={X.max():.3f}"
        )

    visualize_grid(dataset["X_train"], dataset["y_train"], n=16)