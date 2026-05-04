"""
ohlc_chart.py
-------------
Génère des images OHLC noir et blanc à partir de données de prix,
en suivant exactement la spécification du papier :

  - Fond noir, objets blancs
  - Chaque jour = 3 pixels de large (barre centrale + open + close)
  - Hauteur de l'image normalisée : max High → haut, min Low → bas
  - Volume dans le 1/5 inférieur de l'image
  - Moyenne mobile superposée (1 pixel par jour)

Référence : Jiang, Kelly & Xiu (2023), Section I
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from typing import Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection


# ---------------------------------------------------------------------------
# Dimensions des images (papier : 32×15 pour 5j, 64×60 pour 20j, 96×180 pour 60j)
# ---------------------------------------------------------------------------
IMAGE_SPECS = {
    5:  {"width": 32,  "height": 15},
    20: {"width": 64,  "height": 60},
    60: {"width": 96,  "height": 180},
}

VOLUME_FRACTION = 0.20   # fraction de la hauteur réservée au volume


# ---------------------------------------------------------------------------
# Génération d'une image OHLC pour une fenêtre
# ---------------------------------------------------------------------------
def _normalize_prices(
    opens: np.ndarray,
    highs: np.ndarray,
    lows:  np.ndarray,
    closes: np.ndarray,
    ma:    Optional[np.ndarray] = None,
) -> tuple:
    """
    Normalise les prix sur [0, 1] :
      - 1 correspond au max des High (ou MA si supérieur)
      - 0 correspond au min des Low
    """
    all_prices = np.concatenate([highs, lows])
    if ma is not None:
        all_prices = np.concatenate([all_prices, ma[~np.isnan(ma)]])

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
    """Normalise le volume sur [0, 1]."""
    v_max = volumes.max()
    return volumes / v_max if v_max > 0 else volumes


def generate_ohlc_image(
    df:          pd.DataFrame,
    window:      int = 20,
    include_vol: bool = True,
    include_ma:  bool = True,
) -> np.ndarray:
    """
    Génère une image OHLC numpy uint8 (H, W) pour une fenêtre de `window` jours.

    Parameters
    ----------
    df          : DataFrame avec colonnes Open/High/Low/Close/Volume (window lignes)
    window      : 5, 20 ou 60
    include_vol : afficher les barres de volume
    include_ma  : afficher la moyenne mobile

    Returns
    -------
    np.ndarray de shape (height, width) en uint8 — 0=noir, 255=blanc
    """
    assert len(df) == window, f"Le DataFrame doit contenir exactement {window} lignes."
    assert window in IMAGE_SPECS, f"window doit être dans {list(IMAGE_SPECS.keys())}"

    specs  = IMAGE_SPECS[window]
    W      = specs["width"]
    H      = specs["height"]

    # Zone prix et zone volume
    if include_vol:
        h_vol   = int(H * VOLUME_FRACTION)
        h_price = H - h_vol
    else:
        h_vol   = 0
        h_price = H

    # Extraction des séries
    opens   = df["Open"].values.astype(float)
    highs   = df["High"].values.astype(float)
    lows    = df["Low"].values.astype(float)
    closes  = df["Close"].values.astype(float)
    volumes = df["Volume"].values.astype(float) if include_vol else None

    ma = None
    if include_ma and "MA" in df.columns:
        ma = df["MA"].values.astype(float)

    # Normalisation
    opens_n, highs_n, lows_n, closes_n, ma_n = _normalize_prices(
        opens, highs, lows, closes, ma
    )

    # Image noire
    img = np.zeros((H, W), dtype=np.uint8)

    # Chaque jour occupe 3 pixels de large : [open | barre | close]
    # On centre les jours dans la largeur disponible
    day_width = 3
    x_start   = (W - window * day_width) // 2

    for i in range(window):
        x_open  = x_start + i * day_width          # pixel open (gauche)
        x_bar   = x_start + i * day_width + 1      # pixel barre centrale
        x_close = x_start + i * day_width + 2      # pixel close (droite)

        if x_close >= W:
            break

        # Conversion en pixels (axe y inversé : 0 en haut)
        def to_px(v, h):
            return int((1.0 - np.clip(v, 0, 1)) * (h - 1))

        y_high  = to_px(highs_n[i],  h_price)
        y_low   = to_px(lows_n[i],   h_price)
        y_open  = to_px(opens_n[i],  h_price)
        y_close = to_px(closes_n[i], h_price)

        # Barre haute-basse
        img[y_high:y_low + 1, x_bar] = 255

        # Marque d'ouverture (gauche)
        img[y_open,  x_open]  = 255

        # Marque de clôture (droite)
        img[y_close, x_close] = 255

    # Moyenne mobile
    if ma_n is not None:
        prev_px = None
        for i in range(window):
            if np.isnan(ma_n[i]):
                continue
            x = x_start + i * day_width + 1
            y = to_px(ma_n[i], h_price)
            if 0 <= x < W and 0 <= y < h_price:
                img[y, x] = 255
                # Connexion avec le point précédent (ligne)
                if prev_px is not None:
                    x0, y0 = prev_px
                    _draw_line(img, x0, y0, x, y, h_price, W)
            prev_px = (x, y)

    # Volume
    if include_vol and volumes is not None:
        vol_n = _normalize_volume(volumes)
        for i in range(window):
            x_bar = x_start + i * day_width + 1
            if x_bar >= W:
                break
            bar_h = int(vol_n[i] * h_vol)
            if bar_h > 0:
                y_top = H - bar_h
                img[y_top:H, x_bar] = 255

    return img


def _draw_line(
    img: np.ndarray,
    x0: int, y0: int,
    x1: int, y1: int,
    h: int, w: int,
) -> None:
    """Tracé de Bresenham pour connecter deux points (MA)."""
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
# Dataset d'images complet
# ---------------------------------------------------------------------------
def make_image_dataset(
    data:        dict[str, pd.DataFrame],
    window:      int = 20,
    horizon:     int = 5,
    include_vol: bool = True,
    include_ma:  bool = True,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
) -> dict:
    """
    Génère le dataset images + labels pour TOUS les tickers.

    Returns
    -------
    dict avec X_train/val/test (N, H, W, 1) et y_train/val/test
    """
    X_list, y_list = [], []
    specs = IMAGE_SPECS[window]
    H, W  = specs["height"], specs["width"]

    for ticker, df in data.items():
        # Ajouter MA
        if include_ma:
            df = df.copy()
            df["MA"] = df["Close"].rolling(window).mean()
            df = df.dropna()

        n = len(df)
        if n < window + horizon:
            continue

        for i in range(window, n - horizon):
            window_df = df.iloc[i - window:i]
            future_ret = df["Close"].iloc[i + horizon - 1] / df["Close"].iloc[i - 1] - 1
            label = int(future_ret > 0)

            try:
                img = generate_ohlc_image(window_df, window, include_vol, include_ma)
                X_list.append(img)
                y_list.append(label)
            except Exception:
                continue

    if not X_list:
        raise ValueError("Aucune image générée.")

    X = np.array(X_list, dtype=np.float32) / 255.0  # normalisation [0,1]
    X = X[:, :, :, np.newaxis]                        # ajout canal : (N, H, W, 1)
    y = np.array(y_list, dtype=np.int64)

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    n1 = int(len(X) * train_ratio)
    n2 = int(len(X) * (train_ratio + val_ratio))

    return {
        "X_train": X[:n1],   "y_train": y[:n1],
        "X_val":   X[n1:n2], "y_val":   y[n1:n2],
        "X_test":  X[n2:],   "y_test":  y[n2:],
        "image_shape": (H, W, 1),
        "window": window, "horizon": horizon,
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def visualize_sample(
    img:   np.ndarray,
    label: int,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """Affiche une image OHLC avec son label."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(img.squeeze(), cmap="gray", aspect="auto", interpolation="nearest")
    color = "green" if label == 1 else "red"
    lbl   = "UP ↑" if label == 1 else "DOWN ↓"
    ax.set_title(title or f"Label : {lbl}", color=color, fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def visualize_grid(
    X: np.ndarray,
    y: np.ndarray,
    n: int = 16,
    save_path: Optional[str] = None,
) -> None:
    """Affiche une grille de n images avec leurs labels."""
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

    fig.suptitle("Échantillons d'images OHLC", fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from data.fetch_data import download_ohlcv, add_moving_average

    print("=== Test de génération d'images OHLC ===")

    # Données de test (1 ticker)
    raw = download_ohlcv(["AAPL"], start="2020-01-01", end="2022-12-31")
    df  = raw["AAPL"]
    df  = add_moving_average(df, window=20)

    # Image 20 jours
    window_df = df.iloc[20:40]
    img = generate_ohlc_image(window_df, window=20, include_vol=True, include_ma=True)
    print(f"Image générée : shape={img.shape}, min={img.min()}, max={img.max()}")

    # Visualisation
    visualize_sample(img, label=1, title="Test image 20 jours — AAPL")

    # Dataset complet
    print("\n=== Construction du dataset images ===")
    raw_multi = download_ohlcv(["AAPL", "MSFT", "GOOGL"], start="2015-01-01", end="2022-12-31")
    dataset   = make_image_dataset(raw_multi, window=20, horizon=5)

    for split in ["train", "val", "test"]:
        X = dataset[f"X_{split}"]
        y = dataset[f"y_{split}"]
        print(f"  {split:5s} : {X.shape}  —  {y.mean()*100:.1f}% positifs")

    visualize_grid(dataset["X_train"], dataset["y_train"], n=16)
