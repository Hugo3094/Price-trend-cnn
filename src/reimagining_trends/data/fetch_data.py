"""
fetch_data.py
-------------
Télécharge et prépare les données OHLCV depuis Yahoo Finance.
Génère les labels de rendement (up/down) pour la classification binaire.

Référence : Jiang, Kelly & Xiu (2023) - (Re-)Imag(in)ing Price Trends
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Optional


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "JPM", "GS", "BAC", "WMT",
    "XOM", "CVX", "JNJ", "PFE", "KO",
    "PEP", "NVDA", "AMD", "INTC", "CSCO",
]

IMAGE_WINDOWS   = [5, 20, 60]   # fenêtres d'image (jours)
RETURN_HORIZONS = [5, 20, 60]   # horizons de prédiction (jours)


# ---------------------------------------------------------------------------
# Téléchargement
# ---------------------------------------------------------------------------
def download_ohlcv(
    tickers: list[str] = DEFAULT_TICKERS,
    start:   str = "2000-01-01",
    end:     str = "2023-12-31",
    save_dir: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """
    Télécharge les données OHLCV pour une liste de tickers.

    Returns
    -------
    dict  {ticker: DataFrame avec colonnes Open/High/Low/Close/Volume}
    """
    data = {}
    for ticker in tickers:
        print(f"  Téléchargement : {ticker} ...", end=" ")
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            if df.empty:
                print("VIDE — ignoré")
                continue
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            data[ticker] = df
            print(f"OK ({len(df)} jours)")
        except Exception as e:
            print(f"ERREUR : {e}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for ticker, df in data.items():
            df.to_csv(os.path.join(save_dir, f"{ticker}.csv"))
        print(f"\nDonnées sauvegardées dans : {save_dir}")

    return data


def load_ohlcv(data_dir: str) -> dict[str, pd.DataFrame]:
    """Charge des CSV déjà téléchargés depuis un dossier."""
    data = {}
    for fname in os.listdir(data_dir):
        if fname.endswith(".csv"):
            ticker = fname.replace(".csv", "")
            df = pd.read_csv(os.path.join(data_dir, fname), index_col=0, parse_dates=True)
            data[ticker] = df
    print(f"{len(data)} tickers chargés depuis {data_dir}")
    return data


# ---------------------------------------------------------------------------
# Normalisation (image scaling — cf. papier Section I)
# ---------------------------------------------------------------------------
def image_scale(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Normalise les prix sur une fenêtre glissante de `window` jours.
    Le max High et le min Low de la fenêtre sont mappés sur [0, 1].
    Idem pour le volume (max volume de la fenêtre → 1).

    C'est exactement la normalisation utilisée dans le papier pour
    convertir toutes les actions sur une échelle comparable.
    """
    scaled = df.copy().astype(float)

    price_cols = ["Open", "High", "Low", "Close"]
    price_max  = df[price_cols].rolling(window).max().max(axis=1)
    price_min  = df[price_cols].rolling(window).min().min(axis=1)
    denom      = (price_max - price_min).replace(0, np.nan)

    for col in price_cols:
        scaled[col] = (df[col] - price_min) / denom

    vol_max     = df["Volume"].rolling(window).max().replace(0, np.nan)
    scaled["Volume"] = df["Volume"] / vol_max

    return scaled.dropna()


def cumret_scale(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Normalise les prix par le Close du début de chaque fenêtre.
    Équivalent d'un rendement cumulé — baseline de comparaison.
    """
    scaled     = df.copy().astype(float)
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
    """Ajoute la moyenne mobile sur `window` jours (colonne MA)."""
    df = df.copy()
    df["MA"] = df["Close"].rolling(window).mean()
    return df.dropna()


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def make_labels(
    df:      pd.DataFrame,
    horizon: int,
    col:     str = "Close",
) -> pd.Series:
    """
    Génère des labels binaires :
      1  si le rendement sur `horizon` jours est positif
      0  sinon

    Le label à t correspond au rendement entre t et t+horizon.
    """
    future_return = df[col].shift(-horizon) / df[col] - 1
    return (future_return > 0).astype(int).rename(f"label_{horizon}d")


# ---------------------------------------------------------------------------
# Sliding windows — données tabulaires (pour MLP / LSTM)
# ---------------------------------------------------------------------------
def make_tabular_dataset(
    df:      pd.DataFrame,
    window:  int,
    horizon: int,
    scaling: str = "image",        # "image" | "cumret"
    add_ma:  bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construit (X, y) pour MLP et LSTM à partir d'une série OHLCV.

    Parameters
    ----------
    df      : DataFrame OHLCV pour un seul ticker
    window  : longueur de la fenêtre d'entrée (ex. 5, 20, 60)
    horizon : horizon de prédiction
    scaling : méthode de normalisation
    add_ma  : inclure la moyenne mobile

    Returns
    -------
    X : (n_samples, window, n_features)   — format séquentiel pour LSTM
    y : (n_samples,)                       — labels binaires
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
    data:    dict[str, pd.DataFrame],
    window:  int,
    horizon: int,
    scaling: str = "image",
    add_ma:  bool = True,
    train_ratio: float = 0.7,
) -> dict:
    """
    Construit le dataset complet (tous tickers) et effectue le split
    train / val / test chronologique.

    Split utilisé dans le papier :
      - Train + Val : 1993-2000
      - Test        : 2001-2019

    Ici on adapte avec un ratio configurable.
    """
    X_all, y_all = [], []

    for ticker, df in data.items():
        X, y = make_tabular_dataset(df, window, horizon, scaling, add_ma)
        if len(X) > 0:
            X_all.append(X)
            y_all.append(y)

    if not X_all:
        raise ValueError("Aucune donnée disponible.")

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)

    # Shuffle aléatoire (comme dans le papier pour train/val)
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    n_train = int(len(X) * train_ratio)
    n_val   = int(len(X) * (1 - train_ratio) / 2)

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
# Point d'entrée
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Téléchargement des données ===")
    raw_data = download_ohlcv(
        tickers  = DEFAULT_TICKERS,
        start    = "2000-01-01",
        end      = "2023-12-31",
        save_dir = "data/raw",
    )

    print("\n=== Construction du dataset tabulaire (window=20, horizon=5) ===")
    dataset = make_multi_stock_dataset(
        data    = raw_data,
        window  = 20,
        horizon = 5,
        scaling = "image",
    )
    for split in ["train", "val", "test"]:
        X = dataset[f"X_{split}"]
        y = dataset[f"y_{split}"]
        pos = y.mean() * 100
        print(f"  {split:5s} : {X.shape}  —  {pos:.1f}% positifs")
