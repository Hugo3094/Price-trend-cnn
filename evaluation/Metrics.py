"""
metrics.py
----------
Full evaluation and comparison of MLP, LSTM, and CNN models.

Financial metrics (paper):
  - Annualized Sharpe ratio of decile H-L strategies
  - Mean return per decile

Standard ML metrics:
  - Accuracy, F1, AUC-ROC, Brier score

Visualizations:
  - Cumulative return curves
  - Model comparison heatmap
  - Decile return bar charts
  - Grad-CAM overlay
  - Confusion matrix

Reference: Jiang, Kelly & Xiu (2023) - (Re-)Imag(in)ing Price Trends
"""

import os
import numpy as np
import pandas as pd
from typing import Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# ---------------------------------------------------------------------------
# ML metrics
# ---------------------------------------------------------------------------
def compute_ml_metrics(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    y_proba: np.ndarray,
) -> dict:
    """
    Computes standard classification metrics.

    Parameters
    ----------
    y_true  : ground truth labels (0 or 1), shape (N,)
    y_pred  : predicted labels (0 or 1), shape (N,)
    y_proba : predicted probability of class 1, shape (N,)

    Returns
    -------
    dict with keys: accuracy, f1, auc, brier
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1":       f1_score(y_true, y_pred, zero_division=0),
        "auc":      roc_auc_score(y_true, y_proba),
        "brier":    brier_score_loss(y_true, y_proba),
    }


def print_classification_report(
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    model_name: str = "Model",
) -> None:
    """Prints a formatted sklearn classification report."""
    print(f"\n{'=' * 40}")
    print(f"  {model_name}")
    print("=" * 40)
    print(classification_report(y_true, y_pred, target_names=["DOWN", "UP"]))


# ---------------------------------------------------------------------------
# Financial metrics
# ---------------------------------------------------------------------------
def annualized_sharpe(
    returns:          np.ndarray,
    periods_per_year: int   = 52,
    risk_free:        float = 0.0,
) -> float:
    """
    Computes the annualized Sharpe ratio.

    Parameters
    ----------
    returns          : array of periodic returns
    periods_per_year : 52 (weekly) | 12 (monthly) | 4 (quarterly)
                       Default is 52 to match the paper's weekly strategy.
    risk_free        : annualized risk-free rate (default 0)

    Returns
    -------
    float — annualized Sharpe ratio, 0.0 if std is zero or NaN
    """
    excess = returns - risk_free / periods_per_year
    mean   = excess.mean()
    std    = excess.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(mean / std * np.sqrt(periods_per_year))


def decile_portfolio_returns(
    y_proba:          np.ndarray,
    returns:          np.ndarray,
    n_deciles:        int = 10,
    periods_per_year: int = 52,
) -> pd.DataFrame:
    """
    Sorts samples into deciles by predicted up-probability and computes
    mean return and annualized Sharpe ratio per decile.

    Parameters
    ----------
    y_proba          : predicted P(up), shape (N,) — used for sorting
    returns          : realized returns, shape (N,)
    n_deciles        : number of deciles (default 10, as in the paper)
    periods_per_year : annualization factor for Sharpe ratio

    Returns
    -------
    DataFrame indexed by decile (1=lowest prob, 10=highest prob)
    with columns: mean_return, std, sharpe, n
    """
    df = pd.DataFrame({
        "prob_up": y_proba,
        "return":  returns,
    })
    df["decile"] = pd.qcut(df["prob_up"], n_deciles, labels=False) + 1

    results = []
    for d in range(1, n_deciles + 1):
        subset = df[df["decile"] == d]["return"].values
        results.append({
            "decile":      d,
            "mean_return": subset.mean(),
            "std":         subset.std(),
            "sharpe":      annualized_sharpe(subset, periods_per_year),
            "n":           len(subset),
        })

    return pd.DataFrame(results).set_index("decile")


def hl_sharpe(
    decile_df:        pd.DataFrame,
    y_proba:          np.ndarray,
    returns:          np.ndarray,
    n_deciles:        int   = 10,
    periods_per_year: int   = 52,
) -> float:
    """
    Computes the annualized Sharpe ratio of the H-L (High minus Low) strategy,
    defined as the return of the top decile minus the return of the bottom decile.

    This matches the paper's main performance metric.

    Parameters
    ----------
    decile_df        : output of decile_portfolio_returns()
    y_proba          : predicted P(up), shape (N,) — used to identify deciles
    returns          : realized returns, shape (N,)
    n_deciles        : number of deciles
    periods_per_year : annualization factor

    Returns
    -------
    float — annualized Sharpe ratio of the H-L strategy
    """
    df = pd.DataFrame({"prob_up": y_proba, "return": returns})
    df["decile"] = pd.qcut(df["prob_up"], n_deciles, labels=False) + 1

    high_returns = df[df["decile"] == n_deciles]["return"].values
    low_returns  = df[df["decile"] == 1]["return"].values

    # H-L return series: long top decile, short bottom decile
    min_len  = min(len(high_returns), len(low_returns))
    hl_rets  = high_returns[:min_len] - low_returns[:min_len]

    return annualized_sharpe(hl_rets, periods_per_year)


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------
def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """
    Builds a comparison DataFrame from per-model metric dictionaries.

    Parameters
    ----------
    results : {model_name: {"accuracy": ..., "f1": ..., "auc": ..., "brier": ...}}

    Returns
    -------
    DataFrame sorted by AUC (descending), indexed by model name
    """
    rows = []
    for name, metrics in results.items():
        rows.append({"model": name, **metrics})
    df = pd.DataFrame(rows).set_index("model")
    return df.sort_values("auc", ascending=False)


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
def plot_decile_returns(
    decile_df:  pd.DataFrame,
    model_name: str = "Model",
    save_path:  Optional[str] = None,
) -> None:
    """
    Bar chart of mean return and annualized Sharpe ratio per decile.

    Parameters
    ----------
    decile_df  : output of decile_portfolio_returns()
    model_name : model name for the plot title
    save_path  : optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Mean return per decile
    colors_ret = ["#d73027" if v < 0 else "#1a9850"
                  for v in decile_df["mean_return"]]
    axes[0].bar(
        decile_df.index, decile_df["mean_return"],
        color=colors_ret, edgecolor="white",
    )
    axes[0].set_title(f"{model_name} — Mean return per decile")
    axes[0].set_xlabel("Decile (1=lowest prob, 10=highest prob)")
    axes[0].set_ylabel("Mean return")
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].grid(axis="y", alpha=0.3)

    # Sharpe ratio per decile
    colors_sr = ["#d73027" if v < 0 else "#1a9850"
                 for v in decile_df["sharpe"]]
    axes[1].bar(
        decile_df.index, decile_df["sharpe"],
        color=colors_sr, edgecolor="white",
    )
    axes[1].set_title(f"{model_name} — Annualized Sharpe ratio per decile")
    axes[1].set_xlabel("Decile")
    axes[1].set_ylabel("Sharpe ratio")
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    save_path:     Optional[str] = None,
) -> None:
    """
    Heatmap comparing ML metrics across models.

    Parameters
    ----------
    comparison_df : output of compare_models()
    save_path     : optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(8, max(3, len(comparison_df) * 0.8)))

    data = comparison_df.values.astype(float)
    im   = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(comparison_df.columns)))
    ax.set_xticklabels(comparison_df.columns, fontsize=10)
    ax.set_yticks(range(len(comparison_df)))
    ax.set_yticklabels(comparison_df.index, fontsize=10)

    for i in range(len(comparison_df)):
        for j in range(len(comparison_df.columns)):
            ax.text(
                j, i, f"{data[i, j]:.3f}",
                ha="center", va="center",
                color="black", fontsize=9, fontweight="bold",
            )

    ax.set_title("Model Comparison", fontsize=13, pad=15)
    plt.colorbar(im, ax=ax, fraction=0.03)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_cumulative_returns(
    strategies: dict[str, np.ndarray],
    save_path:  Optional[str] = None,
) -> None:
    """
    Plots cumulative returns for multiple H-L strategies.

    Parameters
    ----------
    strategies : {model_name: array of periodic H-L returns}
    save_path  : optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    colors  = plt.cm.tab10.colors

    for i, (name, rets) in enumerate(strategies.items()):
        cum = np.cumprod(1 + rets) - 1
        ax.plot(cum, label=name, color=colors[i % 10], linewidth=1.8)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Cumulative Returns — H-L Strategies", fontsize=13)
    ax.set_xlabel("Period")
    ax.set_ylabel("Cumulative return")
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_gradcam_overlay(
    image:     np.ndarray,
    cam:       np.ndarray,
    label:     int,
    pred:      int,
    save_path: Optional[str] = None,
) -> None:
    """
    Overlays a Grad-CAM heatmap on the original OHLC image.

    Parameters
    ----------
    image     : (H, W) original image, values in [0, 1]
    cam       : (H, W) Grad-CAM heatmap, values in [0, 1]
    label     : ground truth label (0=DOWN, 1=UP)
    pred      : predicted label (0=DOWN, 1=UP)
    save_path : optional path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].imshow(image, cmap="gray", aspect="auto", interpolation="nearest")
    axes[0].set_title("Original OHLC image")
    axes[0].axis("off")

    axes[1].imshow(cam, cmap="jet", aspect="auto", interpolation="nearest")
    axes[1].set_title("Grad-CAM heatmap")
    axes[1].axis("off")

    # Overlay: blend grayscale image with colormap heatmap
    rgb         = np.stack([image, image, image], axis=-1)
    cam_colored = cm.jet(cam)[..., :3]
    overlay     = 0.5 * rgb + 0.5 * cam_colored
    correct     = "✓" if label == pred else "✗"
    color       = "green" if label == pred else "red"
    axes[2].imshow(overlay, aspect="auto", interpolation="nearest")
    axes[2].set_title(
        f"Overlay — True: {'UP' if label == 1 else 'DOWN'} | "
        f"Pred: {'UP' if pred == 1 else 'DOWN'} {correct}",
        color=color,
    )
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_confusion_matrix(
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    model_name: str = "Model",
    save_path:  Optional[str] = None,
) -> None:
    """
    Plots and optionally saves a confusion matrix.

    Parameters
    ----------
    y_true     : ground truth labels
    y_pred     : predicted labels
    model_name : model name for the plot title
    save_path  : optional path to save the figure
    """
    cm_arr = confusion_matrix(y_true, y_pred)
    disp   = ConfusionMatrixDisplay(cm_arr, display_labels=["DOWN", "UP"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Full evaluation report
# ---------------------------------------------------------------------------
def full_evaluation(
    model_results: dict[str, dict],
    save_dir:      Optional[str] = None,
    periods_per_year: int = 52,
) -> pd.DataFrame:
    """
    Generates a complete evaluation report: comparison table + figures.

    Parameters
    ----------
    model_results : {
        "MLP":  {
            "y_true":  np.ndarray,
            "y_pred":  np.ndarray,
            "y_proba": np.ndarray,
            "returns": np.ndarray,  # optional — realized returns for Sharpe
        },
        "LSTM": {...},
        "CNN":  {...},
    }
    save_dir         : directory to save all figures (optional)
    periods_per_year : annualization factor (52=weekly, 12=monthly, 4=quarterly)

    Returns
    -------
    DataFrame with ML metrics comparison across models
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    ml_metrics = {}
    hl_sharpes = {}

    for name, res in model_results.items():
        # ML metrics
        ml_metrics[name] = compute_ml_metrics(
            res["y_true"], res["y_pred"], res["y_proba"]
        )
        print_classification_report(res["y_true"], res["y_pred"], model_name=name)

        # Financial metrics (only if realized returns are provided)
        if "returns" in res:
            dec = decile_portfolio_returns(
                res["y_proba"], res["returns"],
                periods_per_year=periods_per_year,
            )
            hl_sharpes[name] = hl_sharpe(
                dec, res["y_proba"], res["returns"],
                periods_per_year=periods_per_year,
            )
            plot_decile_returns(
                dec,
                model_name=name,
                save_path=os.path.join(save_dir, f"{name}_deciles.png")
                          if save_dir else None,
            )
            plot_confusion_matrix(
                res["y_true"], res["y_pred"],
                model_name=name,
                save_path=os.path.join(save_dir, f"{name}_confusion.png")
                          if save_dir else None,
            )

    # Global comparison table
    cmp_df = compare_models(ml_metrics)
    print("\n=== Global Model Comparison ===")
    print(cmp_df.to_string())

    if hl_sharpes:
        print("\n=== H-L Annualized Sharpe Ratios ===")
        for name, sr in hl_sharpes.items():
            print(f"  {name:10s} : {sr:.4f}")

    plot_model_comparison(
        cmp_df,
        save_path=os.path.join(save_dir, "model_comparison.png")
                  if save_dir else None,
    )

    return cmp_df