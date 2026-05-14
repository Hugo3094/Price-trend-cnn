"""
metrics.py
----------
Full model evaluation and comparison.

Financial metrics (paper):
  - Annualised Sharpe ratio for decile H-L strategies
  - Mean return per decile

ML metrics:
  - Accuracy, F1, AUC-ROC, Brier score

Visualisations:
  - Cumulative return curves
  - Model comparison heatmap
  - Grad-CAM overlay
"""

import logging
import os
import numpy as np
import pandas as pd
from typing import Optional
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
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
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict:
    """
    Parameters
    ----------
    y_true  : ground-truth labels (0 or 1)
    y_pred  : predicted labels (0 or 1)
    y_proba : class-1 probabilities — shape (N,)

    Returns
    -------
    dict with accuracy, f1, auc, brier
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_proba),
        "brier": brier_score_loss(y_true, y_proba),
    }


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> None:
    logger.info("=" * 40)
    logger.info("  %s", model_name)
    logger.info("=" * 40)
    logger.info("\n%s", classification_report(y_true, y_pred, target_names=["DOWN", "UP"]))


# ---------------------------------------------------------------------------
# Financial metrics
# ---------------------------------------------------------------------------
def decile_portfolio_returns(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    returns: np.ndarray,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """
    Sorts stocks into n deciles by predicted up-probability
    and computes the mean return per decile.

    Parameters
    ----------
    y_true    : ground-truth labels
    y_proba   : predicted P(up) — used for sorting
    returns   : realised returns (float)
    n_deciles : number of deciles

    Returns
    -------
    DataFrame with mean return and Sharpe per decile
    """
    df = pd.DataFrame({"prob_up": y_proba, "return": returns})
    df["decile"] = pd.qcut(df["prob_up"], n_deciles, labels=False) + 1

    results = []
    for d in range(1, n_deciles + 1):
        subset = df[df["decile"] == d]["return"]
        results.append({
            "decile": d,
            "mean_return": subset.mean(),
            "std": subset.std(),
            "sharpe": annualized_sharpe(subset.values),
            "n": len(subset),
        })

    return pd.DataFrame(results).set_index("decile")


def annualized_sharpe(
    returns: np.ndarray,
    periods_per_year: int = 52,
    risk_free: float = 0.0,
) -> float:
    """
    Annualised Sharpe ratio.

    Parameters
    ----------
    periods_per_year : 52 (weekly) | 12 (monthly) | 4 (quarterly)
    """
    excess = returns - risk_free / periods_per_year
    mean = excess.mean()
    std = excess.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(mean / std * np.sqrt(periods_per_year))


def hl_sharpe(decile_df: pd.DataFrame) -> float:
    """Sharpe ratio of the H-L strategy (decile 10 - decile 1)."""
    high = decile_df.loc[10, "mean_return"] if 10 in decile_df.index else 0
    low = decile_df.loc[1, "mean_return"] if 1 in decile_df.index else 0
    return high - low


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------
def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """
    Parameters
    ----------
    results : {model_name: {"accuracy": ..., "f1": ..., "auc": ..., "brier": ...}}

    Returns
    -------
    Comparison DataFrame sorted by AUC descending
    """
    rows = [{"model": name, **metrics} for name, metrics in results.items()]
    df = pd.DataFrame(rows).set_index("model")
    return df.sort_values("auc", ascending=False)


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------
def plot_decile_returns(
    decile_df: pd.DataFrame,
    model_name: str = "Model",
    save_path: Optional[str] = None,
) -> None:
    """Bar chart of mean returns and Sharpe ratio per decile."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#d73027" if v < 0 else "#1a9850" for v in decile_df["mean_return"]]
    axes[0].bar(decile_df.index, decile_df["mean_return"], color=colors, edgecolor="white")
    axes[0].set_title(f"{model_name} — Mean return per decile")
    axes[0].set_xlabel("Decile (1=low prob, 10=high prob)")
    axes[0].set_ylabel("Mean return")
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].grid(axis="y", alpha=0.3)

    colors_sr = ["#d73027" if v < 0 else "#1a9850" for v in decile_df["sharpe"]]
    axes[1].bar(decile_df.index, decile_df["sharpe"], color=colors_sr, edgecolor="white")
    axes[1].set_title(f"{model_name} — Annualised Sharpe ratio per decile")
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
    save_path: Optional[str] = None,
) -> None:
    """Heatmap comparing ML metrics across models."""
    fig, ax = plt.subplots(figsize=(8, max(3, len(comparison_df) * 0.8)))

    data = comparison_df.values.astype(float)
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(comparison_df.columns)))
    ax.set_xticklabels(comparison_df.columns, fontsize=10)
    ax.set_yticks(range(len(comparison_df)))
    ax.set_yticklabels(comparison_df.index, fontsize=10)

    for i in range(len(comparison_df)):
        for j in range(len(comparison_df.columns)):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center",
                    color="black", fontsize=9, fontweight="bold")

    ax.set_title("Model comparison", fontsize=13, pad=15)
    plt.colorbar(im, ax=ax, fraction=0.03)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_cumulative_returns(
    strategies: dict[str, np.ndarray],
    save_path: Optional[str] = None,
) -> None:
    """
    Cumulative return curves for multiple strategies.

    Parameters
    ----------
    strategies : {model_name: array of daily H-L returns}
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10.colors

    for i, (name, rets) in enumerate(strategies.items()):
        cum = np.cumprod(1 + rets) - 1
        ax.plot(cum, label=name, color=colors[i % 10], linewidth=1.8)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Cumulative returns — H-L strategies", fontsize=13)
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
    image: np.ndarray,
    cam: np.ndarray,
    label: int,
    pred: int,
    save_path: Optional[str] = None,
) -> None:
    """
    Overlays the Grad-CAM heatmap on the original OHLC image.

    Parameters
    ----------
    image : (H, W)  — original image [0,1]
    cam   : (H, W)  — heatmap [0,1]
    label : ground-truth label
    pred  : predicted label
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].imshow(image, cmap="gray", aspect="auto", interpolation="nearest")
    axes[0].set_title("Original OHLC image")
    axes[0].axis("off")

    axes[1].imshow(cam, cmap="jet", aspect="auto", interpolation="nearest")
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    rgb = np.stack([image, image, image], axis=-1)
    cam_colored = cm.jet(cam)[..., :3]
    overlay = 0.5 * rgb + 0.5 * cam_colored
    axes[2].imshow(overlay, aspect="auto", interpolation="nearest")
    correct = "✓" if label == pred else "✗"
    color = "green" if label == pred else "red"
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
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None,
) -> None:
    cm_arr = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm_arr, display_labels=["DOWN", "UP"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion matrix — {model_name}")
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
    save_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generates a full report: comparison table + figures.

    Parameters
    ----------
    model_results : {
        "MLP":  {"y_true": ..., "y_pred": ..., "y_proba": ..., "returns": ...},
        "LSTM": {...},
        "CNN":  {...},
    }
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    ml_metrics = {}
    for name, res in model_results.items():
        ml_metrics[name] = compute_ml_metrics(
            res["y_true"], res["y_pred"], res["y_proba"]
        )
        print_classification_report(res["y_true"], res["y_pred"], model_name=name)

        if "returns" in res:
            dec = decile_portfolio_returns(res["y_true"], res["y_proba"], res["returns"])
            plot_decile_returns(
                dec,
                model_name=name,
                save_path=os.path.join(save_dir, f"{name}_deciles.png") if save_dir else None,
            )

    cmp_df = compare_models(ml_metrics)
    logger.info("=== Global comparison ===")
    logger.info("\n%s", cmp_df.to_string())

    plot_model_comparison(
        cmp_df,
        save_path=os.path.join(save_dir, "model_comparison.png") if save_dir else None,
    )

    return cmp_df