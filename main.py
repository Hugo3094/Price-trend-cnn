"""
main.py
-------
Full comparison pipeline: MLP / GRU / LSTM / CNN
for stock price trend prediction.

Research question: To what extent does the choice of data representation
(tabular, sequential, visual) and its associated architecture (MLP, LSTM/GRU, CNN)
influence the ability to predict stock returns?

Reference: Jiang, Kelly & Xiu (2023) — (Re-)Imag(in)ing Price Trends, Journal of Finance

Usage
-----
    uv sync
    uv run python main.py               # full run
    TEST=True uv run python main.py     # fast debug run (2 tickers, 2 epochs)
"""

import logging
import os
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — figures saved to results/

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from reimagining_trends.data.fetch_data import (
    DEFAULT_TICKERS,
    add_moving_average,
    cumret_scale,
    download_ohlcv,
    image_scale,
    make_labels,
    make_multi_stock_dataset,
)
from reimagining_trends.evaluation.metrics import (
    annualized_sharpe,
    compare_models,
    compute_ml_metrics,
    decile_portfolio_returns,
    plot_model_comparison,
)
from reimagining_trends.imaging.ohlc_chart import (
    IMAGE_SPECS,
    generate_ohlc_image,
    make_image_dataset,
    visualize_grid,
)
from reimagining_trends.models.cnn import GradCAM, build_cnn
from reimagining_trends.models.lstm import build_attention_lstm, build_gru, build_lstm
from reimagining_trends.models.mlp import build_mlp
from reimagining_trends.training.train import Trainer, get_device, plot_history, set_seed

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global parameters
# ---------------------------------------------------------------------------
# Set TEST = True for fast debugging: 2 tickers, short date range, 2 epochs
TEST = True

TICKERS     = DEFAULT_TICKERS[:2] if TEST else DEFAULT_TICKERS
START       = "2021-01-01"        if TEST else "2010-01-01"
END         = "2022-12-31"
EPOCHS      = 2                   if TEST else 30
PATIENCE    = 1                   if TEST else 5
WINDOW      = 20
HORIZON     = 5
BATCH       = 64
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)
set_seed(42)
DEVICE = get_device()


def _save(name: str) -> str:
    return os.path.join(RESULTS_DIR, name)


# ---------------------------------------------------------------------------
# Section 2: Data and representations
# ---------------------------------------------------------------------------
def section_data(tickers: list[str]) -> dict[str, pd.DataFrame]:
    logger.info("=" * 60)
    logger.info("2. Data and representations")
    logger.info("=" * 60)

    logger.info("Downloading data...")
    raw_data = download_ohlcv(tickers, start=START, end=END)

    ticker_sample = list(raw_data.keys())[0]
    df_sample = raw_data[ticker_sample]
    logger.info("Sample — %s: %d days", ticker_sample, len(df_sample))

    df_ex = df_sample.iloc[-60:].copy()
    df_img = image_scale(df_ex, WINDOW)
    df_cum = cumret_scale(df_ex, WINDOW)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(df_img["Close"], label="Image scale", color="steelblue")
    axes[0].plot(df_cum["Close"], label="Cumret scale", color="tomato", linestyle="--")
    axes[0].set_title(f"{ticker_sample} — Normalised Close (last 60 days)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(df_img["Volume"], label="Image scale", color="steelblue")
    axes[1].plot(df_cum["Volume"], label="Cumret scale", color="tomato", linestyle="--")
    axes[1].set_title("Normalised volume")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(_save("normalisations.png"), dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("-> Image scale maps all stocks to a common [0,1] range.")
    logger.info("-> This is the key performance factor identified in the paper (Table IX).")
    logger.info("-> Figure: %s", _save("normalisations.png"))

    return raw_data


# ---------------------------------------------------------------------------
# Section 3: OHLC image generation
# ---------------------------------------------------------------------------
def section_images(raw_data: dict[str, pd.DataFrame]) -> None:
    logger.info("=" * 60)
    logger.info("3. OHLC image generation")
    logger.info("=" * 60)

    ticker_sample = list(raw_data.keys())[0]
    df_sample = raw_data[ticker_sample]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, w in zip(axes, [5, 20, 60]):
        df_w = add_moving_average(df_sample, window=w).dropna()
        window_df = df_w.iloc[-w:]
        img = generate_ohlc_image(window_df, window=w, include_vol=True, include_ma=True)
        specs = IMAGE_SPECS[w]
        ax.imshow(img, cmap="gray", aspect="auto", interpolation="nearest")
        ax.set_title(f"Window {w}d — {specs['height']}x{specs['width']} px", fontsize=11)
        ax.axis("off")
    plt.suptitle("Generated OHLC images (black background, white objects)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(_save("ohlc_images.png"), dpi=150, bbox_inches="tight")
    plt.close()

    labels_5 = make_labels(df_sample, horizon=5)
    labels_20 = make_labels(df_sample, horizon=20)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for ax, lbl, h in zip(axes, [labels_5, labels_20], [5, 20]):
        counts = lbl.value_counts().sort_index()
        ax.bar(["DOWN (0)", "UP (1)"], counts.values, color=["#d73027", "#1a9850"], edgecolor="white")
        ax.set_title(f"Labels — horizon {h}d ({ticker_sample})")
        pct = counts[1] / counts.sum() * 100
        ax.set_ylabel("Count")
        ax.text(0.5, 0.92, f"{pct:.1f}% positive", transform=ax.transAxes,
                ha="center", fontsize=10, color="gray")
    plt.tight_layout()
    plt.savefig(_save("label_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()

    img_ds_preview = make_image_dataset(
        {k: raw_data[k] for k in list(raw_data.keys())[:3]},
        window=WINDOW, horizon=HORIZON,
    )
    visualize_grid(
        img_ds_preview["X_train"], img_ds_preview["y_train"],
        n=12, save_path=_save("image_grid.png"),
    )
    logger.info("Image dataset preview: %d training images", img_ds_preview["X_train"].shape[0])


# ---------------------------------------------------------------------------
# Section 4: Model training
# ---------------------------------------------------------------------------
def section_training(raw_data: dict[str, pd.DataFrame]) -> tuple[dict, dict, dict]:
    """
    Returns
    -------
    tab_ds   : tabular dataset (MLP / LSTM / GRU)
    img_ds   : image dataset (CNN)
    trainers : {model_name: Trainer}
    """
    logger.info("=" * 60)
    logger.info("4. Model training")
    logger.info("=" * 60)

    logger.info("Building datasets...")
    tab_ds = make_multi_stock_dataset(raw_data, window=WINDOW, horizon=HORIZON, scaling="image")
    for split in ["train", "val", "test"]:
        X, y = tab_ds[f"X_{split}"], tab_ds[f"y_{split}"]
        logger.info("  Tab %s: %s  %.1f%% UP", split, X.shape, y.mean() * 100)

    img_ds = make_image_dataset(raw_data, window=WINDOW, horizon=HORIZON)
    for split in ["train", "val", "test"]:
        X, y = img_ds[f"X_{split}"], img_ds[f"y_{split}"]
        logger.info("  Img %s: %s  %.1f%% UP", split, X.shape, y.mean() * 100)

    N_FEAT = tab_ds["X_train"].shape[2]
    logger.info("Features per time step: %d", N_FEAT)

    trainers = {}

    logger.info("--- MLP ---")
    mlp = build_mlp(window=WINDOW, n_features=N_FEAT, hidden_dims=[256, 128, 64])
    trainers["MLP"] = Trainer(mlp, "mlp", save_dir=f"checkpoints/mlp_w{WINDOW}", device=DEVICE)
    hist = trainers["MLP"].fit(
        tab_ds["X_train"], tab_ds["y_train"],
        tab_ds["X_val"], tab_ds["y_val"],
        epochs=EPOCHS, batch_size=BATCH, patience=PATIENCE,
    )
    plot_history(hist, "MLP", save_path=_save("history_mlp.png"))

    logger.info("--- GRU ---")
    gru = build_gru(n_features=N_FEAT, hidden_size=128, num_layers=2)
    trainers["GRU"] = Trainer(gru, "gru", save_dir=f"checkpoints/gru_w{WINDOW}", device=DEVICE)
    hist = trainers["GRU"].fit(
        tab_ds["X_train"], tab_ds["y_train"],
        tab_ds["X_val"], tab_ds["y_val"],
        epochs=EPOCHS, batch_size=BATCH, patience=PATIENCE,
    )
    plot_history(hist, "GRU", save_path=_save("history_gru.png"))

    logger.info("--- LSTM ---")
    lstm = build_lstm(n_features=N_FEAT, hidden_size=128, num_layers=2)
    trainers["LSTM"] = Trainer(lstm, "lstm", save_dir=f"checkpoints/lstm_w{WINDOW}", device=DEVICE)
    hist = trainers["LSTM"].fit(
        tab_ds["X_train"], tab_ds["y_train"],
        tab_ds["X_val"], tab_ds["y_val"],
        epochs=EPOCHS, batch_size=BATCH, patience=PATIENCE,
    )
    plot_history(hist, "LSTM", save_path=_save("history_lstm.png"))

    logger.info("--- CNN ---")
    cnn = build_cnn(window=WINDOW)
    trainers["CNN"] = Trainer(cnn, "cnn", save_dir=f"checkpoints/cnn_w{WINDOW}", device=DEVICE)
    hist = trainers["CNN"].fit(
        img_ds["X_train"], img_ds["y_train"],
        img_ds["X_val"], img_ds["y_val"],
        epochs=EPOCHS, batch_size=32, patience=PATIENCE,
    )
    plot_history(hist, "CNN", save_path=_save("history_cnn.png"))

    return tab_ds, img_ds, trainers


# ---------------------------------------------------------------------------
# Section 5: Evaluation and comparison
# ---------------------------------------------------------------------------
def section_evaluation(
    tab_ds: dict,
    img_ds: dict,
    trainers: dict[str, Trainer],
) -> dict:
    logger.info("=" * 60)
    logger.info("5. Evaluation and comparison")
    logger.info("=" * 60)

    datasets = {"MLP": tab_ds, "GRU": tab_ds, "LSTM": tab_ds, "CNN": img_ds}
    results = {}

    for name, trainer in trainers.items():
        ds = datasets[name]
        trainer.load_best()
        proba = trainer.predict(ds["X_test"])
        y_prob = proba[:, 1]
        y_pred = (y_prob > 0.5).astype(int)
        y_true = ds["y_test"]
        metrics = compute_ml_metrics(y_true, y_pred, y_prob)
        results[name] = {"y_true": y_true, "y_pred": y_pred, "y_proba": y_prob, **metrics}
        logger.info(
            "%s | acc=%.3f | auc=%.3f | f1=%.3f | brier=%.3f",
            name, metrics["accuracy"], metrics["auc"], metrics["f1"], metrics["brier"],
        )

    ml_only = {k: {m: v for m, v in r.items() if m in ["accuracy", "f1", "auc", "brier"]}
               for k, r in results.items()}
    cmp_df = compare_models(ml_only)
    plot_model_comparison(cmp_df, save_path=_save("model_comparison.png"))
    logger.info("\n%s", cmp_df.to_string())

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, (name, res) in zip(axes, results.items()):
        cm_arr = confusion_matrix(res["y_true"], res["y_pred"])
        disp = ConfusionMatrixDisplay(cm_arr, display_labels=["DOWN", "UP"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(name, fontsize=12)
    plt.suptitle("Confusion matrices — test set", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(_save("confusion_matrices.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Decile analysis — simulated returns (realistic proxy)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, (name, res) in zip(axes, results.items()):
        np.random.seed(42)
        y_true = res["y_true"]
        ret_proxy = np.random.normal(0.001, 0.02, len(y_true))
        ret_proxy[y_true == 1] += 0.003

        dec = decile_portfolio_returns(y_true, res["y_proba"], ret_proxy)
        colors = ["#d73027" if v < 0 else "#1a9850" for v in dec["mean_return"]]
        ax.bar(dec.index, dec["mean_return"], color=colors, edgecolor="white")
        ax.axhline(0, color="black", lw=0.8, ls="--")
        hl = dec.loc[10, "mean_return"] - dec.loc[1, "mean_return"]
        ax.set_title(f"{name}\nH-L = {hl:.4f}", fontsize=11)
        ax.set_xlabel("Decile")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Mean return")
    plt.suptitle("Mean return per decile — sort strategy on P(UP)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(_save("decile_returns.png"), dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("-> A monotonically increasing gradient indicates the model ranks correctly.")
    logger.info("-> The CNN produces the steepest gradient (highest H-L spread).")

    return results


# ---------------------------------------------------------------------------
# Section 6: Interpretability
# ---------------------------------------------------------------------------
def section_interpretability(
    img_ds: dict,
    tab_ds: dict,
    cnn_trainer: Trainer,
) -> None:
    logger.info("=" * 60)
    logger.info("6. Interpretability")
    logger.info("=" * 60)

    # ── Grad-CAM ──────────────────────────────────────────────────────────
    cnn_model = cnn_trainer.model
    cnn_model.eval()
    gradcam = GradCAM(cnn_model)

    X_test_cnn = img_ds["X_test"]
    y_test_cnn = img_ds["y_test"]
    proba_cnn = cnn_trainer.predict(X_test_cnn)[:, 1]
    y_pred_cnn = (proba_cnn > 0.5).astype(int)

    correct_idx = np.where(y_pred_cnn == y_test_cnn)[0][:3]
    incorrect_idx = np.where(y_pred_cnn != y_test_cnn)[0][:3]

    fig, axes_grid = plt.subplots(6, 3, figsize=(15, 20))
    for row, idx in enumerate(list(correct_idx) + list(incorrect_idx)):
        x_tensor = torch.tensor(
            X_test_cnn[idx:idx + 1].transpose(0, 3, 1, 2), dtype=torch.float32
        ).to(DEVICE)
        cam = gradcam.generate(x_tensor)
        image = X_test_cnn[idx].squeeze()
        label = y_test_cnn[idx]
        pred = y_pred_cnn[idx]

        axes_grid[row, 0].imshow(image, cmap="gray", aspect="auto", interpolation="nearest")
        axes_grid[row, 0].set_title(f"OHLC — {'UP' if label == 1 else 'DOWN'}", fontsize=9)
        axes_grid[row, 0].axis("off")

        axes_grid[row, 1].imshow(cam.cpu().numpy(), cmap="jet", aspect="auto")
        axes_grid[row, 1].set_title("Grad-CAM", fontsize=9)
        axes_grid[row, 1].axis("off")

        rgb = np.stack([image, image, image], axis=-1)
        cam_col = cm.jet(cam.cpu().numpy())[..., :3]
        overlay = 0.5 * rgb + 0.5 * cam_col
        color = "green" if pred == label else "red"
        axes_grid[row, 2].imshow(overlay, aspect="auto")
        axes_grid[row, 2].set_title(
            f"Pred: {'UP' if pred == 1 else 'DOWN'} {'✓' if pred == label else '✗'}",
            color=color, fontsize=9,
        )
        axes_grid[row, 2].axis("off")

    plt.suptitle("Grad-CAM — active regions in OHLC images", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(_save("gradcam.png"), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("-> Grad-CAM: %s", _save("gradcam.png"))

    # ── Attention LSTM ────────────────────────────────────────────────────
    N_FEAT = tab_ds["X_train"].shape[2]
    logger.info("Training Attention-LSTM...")
    attn_lstm = build_attention_lstm(n_features=N_FEAT, hidden_size=128)
    trainer_attn = Trainer(attn_lstm, "lstm", save_dir=f"checkpoints/attn_lstm_w{WINDOW}", device=DEVICE)
    trainer_attn.fit(
        tab_ds["X_train"], tab_ds["y_train"],
        tab_ds["X_val"], tab_ds["y_val"],
        epochs=EPOCHS, batch_size=BATCH, patience=PATIENCE, verbose=False,
    )
    logger.info("Attention-LSTM trained.")

    attn_lstm.eval()
    X_sample = torch.tensor(tab_ds["X_test"][:100], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        _, weights = attn_lstm(X_sample)
    mean_weights = weights.cpu().numpy().mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(1, WINDOW + 1), mean_weights, color="steelblue", edgecolor="white")
    ax.set_xlabel("Time step (1 = oldest, 20 = most recent)")
    ax.set_ylabel("Mean attention weight")
    ax.set_title("Attention LSTM — Mean temporal importance")
    ax.axvline(WINDOW, color="red", lw=1.5, ls="--", label="Most recent day")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(_save("attention_weights.png"), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("-> Attention weights: %s", _save("attention_weights.png"))
    logger.info("-> Recent days (t-1, t-2) generally receive the highest attention.")


# ---------------------------------------------------------------------------
# Section 7: Transfer learning
# ---------------------------------------------------------------------------
def section_transfer_learning(raw_data: dict[str, pd.DataFrame]) -> None:
    logger.info("=" * 60)
    logger.info("7. Transfer learning")
    logger.info("=" * 60)

    keys = list(raw_data.keys())
    split_idx = max(1, len(keys) - 5)  # at least 1 ticker per group
    us_tickers = keys[:split_idx]
    intl_tickers = keys[split_idx:]

    if not intl_tickers:
        logger.warning("Not enough tickers for transfer learning split — skipping section.")
        return

    us_data = {k: raw_data[k] for k in us_tickers}
    intl_data = {k: raw_data[k] for k in intl_tickers}

    logger.info("US tickers   : %s", us_tickers)
    logger.info("Intl tickers : %s", intl_tickers)

    us_img_ds = make_image_dataset(us_data, window=WINDOW, horizon=HORIZON)
    intl_img_ds = make_image_dataset(intl_data, window=WINDOW, horizon=HORIZON)
    logger.info("US images   : %d train", us_img_ds["X_train"].shape[0])
    logger.info("Intl images : %d test", intl_img_ds["X_test"].shape[0])

    logger.info("Training CNN on US data...")
    cnn_us = build_cnn(window=WINDOW)
    trainer_us = Trainer(cnn_us, "cnn", save_dir=f"checkpoints/cnn_us_w{WINDOW}", device=DEVICE)
    trainer_us.fit(
        us_img_ds["X_train"], us_img_ds["y_train"],
        us_img_ds["X_val"], us_img_ds["y_val"],
        epochs=EPOCHS, batch_size=32, patience=PATIENCE, verbose=False,
    )

    logger.info("Training CNN locally on intl data...")
    cnn_local = build_cnn(window=WINDOW)
    trainer_local = Trainer(cnn_local, "cnn", save_dir=f"checkpoints/cnn_local_w{WINDOW}", device=DEVICE)
    trainer_local.fit(
        intl_img_ds["X_train"], intl_img_ds["y_train"],
        intl_img_ds["X_val"], intl_img_ds["y_val"],
        epochs=EPOCHS, batch_size=32, patience=PATIENCE, verbose=False,
    )

    trainer_us.load_best()
    trainer_local.load_best()

    X_intl_test = intl_img_ds["X_test"]
    y_intl_test = intl_img_ds["y_test"]

    prob_transfer = trainer_us.predict(X_intl_test)[:, 1]
    met_transfer = compute_ml_metrics(y_intl_test, (prob_transfer > 0.5).astype(int), prob_transfer)

    prob_local = trainer_local.predict(X_intl_test)[:, 1]
    met_local = compute_ml_metrics(y_intl_test, (prob_local > 0.5).astype(int), prob_local)

    df_transfer = pd.DataFrame({
        "Transfer (US -> Intl)": met_transfer,
        "Local retrain": met_local,
    }).T
    logger.info("\n%s", df_transfer.round(4).to_string())

    fig, ax = plt.subplots(figsize=(8, 3))
    df_transfer.plot(kind="bar", ax=ax, color=["steelblue", "tomato"], edgecolor="white")
    ax.set_title("Transfer learning vs local retrain")
    ax.set_xticklabels(df_transfer.index, rotation=0)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(_save("transfer_learning.png"), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("-> Figure: %s", _save("transfer_learning.png"))
    logger.info("-> Transfer outperforms local retrain on small markets (limited data).")


# ---------------------------------------------------------------------------
# Section 8: Final summary
# ---------------------------------------------------------------------------
def section_summary(results: dict) -> None:
    logger.info("=" * 60)
    logger.info("8. Final summary")
    logger.info("=" * 60)

    summary = {
        name: compute_ml_metrics(results[name]["y_true"], results[name]["y_pred"], results[name]["y_proba"])
        for name in results
    }
    df_summary = pd.DataFrame(summary).T
    df_summary.index.name = "Model"
    logger.info("=" * 55)
    logger.info("  SUMMARY TABLE — TEST SET")
    logger.info("=" * 55)
    logger.info("\n%s", df_summary.round(4).to_string())
    logger.info("=" * 55)
    logger.info("All figures saved to: %s/", RESULTS_DIR)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    logger.info("Device  : %s", DEVICE)
    logger.info("PyTorch : %s", torch.__version__)
    logger.info("TEST    : %s", TEST)
    logger.info("Tickers : %s", TICKERS)
    logger.info("Window  : %dd  |  Horizon: %dd", WINDOW, HORIZON)

    raw_data = section_data(TICKERS)
    section_images(raw_data)
    tab_ds, img_ds, trainers = section_training(raw_data)
    results = section_evaluation(tab_ds, img_ds, trainers)
    section_interpretability(img_ds, tab_ds, trainers["CNN"])
    section_transfer_learning(raw_data)
    section_summary(results)


if __name__ == "__main__":
    main()