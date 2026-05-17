"""
backtester.py
-------------
Backtester — orchestrates the full backtest pipeline:
  aux data → benchmark → betas → scores → portfolio → simulate → metrics → plots.
"""

import logging
import os
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reimagining_trends.backtest.metrics import compute_backtest_metrics
from reimagining_trends.backtest.portfolio import construct_ls_portfolio, construct_lo_portfolio
from reimagining_trends.backtest.simulator import simulate_portfolio
from reimagining_trends.data.fetch_data import (
    _ensure_flat_columns, add_moving_average, cumret_scale, image_scale,
)
from reimagining_trends.imaging.ohlc_chart import generate_ohlc_image
from reimagining_trends.utils.config import Config

logger = logging.getLogger(__name__)

_MODEL_TYPE = {"MLP": "mlp", "GRU": "gru", "LSTM": "lstm", "CNN": "cnn"}


class Backtester:
    """
    Full backtest engine.

    Parameters
    ----------
    config    : Config
    raw_data  : {ticker: OHLCV DataFrame}  — full date range
    trainers  : {model_name: Trainer}
    """

    def __init__(self, config: Config, raw_data: dict, trainers: dict) -> None:
        self.cfg      = config
        self.raw_data = raw_data
        self.trainers = trainers
        os.makedirs(config.results_dir, exist_ok=True)

        # populated in _load_aux_data / _build_panels
        self._rf: pd.Series          = pd.Series(dtype=float)
        self._mktcap: pd.DataFrame   = pd.DataFrame()  # date x ticker
        self._sectors: dict          = {}               # ticker → gsector int
        self._daily_ret: pd.DataFrame = pd.DataFrame() # date x ticker
        self._betas: pd.DataFrame    = pd.DataFrame()  # date x ticker
        self._benchmark: pd.Series   = pd.Series(dtype=float)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Run all backtests; return nested results dict."""
        logger.info("Backtest: loading auxiliary data …")
        self._load_aux_data()
        self._build_panels()
        self._build_benchmark()
        self._compute_betas()

        weightings = (
            ["equal", "proportional", "cap_weighted"]
            if self.cfg.bt_weighting == "all"
            else [self.cfg.bt_weighting]
        )
        port_types = (
            ["LS", "LO"]
            if self.cfg.bt_portfolio_type == "both"
            else [self.cfg.bt_portfolio_type]
        )

        all_results: dict = {}

        for model_name, trainer in self.trainers.items():
            trainer.load_best()
            mtype = _MODEL_TYPE.get(model_name, "mlp")
            logger.info("Scoring %s on test dates …", model_name)
            scores_by_date = self._compute_all_scores(trainer, mtype)

            for weighting in weightings:
                for port_type in port_types:
                    key = f"{model_name}_{weighting}_{port_type}"
                    logger.info("Backtest: %s", key)
                    try:
                        result = self._single_backtest(
                            scores_by_date, weighting, port_type
                        )
                        all_results[key] = result
                        m = result["metrics"]
                        logger.info(
                            "  net_sharpe=%.3f | net_ann_ret=%.3f | MDD=%.3f | IR=%.3f",
                            m.get("net_sharpe", np.nan),
                            m.get("net_ann_return", np.nan),
                            m.get("net_max_drawdown", np.nan),
                            m.get("net_IR", np.nan),
                        )
                    except Exception as exc:
                        logger.warning(
                            "Backtest %s failed: %s\n%s",
                            key, exc, traceback.format_exc(),
                        )

        if all_results:
            self._plot_results(all_results)
            self._log_summary(all_results)

        return all_results

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _load_aux_data(self) -> None:
        """Load risk-free rate, market_cap and gsector from parquet."""
        rf_path = getattr(self.cfg, "bt_rf_path", None)
        if rf_path and os.path.exists(rf_path):
            rf_df = pd.read_parquet(rf_path)
            rf_df.columns = rf_df.columns.str.strip().str.lower()
            if rf_df.index.dtype != "datetime64[ns]":
                rf_df.index = pd.to_datetime(rf_df.index)
            col = "rf_returns" if "rf_returns" in rf_df.columns else rf_df.columns[0]
            rf_raw = rf_df[col].sort_index()
            if rf_raw.index.duplicated().any():
                rf_raw = rf_raw[~rf_raw.index.duplicated(keep="last")]
            self._rf = rf_raw
        else:
            logger.warning("RF path not found — using zero risk-free rate.")
            self._rf = pd.Series(dtype=float)

        parquet_path = getattr(self.cfg, "parquet_path", None)
        if not parquet_path or not os.path.exists(parquet_path):
            logger.warning("Parquet path unavailable — market_cap/sector set to defaults.")
            return

        df = pd.read_parquet(parquet_path)
        if df.index.name is not None and "date" not in df.columns:
            df = df.reset_index()
        df.columns = df.columns.str.strip().str.lower()
        df["date"] = pd.to_datetime(df["date"])

        if "market_cap" in df.columns:
            self._mktcap = (
                df.pivot_table(index="date", columns="ticker", values="market_cap")
                .sort_index()
            )

        if "gsector" in df.columns:
            raw = df.groupby("ticker")["gsector"].first()
            self._sectors = {}
            for ticker, val in raw.items():
                try:
                    self._sectors[ticker] = int(val)
                except (ValueError, TypeError):
                    pass

    def _build_panels(self) -> None:
        """Build daily returns panel from raw_data."""
        frames = {}
        for ticker, df in self.raw_data.items():
            df = _ensure_flat_columns(df)
            frames[ticker] = df["Close"].pct_change()
        daily = pd.DataFrame(frames).sort_index().fillna(0.0)
        if daily.index.duplicated().any():
            daily = daily[~daily.index.duplicated(keep="last")]
        self._daily_ret = daily

    def _build_benchmark(self) -> None:
        """Market-cap weighted CRSP benchmark daily return."""
        if self._mktcap.empty:
            logger.warning("No market_cap data — benchmark = equal-weight universe.")
            self._benchmark = self._daily_ret.mean(axis=1)
            return

        # Lag market cap by 1 day to avoid look-ahead
        mktcap_lag = self._mktcap.shift(1).reindex(self._daily_ret.index)
        aligned = mktcap_lag.reindex(columns=self._daily_ret.columns)
        w = aligned.div(aligned.sum(axis=1), axis=0).fillna(0.0)
        self._benchmark = (w * self._daily_ret).sum(axis=1)

    def _compute_betas(self) -> None:
        """Rolling OLS beta for each ticker vs benchmark."""
        window    = self.cfg.bt_beta_window
        min_obs   = self.cfg.bt_min_beta_obs
        bm        = self._benchmark

        betas = {}
        for ticker in self._daily_ret.columns:
            r = self._daily_ret[ticker]
            r, bm_aligned = r.align(bm, join="inner")
            cov = r.rolling(window, min_periods=min_obs).cov(bm_aligned)
            var = bm_aligned.rolling(window, min_periods=min_obs).var()
            betas[ticker] = (cov / var.replace(0, np.nan))

        self._betas = pd.DataFrame(betas).sort_index()

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def _compute_all_scores(self, trainer, model_type: str) -> dict:
        """
        For each rebalancing date t (signal date), run model on all tickers.

        Returns
        -------
        {date: {ticker: P(UP)}}
        """
        test_dates   = self._daily_ret.index[self._daily_ret.index > self.cfg.val_end]
        reb_dates    = test_dates[::self.cfg.horizon]   # every h trading days

        scores_by_date: dict = {}
        for t in reb_dates:
            scores = {}
            for ticker in self.raw_data:
                p = self._score_ticker(ticker, t, trainer, model_type)
                if p is not None:
                    scores[ticker] = p
            if scores:
                scores_by_date[t] = scores

        return scores_by_date

    def _score_ticker(self, ticker: str, t, trainer, model_type: str):
        """Return P(UP) for one ticker at signal date t, or None if unavailable."""
        df = self.raw_data[ticker]
        df_t = _ensure_flat_columns(df).loc[:t]

        try:
            if model_type == "cnn":
                return self._score_cnn(df_t, trainer)
            else:
                return self._score_tabular(df_t, trainer)
        except Exception:
            return None

    def _score_tabular(self, df_t: pd.DataFrame, trainer) -> float | None:
        w = self.cfg.window
        if self.cfg.include_ma:
            df_t = add_moving_average(df_t, w)
        if self.cfg.scaling == "image":
            df_s = image_scale(df_t, w)
        else:
            df_s = cumret_scale(df_t, w)
        feature_cols = ["Open", "High", "Low", "Close", "Volume"]
        if self.cfg.include_ma and "MA" in df_s.columns:
            feature_cols.append("MA")
        arr = df_s[feature_cols].values
        if len(arr) < w:
            return None
        x = arr[-w:][np.newaxis].astype(np.float32)   # (1, window, n_features)
        proba = trainer.predict(x)
        return float(proba[0, 1])

    def _score_cnn(self, df_t: pd.DataFrame, trainer) -> float | None:
        w = self.cfg.window
        df_t = df_t.copy()
        if self.cfg.include_ma:
            df_t["MA"] = df_t["Close"].rolling(w).mean()
            df_t = df_t.dropna(subset=["MA"])
        if len(df_t) < w:
            return None
        window_df = df_t.iloc[-w:]
        img = generate_ohlc_image(
            window_df, window=w,
            include_vol=self.cfg.include_vol,
            include_ma=self.cfg.include_ma,
        )
        x = img[np.newaxis, :, :, np.newaxis].astype(np.float32) / 255.0  # (1,H,W,1)
        proba = trainer.predict(x)
        return float(proba[0, 1])

    # ------------------------------------------------------------------
    # Single backtest run
    # ------------------------------------------------------------------

    def _single_backtest(self, scores_by_date: dict, weighting: str, port_type: str) -> dict:
        schedule = self._build_schedule(scores_by_date, weighting, port_type)
        if not schedule:
            raise ValueError("Empty schedule — no rebalancing dates found.")

        borrow_daily = self.cfg.bt_borrow_cost_bps_per_year / 10_000.0 / 252.0

        sim = simulate_portfolio(
            schedule        = schedule,
            daily_returns   = self._daily_ret,
            rf_series       = self._rf,
            cost_bps        = self.cfg.bt_cost_bps,
            borrow_rate_daily = borrow_daily if port_type == "LS" else 0.0,
        )

        bm = self._rf if port_type == "LS" else self._benchmark
        metrics = compute_backtest_metrics(
            sim               = sim,
            rf_series         = self._rf,
            benchmark_returns = bm,
            horizon           = self.cfg.horizon,
            periods_per_year  = 252,
        )
        return {"sim": sim, "metrics": metrics, "schedule": schedule}

    def _build_schedule(self, scores_by_date: dict, weighting: str, port_type: str) -> list:
        """
        Returns [(entry_date, exit_date, weights_dict), …]
        entry = signal + 1 trading day
        exit  = signal + h trading days
        """
        all_dates = self._daily_ret.index
        schedule  = []

        for sig_t, scores in sorted(scores_by_date.items()):
            future = all_dates[all_dates > sig_t]
            if len(future) < self.cfg.horizon:
                continue
            entry_t = future[0]               # t+1
            exit_t  = future[self.cfg.horizon - 1]  # t+h

            n_decile = max(1, len(scores) // self.cfg.n_deciles)

            betas_at_t     = self._betas_at(sig_t)
            mktcap_at_t    = self._mktcap_at(sig_t)

            if port_type == "LS":
                weights = construct_ls_portfolio(
                    scores      = scores,
                    market_caps = mktcap_at_t,
                    sectors     = self._sectors,
                    betas       = betas_at_t,
                    n_decile    = n_decile,
                    weighting   = weighting,
                    neutrality  = self.cfg.bt_neutrality,
                )
            else:
                weights = construct_lo_portfolio(
                    scores      = scores,
                    market_caps = mktcap_at_t,
                    betas       = betas_at_t,
                    n_decile    = n_decile,
                    weighting   = weighting,
                    beta_neutral = "beta" in self.cfg.bt_neutrality,
                )

            schedule.append((entry_t, exit_t, weights))

        return schedule

    def _betas_at(self, t) -> dict:
        if self._betas.empty:
            return {}
        row = self._betas.loc[:t].iloc[-1] if (self._betas.index <= t).any() else pd.Series()
        return {ticker: float(v) for ticker, v in row.items() if not np.isnan(v)}

    def _mktcap_at(self, t) -> dict:
        if self._mktcap.empty:
            return {}
        # Use t-1 (previous day's market cap) to avoid look-ahead
        rows = self._mktcap.loc[:t]
        if len(rows) < 2:
            return {}
        row = rows.iloc[-2]  # one day before t
        return {ticker: float(v) for ticker, v in row.items() if not np.isnan(v)}

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _plot_results(self, results: dict) -> None:
        self._plot_cumulative(results)
        self._plot_metrics_table(results)

    def _plot_cumulative(self, results: dict) -> None:
        fig, ax = plt.subplots(figsize=(12, 5))
        for key, res in results.items():
            sim = res["sim"]
            ax.plot(sim.index, sim["portfolio_value_net"], label=key, lw=1.2)
        ax.set_title("Cumulative portfolio value (net of costs)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio value (start = 1)")
        ax.axhline(1.0, color="black", lw=0.8, ls="--")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        path = os.path.join(self.cfg.results_dir, "backtest_cumulative.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Figure: %s", path)

    def _plot_metrics_table(self, results: dict) -> None:
        rows = []
        for key, res in results.items():
            m = res["metrics"]
            rows.append({
                "Strategy": key,
                "Net Ann. Ret.": f"{m.get('net_ann_return', np.nan):.3f}",
                "Net Sharpe":    f"{m.get('net_sharpe', np.nan):.3f}",
                "Net IR":        f"{m.get('net_IR', np.nan):.3f}",
                "MDD":           f"{m.get('net_max_drawdown', np.nan):.3f}",
                "Calmar":        f"{m.get('net_calmar', np.nan):.3f}",
                "Win Rate":      f"{m.get('net_win_rate', np.nan):.3f}",
                "Ann. Turnover": f"{m.get('ann_turnover', np.nan):.2f}",
            })
        df = pd.DataFrame(rows).set_index("Strategy")

        fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.8), len(df) * 0.5 + 1.5))
        ax.axis("off")
        tbl = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            rowLabels=df.index,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.4)
        plt.title("Backtest performance summary", fontsize=12, pad=10)
        plt.tight_layout()
        path = os.path.join(self.cfg.results_dir, "backtest_metrics.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Figure: %s", path)

    def _log_summary(self, results: dict) -> None:
        logger.info("=" * 70)
        logger.info("BACKTEST SUMMARY")
        logger.info("=" * 70)
        header = f"{'Strategy':<40} {'Net Sharpe':>10} {'Net Ann Ret':>12} {'MDD':>8} {'IR':>8}"
        logger.info(header)
        logger.info("-" * 70)
        for key, res in results.items():
            m = res["metrics"]
            logger.info(
                "%-40s %10.3f %12.3f %8.3f %8.3f",
                key,
                m.get("net_sharpe",     np.nan),
                m.get("net_ann_return", np.nan),
                m.get("net_max_drawdown", np.nan),
                m.get("net_IR",         np.nan),
            )
        logger.info("=" * 70)