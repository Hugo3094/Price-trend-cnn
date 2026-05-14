"""
data_manager.py
---------------
DataManager — owns the full data pipeline:
  download / load → tabular dataset → image dataset.

All parameters are taken from Config, so nothing is hardcoded.
"""

import logging
from typing import Optional

import pandas as pd

from reimagining_trends.data.fetch_data import (
    download_ohlcv,
    load_ohlcv,
    make_multi_stock_dataset,
)
from reimagining_trends.imaging.ohlc_chart import make_image_dataset
from reimagining_trends.utils.config import Config

logger = logging.getLogger(__name__)


class DataManager:
    """
    Orchestrates data download and dataset construction.

    Parameters
    ----------
    config : Config
        Pipeline configuration (tickers, dates, window, horizon, …).

    Attributes
    ----------
    raw_data : dict[str, pd.DataFrame]
        Populated after calling :meth:`download` or :meth:`load`.
    """

    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.raw_data: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Data acquisition
    # ------------------------------------------------------------------

    def download(self) -> dict[str, pd.DataFrame]:
        """Download OHLCV data for all configured tickers."""
        logger.info(
            "Downloading %d tickers from %s to %s",
            len(self.cfg.tickers), self.cfg.start, self.cfg.end,
        )
        self.raw_data = download_ohlcv(
            tickers=self.cfg.tickers,
            start=self.cfg.start,
            end=self.cfg.end,
            save_dir=self.cfg.data_save_dir,
        )
        logger.info("Downloaded %d tickers successfully.", len(self.raw_data))
        return self.raw_data

    def load(self, data_dir: str) -> dict[str, pd.DataFrame]:
        """Load previously downloaded CSVs from *data_dir*."""
        self.raw_data = load_ohlcv(data_dir)
        return self.raw_data

    # ------------------------------------------------------------------
    # Dataset builders
    # ------------------------------------------------------------------

    def build_tabular(
        self,
        raw_data: Optional[dict[str, pd.DataFrame]] = None,
    ) -> dict:
        """
        Build the tabular (MLP / LSTM / GRU) dataset.

        Parameters
        ----------
        raw_data : optional override — defaults to ``self.raw_data``

        Returns
        -------
        dict with X_train/val/test, y_train/val/test, window, horizon, scaling
        """
        data = raw_data if raw_data is not None else self.raw_data
        if not data:
            raise RuntimeError("No raw data available. Call download() or load() first.")

        logger.info(
            "Building tabular dataset: window=%d, horizon=%d, scaling=%s",
            self.cfg.window, self.cfg.horizon, self.cfg.scaling,
        )
        ds = make_multi_stock_dataset(
            data=data,
            window=self.cfg.window,
            horizon=self.cfg.horizon,
            scaling=self.cfg.scaling,
            add_ma=self.cfg.include_ma,
            train_ratio=self.cfg.train_ratio,
        )
        self._log_split_sizes("tabular", ds)
        return ds

    def build_image(
        self,
        raw_data: Optional[dict[str, pd.DataFrame]] = None,
    ) -> dict:
        """
        Build the OHLC image (CNN) dataset.

        Parameters
        ----------
        raw_data : optional override — defaults to ``self.raw_data``

        Returns
        -------
        dict with X_train/val/test (N,H,W,1), y_train/val/test
        """
        data = raw_data if raw_data is not None else self.raw_data
        if not data:
            raise RuntimeError("No raw data available. Call download() or load() first.")

        logger.info(
            "Building image dataset: window=%d, horizon=%d, vol=%s, ma=%s",
            self.cfg.window, self.cfg.horizon, self.cfg.include_vol, self.cfg.include_ma,
        )
        ds = make_image_dataset(
            data=data,
            window=self.cfg.window,
            horizon=self.cfg.horizon,
            include_vol=self.cfg.include_vol,
            include_ma=self.cfg.include_ma,
            train_ratio=self.cfg.train_ratio,
            val_ratio=self.cfg.val_ratio,
        )
        self._log_split_sizes("image", ds)
        return ds

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_split_sizes(kind: str, ds: dict) -> None:
        for split in ["train", "val", "test"]:
            X, y = ds[f"X_{split}"], ds[f"y_{split}"]
            logger.info(
                "  %s %s: %s  %.1f%% UP",
                kind, split, X.shape, y.mean() * 100,
            )
