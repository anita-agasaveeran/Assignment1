"""CRISP-DM Phase 2 -- Data Understanding: acquisition + raw materialisation.

The dataset is the Online Retail transactional log (Kaggle: `carrie1/ecommerce-data`,
originally UCI ML Repository ID 352, Chen et al. 2012).  We pull from the UCI
mirror so the pipeline runs without Kaggle credentials, but the bytes are the
same file Kaggle distributes.
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from armlab.config import Config

RAW_PARQUET = "online_retail_raw.parquet"


def download(cfg: Config, force: bool = False) -> Path:
    raw_dir = cfg.paths.resolve("raw")
    target = raw_dir / cfg.data.source_file
    if target.exists() and not force:
        return target
    with urllib.request.urlopen(cfg.data.source_url, timeout=300) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        name = zf.namelist()[0]
        target.write_bytes(zf.read(name))
    return target


def load_raw(cfg: Config, force: bool = False) -> pd.DataFrame:
    """Read the Excel source once, then cache as Parquet.

    Parsing 541k rows of .xlsx costs ~40s; Parquet reload costs ~0.2s.  Caching
    it is the difference between an interactive AutoResearch loop and a coffee
    break.
    """
    interim = cfg.paths.resolve("interim")
    cache = interim / RAW_PARQUET
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    src = download(cfg)
    df = pd.read_excel(src, dtype={"InvoiceNo": str, "StockCode": str})
    df.columns = [c.strip() for c in df.columns]
    # The Excel source stores a handful of Descriptions as numbers, which makes
    # the column genuinely mixed-type; Arrow refuses it. Normalise to string
    # while preserving true nulls so the missingness profile stays honest.
    for col in ("InvoiceNo", "StockCode", "Description", "Country"):
        if col in df.columns:
            df[col] = df[col].astype("string")
    if "CustomerID" in df.columns:
        df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")
    df.to_parquet(cache, index=False)
    return df
