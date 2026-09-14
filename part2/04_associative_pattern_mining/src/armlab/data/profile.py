"""CRISP-DM Phase 2 -- Data Understanding: the data-quality report.

Produces the numbers the dashboard's "Data Quality" panel renders.  Everything
here is descriptive: no row is dropped, we only measure.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from armlab.config import Config


def _pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 4) if d else 0.0


def column_profile(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        nulls = int(s.isna().sum())
        rec: dict[str, Any] = {
            "column": col,
            "dtype": str(s.dtype),
            "n_missing": nulls,
            "pct_missing": _pct(nulls, n),
            "n_unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s):
            d = s.dropna()
            if len(d):
                q = d.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
                rec |= {
                    "min": float(d.min()), "max": float(d.max()),
                    "mean": float(d.mean()), "std": float(d.std()),
                    "p01": float(q.loc[0.01]), "p25": float(q.loc[0.25]),
                    "p50": float(q.loc[0.50]), "p75": float(q.loc[0.75]),
                    "p99": float(q.loc[0.99]),
                    "n_negative": int((d < 0).sum()),
                    "n_zero": int((d == 0).sum()),
                }
        else:
            top = s.dropna().astype(str).value_counts().head(5)
            rec["top_values"] = [{"value": k, "count": int(v)} for k, v in top.items()]
        out.append(rec)
    return out


def quality_gates(df: pd.DataFrame, cfg: Config) -> list[dict[str, Any]]:
    """Named, assertable checks -- each one is a row in the dashboard table.

    A gate that fails is not fatal; it is *reported*, because for this dataset
    several of them are expected to fail (that is precisely why Phase 3 exists).
    """
    n = len(df)
    inv = df["InvoiceNo"].astype(str)
    gates = [
        ("rows_present", n > 0, f"{n:,} rows loaded"),
        ("no_missing_invoice", int(df["InvoiceNo"].isna().sum()) == 0,
         f"{int(df['InvoiceNo'].isna().sum()):,} missing InvoiceNo"),
        ("no_missing_stockcode", int(df["StockCode"].isna().sum()) == 0,
         f"{int(df['StockCode'].isna().sum()):,} missing StockCode"),
        ("description_complete", int(df["Description"].isna().sum()) == 0,
         f"{int(df['Description'].isna().sum()):,} missing Description "
         f"({_pct(int(df['Description'].isna().sum()), n)}%)"),
        ("customer_id_complete", int(df["CustomerID"].isna().sum()) == 0,
         f"{int(df['CustomerID'].isna().sum()):,} missing CustomerID "
         f"({_pct(int(df['CustomerID'].isna().sum()), n)}%) -- guest checkouts"),
        ("quantity_positive", int((df["Quantity"] <= 0).sum()) == 0,
         f"{int((df['Quantity'] <= 0).sum()):,} rows with Quantity <= 0 (returns)"),
        ("price_positive", int((df["UnitPrice"] <= 0).sum()) == 0,
         f"{int((df['UnitPrice'] <= 0).sum()):,} rows with UnitPrice <= 0"),
        ("no_credit_notes", int(inv.str.startswith("C").sum()) == 0,
         f"{int(inv.str.startswith('C').sum()):,} credit-note invoices (prefix 'C')"),
        ("no_exact_duplicates", int(df.duplicated().sum()) == 0,
         f"{int(df.duplicated().sum()):,} fully duplicated rows"),
    ]
    return [
        {"gate": g, "status": "pass" if ok else "warn", "detail": msg}
        for g, ok, msg in gates
    ]


def profile(df: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    inv = df["InvoiceNo"].astype(str)
    dates = pd.to_datetime(df["InvoiceDate"])
    baskets = df.groupby("InvoiceNo")["StockCode"].nunique()
    rev = (df["Quantity"] * df["UnitPrice"])
    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "n_invoices": int(inv.nunique()),
        "n_items": int(df["StockCode"].nunique()),
        "n_customers": int(df["CustomerID"].nunique(dropna=True)),
        "n_countries": int(df["Country"].nunique()),
        "date_min": str(dates.min()),
        "date_max": str(dates.max()),
        "span_days": int((dates.max() - dates.min()).days),
        "gross_revenue": float(rev.sum()),
        "basket_size": {
            "mean": float(baskets.mean()), "median": float(baskets.median()),
            "p90": float(baskets.quantile(0.90)), "max": int(baskets.max()),
            "n_singleton": int((baskets == 1).sum()),
        },
        "basket_size_hist": _histogram(baskets.to_numpy(), bins=30, clip=60),
        "top_items": [
            {"stock_code": str(k), "count": int(v)}
            for k, v in df["StockCode"].value_counts().head(20).items()
        ],
        "by_country": [
            {"country": str(k), "invoices": int(v)}
            for k, v in df.groupby("Country")["InvoiceNo"].nunique()
                          .sort_values(ascending=False).head(15).items()
        ],
        "monthly_invoices": [
            {"month": str(k), "invoices": int(v)}
            for k, v in inv.groupby(dates.dt.to_period("M").astype(str))
                           .nunique().sort_index().items()
        ],
        "columns": column_profile(df),
        "gates": quality_gates(df, cfg),
    }


def _histogram(values: np.ndarray, bins: int = 30, clip: int | None = None) -> list[dict]:
    v = np.asarray(values, dtype=float)
    if clip is not None:
        v = np.clip(v, None, clip)
    counts, edges = np.histogram(v, bins=bins)
    return [
        {"bin_start": float(edges[i]), "bin_end": float(edges[i + 1]),
         "count": int(counts[i])}
        for i in range(len(counts))
    ]
