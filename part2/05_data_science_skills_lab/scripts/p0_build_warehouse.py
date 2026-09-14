"""CRISP-DM Phase 2/3 support: normalise Online Retail into a small SQLite warehouse.

The raw Kaggle file is one flat transaction-line table. Several skills in the
data-analytics pack (schema-mapper, query-validation, sql-to-business-logic,
metric-reconciliation, referential integrity in data-quality-audit) are about
*relational* data, so we first model the flat file into a realistic star schema.

Deliberately, the analytics mart applies the conventional exclusions for this
dataset (drop rows with no CustomerID, drop cancellations). That creates a
genuine, explainable gap between "raw source" and "mart" totals, which is
exactly the situation the metric-reconciliation skill exists to resolve.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd

from common import PROC, RETAIL, WAREHOUSE, banner, sha256


def main() -> None:
    banner("(setup)", "Phase 3 - Data Preparation", "Normalising Online Retail into a SQLite star schema")
    print(f"  source        : {RETAIL.name}")
    print(f"  source sha256 : {sha256(RETAIL)[:32]}...")

    df = pd.read_csv(RETAIL, parse_dates=["InvoiceDate"], dtype={"InvoiceNo": str, "StockCode": str})
    print(f"  raw rows      : {len(df):,}")

    # -- pandas-patterns: vectorised derivation, no row-wise apply, no chained assignment
    df = df.assign(
        LineRevenue=lambda d: d["Quantity"] * d["UnitPrice"],
        IsCancellation=lambda d: d["InvoiceNo"].str.startswith("C").fillna(False),
    )

    raw_gross = float(df["LineRevenue"].sum())

    # -- Analytics mart exclusions (documented, reconciled later)
    mart = df[~df["IsCancellation"] & df["CustomerID"].notna() & (df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()
    mart["CustomerID"] = mart["CustomerID"].astype("int64")
    print(f"  mart rows     : {len(mart):,}  ({len(mart) / len(df):.1%} of raw)")

    # ---- dimension: products
    products = (
        mart.groupby("StockCode", observed=True)
        .agg(
            Description=("Description", lambda s: s.dropna().mode().iloc[0] if s.notna().any() else None),
            AvgUnitPrice=("UnitPrice", "mean"),
            TotalQtySold=("Quantity", "sum"),
        )
        .reset_index()
    )

    # ---- dimension: customers
    customers = (
        mart.groupby("CustomerID", observed=True)
        .agg(
            Country=("Country", lambda s: s.mode().iloc[0]),
            FirstPurchase=("InvoiceDate", "min"),
            LastPurchase=("InvoiceDate", "max"),
            Orders=("InvoiceNo", "nunique"),
        )
        .reset_index()
    )

    # ---- fact header: invoices
    invoices = (
        mart.groupby("InvoiceNo", observed=True)
        .agg(
            CustomerID=("CustomerID", "first"),
            InvoiceDate=("InvoiceDate", "min"),
            Country=("Country", "first"),
            Lines=("StockCode", "size"),
            InvoiceRevenue=("LineRevenue", "sum"),
        )
        .reset_index()
    )

    # ---- fact grain: invoice_lines
    lines = mart[["InvoiceNo", "StockCode", "Quantity", "UnitPrice", "LineRevenue", "InvoiceDate"]].reset_index(drop=True)
    lines.insert(0, "LineID", range(1, len(lines) + 1))

    WAREHOUSE.unlink(missing_ok=True)
    con = sqlite3.connect(WAREHOUSE)
    con.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE customers (
            CustomerID    INTEGER PRIMARY KEY,
            Country       TEXT NOT NULL,
            FirstPurchase TEXT,
            LastPurchase  TEXT,
            Orders        INTEGER
        );
        CREATE TABLE products (
            StockCode    TEXT PRIMARY KEY,
            Description  TEXT,
            AvgUnitPrice REAL,
            TotalQtySold INTEGER
        );
        CREATE TABLE invoices (
            InvoiceNo      TEXT PRIMARY KEY,
            CustomerID     INTEGER NOT NULL REFERENCES customers(CustomerID),
            InvoiceDate    TEXT NOT NULL,
            Country        TEXT,
            Lines          INTEGER,
            InvoiceRevenue REAL
        );
        CREATE TABLE invoice_lines (
            LineID      INTEGER PRIMARY KEY,
            InvoiceNo   TEXT NOT NULL REFERENCES invoices(InvoiceNo),
            StockCode   TEXT NOT NULL REFERENCES products(StockCode),
            Quantity    INTEGER NOT NULL,
            UnitPrice   REAL NOT NULL,
            LineRevenue REAL NOT NULL,
            InvoiceDate TEXT NOT NULL
        );
        CREATE INDEX idx_lines_invoice ON invoice_lines(InvoiceNo);
        CREATE INDEX idx_lines_stock   ON invoice_lines(StockCode);
        CREATE INDEX idx_inv_customer  ON invoices(CustomerID);
        CREATE INDEX idx_inv_date      ON invoices(InvoiceDate);
        """
    )
    for name, frame in [
        ("customers", customers),
        ("products", products),
        ("invoices", invoices),
        ("invoice_lines", lines),
    ]:
        frame.to_sql(name, con, if_exists="append", index=False)
        print(f"  loaded {name:<14} {len(frame):>8,} rows")
    con.commit()

    mart_gross = con.execute("SELECT SUM(LineRevenue) FROM invoice_lines").fetchone()[0]
    con.close()

    # Flat analysis-ready extracts used by the bundled skill scripts
    mart.to_csv(PROC / "retail_clean.csv", index=False)
    df.to_csv(PROC / "retail_with_flags.csv", index=False)
    customers.to_csv(PROC / "retail_customers.csv", index=False)

    print(f"\n  raw gross revenue  : GBP {raw_gross:>15,.2f}")
    print(f"  mart gross revenue : GBP {mart_gross:>15,.2f}")
    print(f"  gap                : GBP {raw_gross - mart_gross:>15,.2f}  ({(raw_gross - mart_gross)/raw_gross:.2%})")
    print(f"  -> this gap is the input to the metric-reconciliation skill demo")
    print(f"\n  warehouse: {WAREHOUSE}")


if __name__ == "__main__":
    main()
