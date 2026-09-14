"""Phase 3 behaviour, on a hand-built frame where the right answer is obvious."""
import numpy as np
import pandas as pd
import pytest

from armlab.config import Config
from armlab.data import prepare


@pytest.fixture
def raw():
    # Stock codes look like the real dataset's on purpose: single letters such as
    # "B", "D", "M", "S" are genuine *service* codes in Online Retail and are
    # dropped by design, so a fixture using them would test the wrong thing.
    A, B, C = "85123A", "71053", "84406B"
    rows = [
        # invoice, stock, desc, qty, ts, price, cust, country
        ("1", A, "APPLE", 2, "2011-01-01", 1.0, 1, "United Kingdom"),
        ("1", B, "BREAD", 1, "2011-01-01", 2.0, 1, "United Kingdom"),
        ("1", B, "BREAD", 3, "2011-01-01", 2.0, 1, "United Kingdom"),   # same item twice
        ("2", A, "APPLE", 1, "2011-02-01", 1.0, 2, "United Kingdom"),
        ("2", C, "CHEESE", 1, "2011-02-01", 3.0, 2, "United Kingdom"),
        ("C3", A, "APPLE", -1, "2011-03-01", 1.0, 1, "United Kingdom"),  # credit note
        ("4", A, "APPLE", 0, "2011-04-01", 1.0, 3, "United Kingdom"),    # zero qty
        ("5", A, "APPLE", 1, "2011-05-01", 0.0, 3, "United Kingdom"),    # zero price
        ("6", "POST", "POSTAGE", 1, "2011-06-01", 5.0, 3, "United Kingdom"),  # service
        ("7", A, "APPLE", 1, "2011-07-01", 1.0, 4, "United Kingdom"),    # singleton basket
        ("8", A, "APPLE", 1, "2011-08-01", 1.0, 5, "United Kingdom"),
        ("8", B, "BREAD", 1, "2011-08-01", 2.0, 5, "United Kingdom"),
    ]
    df = pd.DataFrame(rows, columns=["InvoiceNo", "StockCode", "Description", "Quantity",
                                     "InvoiceDate", "UnitPrice", "CustomerID", "Country"])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


@pytest.fixture
def cfg():
    return Config().with_data(min_item_support_count=1, min_basket_size=2)


def test_clean_removes_exactly_what_it_documents(raw, cfg):
    out, log = prepare.clean(raw, cfg)
    inv = set(out["InvoiceNo"])
    assert "C3" not in inv, "credit note survived"
    assert "4" not in inv, "zero quantity survived"
    assert "5" not in inv, "zero price survived"
    assert not (out["StockCode"] == "POST").any(), "service code survived"
    assert {s["step"] for s in log} >= {"drop_credit_notes", "drop_non_positive_quantity",
                                        "drop_non_positive_price", "TOTAL"}
    total = log[-1]
    assert total["rows_before"] == len(raw)
    assert total["rows_after"] == len(out)


def test_baskets_are_sets_not_counts(raw, cfg):
    out, _ = prepare.clean(raw, cfg)
    ds = prepare.build_baskets(out, cfg)
    i = ds.basket_ids.index("1")
    assert len(ds.transactions[i]) == 2, "duplicate line item inflated the basket"
    assert len(set(ds.transactions[i])) == len(ds.transactions[i])


def test_singleton_baskets_are_dropped(raw, cfg):
    out, _ = prepare.clean(raw, cfg)
    ds = prepare.build_baskets(out, cfg)
    assert "7" not in ds.basket_ids
    assert all(len(t) >= 2 for t in ds.transactions)


def test_items_encoded_by_descending_support(raw, cfg):
    out, _ = prepare.clean(raw, cfg)
    ds = prepare.build_baskets(out, cfg)
    counts = list(ds.item_counts)
    assert counts == sorted(counts, reverse=True), "FP-Growth ordering violated"


def test_temporal_split_is_disjoint_and_ordered(raw, cfg):
    out, _ = prepare.clean(raw, cfg)
    ds = prepare.build_baskets(out, cfg)
    tr, te = prepare.split(ds, cfg)
    assert not (set(tr.basket_ids) & set(te.basket_ids))
    assert tr.n_transactions + te.n_transactions == ds.n_transactions
    if tr.n_transactions and te.n_transactions:
        assert tr.timestamps.max() <= te.timestamps.min(), "future leaked into training"


def test_random_split_is_also_disjoint(raw, cfg):
    cfg = cfg.with_data(split_strategy="random")
    out, _ = prepare.clean(raw, cfg)
    ds = prepare.build_baskets(out, cfg)
    tr, te = prepare.split(ds, cfg)
    assert not (set(tr.basket_ids) & set(te.basket_ids))
