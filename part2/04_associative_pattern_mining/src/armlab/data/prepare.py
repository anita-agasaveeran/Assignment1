"""CRISP-DM Phase 3 -- Data Preparation: transaction log -> market baskets.

Output is a `BasketDataset`: integer-encoded transactions plus the vocabulary
needed to turn item ids back into human-readable product names.  Integer
encoding (ordered by descending support) is not cosmetic -- FP-Growth's
compression depends on that ordering (Han, Pei & Yin, SIGMOD 2000, Sec. 2.1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from armlab.config import Config

# StockCodes in this dataset that are charges/adjustments, not products.
SERVICE_CODES = {
    "POST", "DOT", "C2", "M", "m", "D", "S", "AMAZONFEE", "BANK CHARGES",
    "CRUK", "PADS", "B", "gift_0001_10", "gift_0001_20", "gift_0001_30",
    "gift_0001_40", "gift_0001_50",
}
# Descriptions that mark manual/adjustment rows.
NOISE_DESCRIPTION_TOKENS = (
    "AMAZON", "ADJUST", "CHECK", "DAMAGE", "FOUND", "MANUAL", "SAMPLES",
    "SMASHED", "LOST", "MISSING", "WET", "MOULDY", "THROWN AWAY", "?",
    "CRUSHED", "TEST", "COUNTED", "INCORRECT", "BROKEN", "WRONG",
)


@dataclass
class BasketDataset:
    """Integer-encoded transactions ready for pattern mining."""
    transactions: list[list[int]]
    item_labels: dict[int, str]         # item_id -> Description
    item_codes: dict[int, str]          # item_id -> StockCode
    item_counts: np.ndarray             # support count per item_id
    basket_ids: list[str]               # InvoiceNo, parallel to transactions
    timestamps: np.ndarray              # invoice datetime, parallel
    split: str = "all"
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_transactions(self) -> int:
        return len(self.transactions)

    @property
    def n_items(self) -> int:
        return len(self.item_labels)

    def label(self, item_id: int) -> str:
        return self.item_labels.get(item_id, f"item_{item_id}")

    def labels(self, itemset) -> list[str]:
        return [self.label(i) for i in sorted(itemset)]

    def subset(self, idx: np.ndarray, split: str) -> "BasketDataset":
        idx = np.asarray(idx)
        return BasketDataset(
            transactions=[self.transactions[i] for i in idx],
            item_labels=self.item_labels,
            item_codes=self.item_codes,
            item_counts=self.item_counts,
            basket_ids=[self.basket_ids[i] for i in idx],
            timestamps=self.timestamps[idx],
            split=split,
            meta=dict(self.meta),
        )


def clean(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Apply the documented cleaning recipe, recording an auditable drop-log."""
    d = cfg.data
    log: list[dict[str, Any]] = []
    n0 = len(df)
    cur = df.copy()

    def step(name: str, mask: pd.Series, rationale: str) -> None:
        nonlocal cur
        before = len(cur)
        cur = cur[mask].copy()
        log.append({
            "step": name, "rows_before": before, "rows_after": len(cur),
            "rows_dropped": before - len(cur),
            "pct_dropped": round(100 * (before - len(cur)) / before, 4) if before else 0.0,
            "rationale": rationale,
        })

    cur["InvoiceNo"] = cur["InvoiceNo"].astype(str).str.strip()
    cur["StockCode"] = cur["StockCode"].astype(str).str.strip().str.upper()
    cur["Description"] = cur["Description"].astype(str).str.strip().str.upper()

    step("drop_exact_duplicates", ~cur.duplicated(),
         "Identical (invoice, item, qty, price, ts) rows are double-scans, not "
         "two independent purchases; they would inflate support counts.")

    if d.drop_cancellations:
        step("drop_credit_notes", ~cur["InvoiceNo"].str.startswith("C"),
             "Invoices prefixed 'C' are cancellations. A returned basket is not "
             "evidence of co-purchase intent.")

    if d.drop_non_positive_quantity:
        step("drop_non_positive_quantity", cur["Quantity"] > 0,
             "Quantity <= 0 encodes returns/adjustments.")

    if d.drop_non_positive_price:
        step("drop_non_positive_price", cur["UnitPrice"] > 0,
             "UnitPrice <= 0 encodes freebies and stock adjustments, not sales.")

    step("drop_missing_description",
         cur["Description"].notna() & (cur["Description"] != "") & (cur["Description"] != "NAN"),
         "Rows with no product name cannot be interpreted by a merchandiser.")

    if d.drop_service_codes:
        svc = cur["StockCode"].isin({c.upper() for c in SERVICE_CODES})
        noisy = cur["Description"].str.contains(
            "|".join(map(lambda t: t.replace("?", r"\?"), NOISE_DESCRIPTION_TOKENS)),
            regex=True, na=False)
        step("drop_service_and_adjustment_rows", ~(svc | noisy),
             "Postage, bank charges and manual stock adjustments are not "
             "products; leaving them in produces rules like {POSTAGE} => {x}.")

    if d.countries:
        step("filter_countries", cur["Country"].isin(d.countries),
             f"Restricted to {d.countries} to keep the population homogeneous.")

    log.append({
        "step": "TOTAL", "rows_before": n0, "rows_after": len(cur),
        "rows_dropped": n0 - len(cur),
        "pct_dropped": round(100 * (n0 - len(cur)) / n0, 4) if n0 else 0.0,
        "rationale": "End of Phase 3 cleaning.",
    })
    return cur, log


def build_baskets(df: pd.DataFrame, cfg: Config) -> BasketDataset:
    """Collapse the line-item log into de-duplicated transactions.

    Market-basket analysis is defined over *sets*: buying six of one item is one
    piece of co-occurrence evidence, not six (Agrawal & Srikant, VLDB 1994).
    """
    d = cfg.data
    prep_log: list[dict[str, Any]] = []

    # Item frequency floor, applied on basket-presence not line count.
    presence = df.drop_duplicates([d.basket_key, d.item_key])
    counts = presence[d.item_key].value_counts()
    keep_items = counts[counts >= d.min_item_support_count].index
    n_items_before = int(counts.size)
    presence = presence[presence[d.item_key].isin(keep_items)]
    prep_log.append({
        "step": "item_frequency_floor",
        "detail": f"kept {len(keep_items):,}/{n_items_before:,} items appearing in "
                  f">= {d.min_item_support_count} baskets",
        "rationale": "Items below the floor cannot reach any useful min_support, "
                     "but they inflate the FP-tree and the search space.",
    })

    # Stable, human-readable label per item (modal description).
    label_map = (presence.groupby(d.item_key)[d.label_key]
                 .agg(lambda s: s.value_counts().index[0]).to_dict())

    # Encode items by DESCENDING support -- required by FP-Growth ordering.
    ordered = counts.loc[list(keep_items)].sort_values(ascending=False)
    code_to_id = {code: i for i, code in enumerate(ordered.index)}
    item_labels = {i: str(label_map.get(code, code)) for code, i in code_to_id.items()}
    item_codes = {i: str(code) for code, i in code_to_id.items()}
    item_counts_arr = ordered.to_numpy()

    grouped = presence.groupby(d.basket_key)
    basket_ids: list[str] = []
    transactions: list[list[int]] = []
    stamps: list[Any] = []
    n_raw_baskets = 0
    for inv, g in grouped:
        n_raw_baskets += 1
        ids = sorted({code_to_id[c] for c in g[d.item_key]})
        if len(ids) < d.min_basket_size or len(ids) > d.max_basket_size:
            continue
        basket_ids.append(str(inv))
        transactions.append(ids)
        stamps.append(g["InvoiceDate"].min())

    prep_log.append({
        "step": "basket_size_filter",
        "detail": f"kept {len(transactions):,}/{n_raw_baskets:,} baskets with "
                  f"{d.min_basket_size} <= size <= {d.max_basket_size}",
        "rationale": "Size-1 baskets cannot support a rule; oversized baskets are "
                     "wholesale orders whose co-occurrence is an artefact of bulk "
                     "picking rather than shopper intent.",
    })

    order = np.argsort(np.asarray(stamps, dtype="datetime64[ns]"), kind="stable")
    ts = np.asarray(stamps, dtype="datetime64[ns]")[order]
    return BasketDataset(
        transactions=[transactions[i] for i in order],
        item_labels=item_labels,
        item_codes=item_codes,
        item_counts=item_counts_arr,
        basket_ids=[basket_ids[i] for i in order],
        timestamps=ts,
        split="all",
        meta={"prep_log": prep_log,
              "avg_basket_size": float(np.mean([len(t) for t in transactions])) if transactions else 0.0},
    )


def split(ds: BasketDataset, cfg: Config) -> tuple[BasketDataset, BasketDataset]:
    """Exploratory / holdout split.

    Temporal by default.  Webb (2007, *Discovering Significant Patterns*, Mach.
    Learn. 68:1-33) shows that testing a rule on the same data that suggested it
    inflates the false-discovery rate arbitrarily; the holdout must be untouched
    during search.  For retail we go further and split on *time*, because a
    random split lets December co-purchases vouch for a rule discovered in
    December.
    """
    n = ds.n_transactions
    n_hold = max(1, int(round(n * cfg.data.holdout_fraction)))
    if cfg.data.split_strategy == "temporal":
        cut = n - n_hold
        train_idx = np.arange(cut)
        test_idx = np.arange(cut, n)
    else:
        rng = np.random.default_rng(cfg.data.random_seed)
        perm = rng.permutation(n)
        test_idx, train_idx = perm[:n_hold], perm[n_hold:]
        train_idx.sort(); test_idx.sort()
    return ds.subset(train_idx, "explore"), ds.subset(test_idx, "holdout")
