"""Segment interpretation - the part the business actually consumes.

A cluster id is not a deliverable. This module turns the champion partition into
(a) a quantitative profile per segment, (b) a surrogate rule set that a human can
apply without the model, (c) a named persona with a recommended action, and
(d) an explicit unit-economics estimate so segments can be ranked by value rather
than by size.

The surrogate follows the standard global-explanation recipe (Craven & Shavlik's
TREPAN, NIPS 1995; the "surrogate model" pattern in Molnar's *Interpretable
Machine Learning*): fit a shallow decision tree to reproduce the model's own
assignments and report its **fidelity** - agreement with the model, not accuracy
against truth. Low fidelity means the tree is a nice story about a model that is
doing something else.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.tree import DecisionTreeClassifier, export_text

from . import config
from .data import FEATURES
from .prepare import ENGINEERED

PROFILE_FEATURES = FEATURES + [c for c in ENGINEERED if c != "MINPAY_IMPUTED"]


# --------------------------------------------------------------------------- #
# Quantitative profile
# --------------------------------------------------------------------------- #
def segment_profiles(frame: pd.DataFrame, labels: np.ndarray) -> list[dict]:
    """Per-segment means plus a standardised deviation from the book average.

    The z-score is what makes segments comparable: 'this segment's CASH_ADVANCE is
    +1.8 SD above book' travels; '$2,431' does not.
    """
    lab = np.asarray(labels)
    X = frame[PROFILE_FEATURES]
    book_mean = X.mean()
    book_std = X.std(ddof=0).replace(0, np.nan)
    out = []
    for c in sorted({int(v) for v in np.unique(lab) if v != -1}):
        m = lab == c
        sub = X[m]
        z = ((sub.mean() - book_mean) / book_std).fillna(0.0)
        ranked = z.reindex(z.abs().sort_values(ascending=False).index)
        out.append({
            "cluster": c,
            "n": int(m.sum()),
            "share": round(float(m.mean()), 4),
            "means": {k: round(float(v), 3) for k, v in sub.mean().items()},
            "medians": {k: round(float(v), 3) for k, v in sub.median().items()},
            "z": {k: round(float(v), 3) for k, v in z.items()},
            "signature": [{"feature": k, "z": round(float(v), 2),
                           "mean": round(float(sub[k].mean()), 2),
                           "book_mean": round(float(book_mean[k]), 2)}
                          for k, v in list(ranked.items())[:6]],
        })
    return out


# --------------------------------------------------------------------------- #
# Personas
# --------------------------------------------------------------------------- #
def _axis(z: dict, key: str) -> float:
    return float(z.get(key, 0.0))


ARCHETYPES: list[dict] = [
    {
        "name": "Cash-Advance Reliant",
        "fit": lambda a: a["cash_dependence"],
        "thesis": "Uses the card as a liquidity line rather than a payment instrument. The "
                  "highest credit risk and the highest fee income per account on the book.",
        "action": "Do not upsell spend. Offer a structured personal loan at a lower APR to "
                  "refinance the balance, and set a cash-advance velocity alert.",
        "risk": "high",
    },
    {
        "name": "High-Spend Transactor",
        "fit": lambda a: a["spend"] - a["revolving"],
        "thesis": "Heavy purchase volume settled in full - interchange-rich, interest-poor. "
                  "The most valuable and the most poachable segment on the book.",
        "action": "Protect with rewards and a proactive limit increase. Never target with "
                  "balance-transfer offers; it reads as a downgrade.",
        "risk": "low",
    },
    {
        "name": "Revolving Borrower",
        "fit": lambda a: a["revolving"],
        "thesis": "Carries a persistent balance against a meaningful limit and rarely settles "
                  "in full. The core net-interest-margin segment.",
        "action": "Retention pricing and a fixed-instalment conversion offer. Monitor "
                  "utilisation drift as an early-warning signal.",
        "risk": "medium-high",
    },
    {
        "name": "Instalment Planner",
        "fit": lambda a: a["instalment_bias"],
        "thesis": "Prefers instalment purchases over one-off spend; predictable, budget-driven "
                  "behaviour with moderate balances.",
        "action": "Merchant-funded instalment offers at point of sale; pre-approved "
                  "buy-now-pay-later limits on large-ticket categories.",
        "risk": "medium",
    },
    {
        "name": "Affluent Under-Utiliser",
        "fit": lambda a: a["limit"] - abs(a["spend"]) - a["cash_dependence"],
        "thesis": "Large granted line, comfortable balance, spend well below what the limit "
                  "implies. Wallet share is going somewhere else.",
        "action": "Category-specific accelerators to win back share of wallet; premium product "
                  "migration.",
        "risk": "low",
    },
    {
        "name": "Dormant / Low-Engagement",
        "fit": lambda a: -a["spend"] - a["activity"] - a["revolving"],
        "thesis": "Minimal spend, minimal balance, card effectively idle. Costs more to service "
                  "than it earns.",
        "action": "Low-cost reactivation only - a single targeted first-purchase incentive, "
                  "then let attrition run rather than spending against it.",
        "risk": "low",
    },
    {
        "name": "Mainstream Moderate",
        "fit": lambda a: -max(abs(v) for k, v in a.items()),
        "thesis": "Middle of the book on every behavioural axis - the reference segment against "
                  "which the others are defined.",
        "action": "Baseline programme. Use as the control group for every segment-level test.",
        "risk": "medium",
    },
]


def _axes(z: dict) -> dict:
    """Five behavioural axes, each a mean of the z-scores that define it."""
    return {
        "spend": float(np.mean([_axis(z, "PURCHASES"), _axis(z, "PURCHASES_TRX"),
                                _axis(z, "MONTHLY_SPEND")])),
        "cash_dependence": float(np.mean([_axis(z, "CASH_ADVANCE"),
                                          _axis(z, "CASH_ADVANCE_FREQUENCY"),
                                          _axis(z, "CASH_DEPENDENCE")])),
        "revolving": float(np.mean([_axis(z, "CREDIT_UTILISATION"), _axis(z, "BALANCE")])
                           - _axis(z, "PRC_FULL_PAYMENT")),
        "instalment_bias": float(_axis(z, "INSTALMENT_SHARE")
                                 - _axis(z, "ONEOFF_PURCHASES_FREQUENCY")),
        "limit": float(_axis(z, "CREDIT_LIMIT")),
        "activity": float(np.mean([_axis(z, "BALANCE_FREQUENCY"),
                                   _axis(z, "PURCHASES_FREQUENCY")])),
    }


def name_segments(profiles: list[dict]) -> list[dict]:
    """Assign each segment the archetype it fits best, with no two segments alike.

    Scoring each segment independently and taking its own best match produces
    collisions - an earlier version of this study labelled two different segments
    "Cash-Advance Reliant" and disambiguated them with "(Larger cohort)", which is
    not a name a campaign manager can do anything with.

    So the assignment is global and greedy: build the segment x archetype fit
    matrix, repeatedly take the strongest remaining pair, and retire both sides.
    Deterministic, and it forces each segment onto the archetype that describes it
    *distinctively* rather than the one it merely resembles. Segments left over
    after the archetypes run out are named by their own dominant axis rather than
    by a suffix.
    """
    axes_by_seg = {p["cluster"]: _axes(p["z"]) for p in profiles}
    pairs = sorted(
        ((arch["fit"](axes_by_seg[p["cluster"]]), p["cluster"], ai)
         for p in profiles for ai, arch in enumerate(ARCHETYPES)),
        key=lambda t: -t[0])

    assigned: dict[int, int] = {}
    used_arch: set[int] = set()
    for score, cid, ai in pairs:
        if cid in assigned or ai in used_arch:
            continue
        assigned[cid], _ = ai, used_arch.add(ai)

    named = []
    for p in profiles:
        a = axes_by_seg[p["cluster"]]
        ai = assigned.get(p["cluster"])
        if ai is not None:
            arch = ARCHETYPES[ai]
            name, thesis, action, risk = arch["name"], arch["thesis"], arch["action"], arch["risk"]
            fit_score = arch["fit"](a)
        else:  # more segments than archetypes - name by the dominant axis
            dom = max(a.items(), key=lambda kv: abs(kv[1]))
            direction = "High" if dom[1] > 0 else "Low"
            name = f"{direction}-{dom[0].replace('_', ' ').title()} Cohort"
            thesis = (f"Defined by its {dom[0].replace('_', ' ')} standing {abs(dom[1]):.1f} SD "
                      f"{'above' if dom[1] > 0 else 'below'} the book average.")
            action = "No archetype matched; treat as exploratory and review before campaigning."
            risk, fit_score = "unknown", float(dom[1])
        named.append({**p, "persona": name, "thesis": thesis, "action": action,
                      "risk_posture": risk, "archetype_fit": round(float(fit_score), 3),
                      "axes": {k: round(v, 2) for k, v in a.items()}})
    return sorted(named, key=lambda q: q["cluster"])


# --------------------------------------------------------------------------- #
# Surrogate explanation
# --------------------------------------------------------------------------- #
def surrogate_rules(frame: pd.DataFrame, labels: np.ndarray, depth: int = 4,
                    seed: int = config.RANDOM_SEED) -> dict:
    """Shallow decision tree fitted to the model's own labels (SC-5)."""
    lab = np.asarray(labels)
    m = lab != -1
    X = frame.loc[m, PROFILE_FEATURES].fillna(frame[PROFILE_FEATURES].median())
    y = lab[m]
    tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=max(30, len(X) // 200),
                                  random_state=seed, class_weight="balanced")
    tree.fit(X, y)
    fidelity = float(tree.score(X, y))
    imp = sorted(({"feature": f, "importance": round(float(i), 4)}
                  for f, i in zip(PROFILE_FEATURES, tree.feature_importances_) if i > 0),
                 key=lambda d: -d["importance"])
    return {
        "fidelity": round(fidelity, 4),
        "depth": depth,
        "n_leaves": int(tree.get_n_leaves()),
        "rules_text": export_text(tree, feature_names=list(PROFILE_FEATURES), max_depth=depth),
        "feature_importance": imp[:12],
        "per_cluster_recall": {
            int(c): round(float((tree.predict(X[y == c]) == c).mean()), 4)
            for c in sorted(set(int(v) for v in np.unique(y)))
        },
    }


# --------------------------------------------------------------------------- #
# Projection for the scatter plot
# --------------------------------------------------------------------------- #
def projection(X: np.ndarray, labels: np.ndarray, seed: int = config.RANDOM_SEED,
               tsne_sample: int = 2000) -> dict:
    """PCA-2 over every record, plus t-SNE on a sample.

    Both are shown because they lie in different ways: PCA preserves global
    geometry and can hide separation that exists in higher components; t-SNE
    (van der Maaten & Hinton, JMLR 2008) shows local neighbourhoods but makes
    inter-cluster distance and cluster size meaningless. Reading either alone
    is how people talk themselves into clusters that are not there.
    """
    rng = np.random.default_rng(seed)
    lab = np.asarray(labels)
    p = PCA(n_components=2, random_state=seed).fit(X)
    P = p.transform(X)
    n_plot = min(4000, len(X))
    idx = rng.choice(len(X), n_plot, replace=False)

    ts_idx = rng.choice(len(X), min(tsne_sample, len(X)), replace=False)
    try:
        T = TSNE(n_components=2, perplexity=30, init="pca", random_state=seed,
                 max_iter=500).fit_transform(X[ts_idx])
        tsne = {"x": [round(float(v), 2) for v in T[:, 0]],
                "y": [round(float(v), 2) for v in T[:, 1]],
                "label": [int(v) for v in lab[ts_idx]]}
    except Exception as exc:
        tsne = {"error": str(exc)[:120]}

    return {
        "pca": {"x": [round(float(v), 3) for v in P[idx, 0]],
                "y": [round(float(v), 3) for v in P[idx, 1]],
                "label": [int(v) for v in lab[idx]],
                "explained_variance": [round(float(v), 4) for v in p.explained_variance_ratio_],
                "n_plotted": n_plot},
        "tsne": tsne,
    }


# --------------------------------------------------------------------------- #
# Unit economics
# --------------------------------------------------------------------------- #
VALUE_ASSUMPTIONS = {
    "interchange_rate": 0.015,
    "cash_advance_fee": 0.03,
    "purchase_apr": 0.185,
    "cash_advance_apr": 0.245,
    "annual_servicing_cost": 22.0,
    "note": ("Illustrative unit economics, not the issuer's P&L. The dataset carries no "
             "revenue fields, so segment value is modelled from observed behaviour with the "
             "rates below. Swap these five numbers for the real ones and every ranking in "
             "this section updates; the ordering is far more robust than the levels."),
}


def unit_economics(frame: pd.DataFrame, labels: np.ndarray) -> list[dict]:
    a = VALUE_ASSUMPTIONS
    lab = np.asarray(labels)
    months = frame["TENURE"].clip(lower=1)
    interchange = frame["PURCHASES"] * a["interchange_rate"]
    ca_fee = frame["CASH_ADVANCE"] * a["cash_advance_fee"]
    revolved = frame["BALANCE"] * (1 - frame["PRC_FULL_PAYMENT"].clip(0, 1))
    interest = revolved * a["purchase_apr"] * (months / 12.0)
    servicing = a["annual_servicing_cost"] * (months / 12.0)
    contribution = interchange + ca_fee + interest - servicing

    total = float(contribution.sum())
    out = []
    for c in sorted({int(v) for v in np.unique(lab) if v != -1}):
        m = lab == c
        cs = float(contribution[m].sum())
        out.append({
            "cluster": c, "n": int(m.sum()),
            "share_of_accounts": round(float(m.mean()), 4),
            "annual_contribution": round(cs, 0),
            "share_of_contribution": round(cs / total, 4) if total else 0.0,
            "contribution_per_account": round(cs / max(int(m.sum()), 1), 2),
            "interchange": round(float(interchange[m].sum()), 0),
            "cash_advance_fees": round(float(ca_fee[m].sum()), 0),
            "net_interest": round(float(interest[m].sum()), 0),
            "servicing_cost": round(float(servicing[m].sum()), 0),
            "value_index": round((cs / max(int(m.sum()), 1)) / (total / len(frame)), 2) if total else 0.0,
        })
    return sorted(out, key=lambda d: -d["annual_contribution"])


def boundary_accounts(margins: np.ndarray, agreement: np.ndarray | None,
                      labels: np.ndarray, q: float = 0.10) -> dict:
    """Accounts whose segment assignment is not trustworthy.

    Two independent signals: a small nearest/second-nearest centroid gap (the record
    sits on a boundary), and low agreement across bootstrap refits (the boundary
    itself moves). Campaigns should either exclude these or treat them as their own
    hold-out - they are the accounts whose measured 'segment response' is noise.
    """
    m = np.asarray(margins, dtype=float)
    thresh = float(np.quantile(m, q))
    low_margin = m <= thresh
    res = {"margin_threshold": round(thresh, 4),
           "pct_low_margin": round(100 * float(low_margin.mean()), 2),
           "per_cluster_low_margin": {}}
    lab = np.asarray(labels)
    for c in sorted({int(v) for v in np.unique(lab) if v != -1}):
        sel = lab == c
        res["per_cluster_low_margin"][c] = round(100 * float(low_margin[sel].mean()), 2)
    if agreement is not None:
        ag = np.asarray(agreement, dtype=float)
        unstable = (ag < 0.8) & low_margin
        res["pct_low_margin_and_unstable"] = round(100 * float(unstable.mean()), 2)
        res["n_flagged"] = int(unstable.sum())
    return res
