"""CRISP-DM Phase 2 - Data Understanding (Track A: Titanic).

Skills demonstrated:
  * exploratory-data-analysis  -- its 7-step workflow, executed in order
  * pandas-patterns            -- vectorised / .loc / np.select / category idioms
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd

from common import FIG, PALETTE, PROC, TITANIC, banner, save_table, seed_everything, style_plots, write_json

TARGET = "Survived"


def eda(df: pd.DataFrame) -> dict:
    """The exploratory-data-analysis skill's workflow, steps 1-7, in order."""
    findings: dict = {}

    # 1. Shape & types -------------------------------------------------------
    banner("exploratory-data-analysis", "Phase 2 - Data Understanding", "step 1/7 shape & types")
    print(f"  shape: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"  memory: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    print(df.dtypes.to_string())
    findings["shape"] = list(df.shape)
    findings["dtypes"] = {k: str(v) for k, v in df.dtypes.items()}

    # 2. Missingness ---------------------------------------------------------
    banner("exploratory-data-analysis", "Phase 2 - Data Understanding", "step 2/7 missingness")
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    print((miss * 100).round(2).to_string())
    findings["missing_pct"] = (miss * 100).round(2).to_dict()
    # Is missingness itself predictive? (the skill explicitly asks this)
    cabin_known = df["Cabin"].notna()
    rate_known = df.loc[cabin_known, TARGET].mean()
    rate_unknown = df.loc[~cabin_known, TARGET].mean()
    print(f"\n  survival | Cabin recorded     : {rate_known:.3f}  (n={cabin_known.sum()})")
    print(f"  survival | Cabin missing      : {rate_unknown:.3f}  (n={(~cabin_known).sum()})")
    print(f"  -> missingness IS predictive (delta {rate_known - rate_unknown:+.3f});"
          f" keep a was_missing indicator rather than dropping the column")
    findings["cabin_missingness_is_informative"] = {
        "survival_when_recorded": round(rate_known, 4),
        "survival_when_missing": round(rate_unknown, 4),
        "delta": round(rate_known - rate_unknown, 4),
    }

    # 3. Target analysis -----------------------------------------------------
    banner("exploratory-data-analysis", "Phase 2 - Data Understanding", "step 3/7 target balance")
    bal = df[TARGET].value_counts(normalize=True).sort_index()
    print(bal.round(4).to_string())
    minority = float(bal.min())
    print(f"  minority class share: {minority:.1%}")
    print(f"  -> {'moderate imbalance' if minority < 0.45 else 'balanced'};"
          f" report ROC-AUC *and* PR-AUC, and tune the decision threshold")
    findings["target_balance"] = bal.round(4).to_dict()

    # 4. Univariate ----------------------------------------------------------
    banner("exploratory-data-analysis", "Phase 2 - Data Understanding", "step 4/7 univariate")
    num = df.select_dtypes("number").drop(columns=["PassengerId"])
    print(num.describe().T.round(3).to_string())
    findings["numeric_describe"] = num.describe().T.round(3).to_dict()
    for c in ["Sex", "Embarked", "Pclass"]:
        print(f"\n  {c}:\n{df[c].value_counts(dropna=False).to_string()}")

    # 5. Bivariate -----------------------------------------------------------
    banner("exploratory-data-analysis", "Phase 2 - Data Understanding", "step 5/7 bivariate")
    corr = num.corr(numeric_only=True)[TARGET].drop(TARGET).sort_values(ascending=False)
    print("  numeric corr with target:")
    print(corr.round(3).to_string())
    # The skill warns .corr() misses categorical / non-linear signal -> grouped means
    print("\n  grouped survival rates (what .corr() cannot see):")
    for c in ["Sex", "Pclass", "Embarked"]:
        g = df.groupby(c, observed=True)[TARGET].agg(["mean", "size"]).round(3)
        print(f"\n  by {c}:\n{g.to_string()}")
    findings["numeric_corr_with_target"] = corr.round(3).to_dict()
    findings["survival_by_sex"] = df.groupby("Sex", observed=True)[TARGET].mean().round(4).to_dict()
    findings["survival_by_pclass"] = df.groupby("Pclass", observed=True)[TARGET].mean().round(4).to_dict()

    # 6. Leakage scan --------------------------------------------------------
    banner("exploratory-data-analysis", "Phase 2 - Data Understanding", "step 6/7 leakage scan")
    abs_corr = corr.abs()
    suspects = abs_corr[abs_corr > 0.95].index.tolist()
    print(f"  |corr| > 0.95 with target : {suspects or 'none'}")
    id_like = [c for c in df.columns if df[c].nunique() == len(df)]
    print(f"  perfectly unique (ID-like): {id_like}")
    print("  -> PassengerId/Name/Ticket are identifiers, not features: excluded from X.")
    print("  -> no post-outcome columns present in this dataset.")
    findings["leakage_suspects"] = suspects
    findings["id_like_columns"] = id_like

    # 7. Cardinality & outliers ---------------------------------------------
    banner("exploratory-data-analysis", "Phase 2 - Data Understanding", "step 7/7 cardinality & outliers")
    card = df.nunique().sort_values(ascending=False)
    print("  cardinality:\n" + card.to_string())
    for c in ["Fare", "Age"]:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        hi = q3 + 1.5 * iqr
        n_out = int((df[c] > hi).sum())
        print(f"  {c}: IQR upper fence {hi:.2f} -> {n_out} high outliers "
              f"(max {df[c].max():.2f})")
        findings.setdefault("outliers", {})[c] = {"upper_fence": round(float(hi), 2), "n_above": n_out}
    print("  -> Fare has a long right tail (max 512.33). Winsorise at train percentiles;"
          " do NOT delete (first-class fares are real signal).")
    findings["high_cardinality"] = card[card > 100].to_dict()
    return findings


def pandas_patterns_demo(df: pd.DataFrame) -> pd.DataFrame:
    """The pandas-patterns skill's 5 core rules, applied to real columns."""
    banner("pandas-patterns", "Phase 2 - Data Understanding", "idiomatic, vectorised, copy-safe")
    out = df.copy()

    # Rule 1: assign with .loc, never chained indexing
    out.loc[out["Age"] >= 60, "life_stage"] = "senior"
    out.loc[out["Age"] < 18, "life_stage"] = "minor"
    out.loc[out["life_stage"].isna() & out["Age"].notna(), "life_stage"] = "adult"
    print("  rule 1 .loc assignment      ->", out["life_stage"].value_counts(dropna=False).to_dict())

    # Rule 2: vectorise instead of apply(axis=1)
    out["family_size"] = out["SibSp"] + out["Parch"] + 1
    print("  rule 2 vectorised arithmetic-> family_size mean", round(out["family_size"].mean(), 3))

    # Rule 3: np.select for multi-condition columns
    out["fare_tier"] = np.select(
        [out["Fare"] > 100, out["Fare"] > 30, out["Fare"] > 0],
        ["premium", "high", "standard"],
        default="free_or_unknown",
    )
    print("  rule 3 np.select            ->", out["fare_tier"].value_counts().to_dict())

    # Rule 4: downcast dtypes
    before = out.memory_usage(deep=True).sum()
    for c in ["Sex", "Embarked", "fare_tier", "life_stage"]:
        out[c] = out[c].astype("category")
    after = out.memory_usage(deep=True).sum()
    print(f"  rule 4 category downcast    -> {before/1e3:.1f} KB -> {after/1e3:.1f} KB "
          f"({(1 - after/before):.1%} smaller)")

    # Rule 5: merge with validated cardinality instead of loops
    deck_lookup = pd.DataFrame({
        "deck": list("ABCDEFGT"),
        "deck_level": [1, 2, 3, 4, 5, 6, 7, 8],
    })
    out["deck"] = out["Cabin"].str[0]
    out = out.merge(deck_lookup, on="deck", how="left", validate="m:1")
    print("  rule 5 validated m:1 merge  -> deck_level non-null:", int(out["deck_level"].notna().sum()))

    # Method chaining: readable and copy-safe
    summary = (
        out
        .query("Fare > 0")
        .assign(fare_per_person=lambda d: d["Fare"] / d["family_size"])
        .groupby(["Pclass", "Sex"], observed=True)
        .agg(n=("PassengerId", "size"),
             survival=(TARGET, "mean"),
             median_fare_pp=("fare_per_person", "median"))
        .round(3)
        .reset_index()
    )
    print("\n  method-chained summary (Pclass x Sex):")
    print(summary.to_string(index=False))
    save_table(summary.set_index(["Pclass", "Sex"]), "titanic_pclass_sex_summary.csv")
    return out


def figures(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    style_plots()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    ax = axes[0, 0]
    r = df.groupby("Sex", observed=True)[TARGET].mean()
    ax.bar(r.index.astype(str), r.values, color=[PALETTE[0], PALETTE[1]])
    ax.set_title("Sex is the strongest single predictor")
    ax.set_ylabel("survival rate")
    for i, v in enumerate(r.values):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontweight="bold")
    ax.set_ylim(0, 1)

    ax = axes[0, 1]
    r = df.groupby("Pclass", observed=True)[TARGET].mean()
    ax.bar(r.index.astype(str), r.values, color=PALETTE[2])
    ax.set_title("Survival falls monotonically with class")
    ax.set_xlabel("passenger class")
    ax.set_ylabel("survival rate")
    for i, v in enumerate(r.values):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontweight="bold")
    ax.set_ylim(0, 1)

    ax = axes[1, 0]
    for lab, sub, col in [("died", df[df[TARGET] == 0], PALETTE[1]), ("survived", df[df[TARGET] == 1], PALETTE[0])]:
        ax.hist(sub["Age"].dropna(), bins=25, alpha=0.6, label=lab, color=col)
    ax.set_title("Children under ~10 survived disproportionately")
    ax.set_xlabel("age")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    ax.barh(miss.index.astype(str), miss.values * 100, color=PALETTE[4])
    ax.set_title("Missingness concentrated in Cabin (77%)")
    ax.set_xlabel("% missing")
    ax.invert_yaxis()

    fig.suptitle("Titanic - EDA overview  (source: Kaggle Titanic, n=891)", fontweight="bold")
    fig.tight_layout()
    p = FIG / "p2a_titanic_eda.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  -> wrote {p.name}")


def main() -> None:
    seed_everything()
    df = pd.read_csv(TITANIC)
    findings = eda(df)
    enriched = pandas_patterns_demo(df)
    figures(df)
    write_json(findings, "p2a_titanic_eda_findings.json")
    enriched.to_csv(PROC / "titanic_explored.csv", index=False)

    banner("exploratory-data-analysis", "Phase 2 - Data Understanding", "hand-off summary")
    print(f"""  Shape 891x12. Target 'Survived' at {findings['target_balance'][1]:.1%} positive (moderate imbalance).
  Top missing: Cabin 77.1%, Age 19.9%, Embarked 0.2%.
  Cabin missingness is itself predictive (delta {findings['cabin_missingness_is_informative']['delta']:+.3f}) -> impute + indicator.
  No leakage suspects (|corr|>0.95): {findings['leakage_suspects'] or 'none'}.
  Identifiers to exclude from X: PassengerId, Name, Ticket, Cabin(raw).
  Candidate features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked + engineered
  (Title, FamilySize, IsAlone, Deck, CabinKnown, FarePerPerson).
  -> consumed by data-cleaning and feature-engineering in Phase 3.""")


if __name__ == "__main__":
    main()
