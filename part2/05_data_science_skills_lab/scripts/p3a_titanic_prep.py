"""CRISP-DM Phase 3 - Data Preparation (Track A: Titanic).

Skills demonstrated:
  * data-cleaning        -- split FIRST, then learn every statistic on train only
  * feature-engineering  -- encoding, transforms, aggregation features,
                            and cross-fitted (out-of-fold) target encoding
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from common import PROC, SEED, TITANIC, banner, seed_everything, write_json

TARGET = "Survived"
DECISIONS: list[dict] = []


def log(step: str, decision: str, why: str) -> None:
    DECISIONS.append({"step": step, "decision": decision, "rationale": why})
    print(f"  [{step}] {decision}\n           why: {why}")


# --------------------------------------------------------------- data-cleaning
def clean(df: pd.DataFrame):
    banner("data-cleaning", "Phase 3 - Data Preparation",
           "golden rule: split first, learn every cleaning statistic on train only")

    # Step 0 -- SPLIT BEFORE CLEANING (the skill's headline pitfall)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    print(f"  split first: train {len(X_tr)}, test {len(X_te)} "
          f"(stratified, positive rate {y_tr.mean():.3f} / {y_te.mean():.3f})")
    log("split", "train_test_split(stratify=y) BEFORE any imputation",
        "cleaning before splitting leaks test information into train statistics")

    # Step 1 -- deduplicate
    dupes = int(df.duplicated().sum())
    key_dupes = int(df.duplicated(subset=["Name", "Ticket"]).sum())
    print(f"\n  step 1 duplicates: {dupes} exact, {key_dupes} on (Name, Ticket)")
    log("deduplicate", f"none removed ({dupes} exact duplicates found)",
        "Titanic is one row per passenger; no fan-out to collapse")

    # Step 2 -- fix types
    print("\n  step 2 types:")
    for frame in (X_tr, X_te):
        frame["Pclass"] = frame["Pclass"].astype("int8")
        frame["SibSp"] = frame["SibSp"].astype("int8")
        frame["Parch"] = frame["Parch"].astype("int8")
    print("          Pclass/SibSp/Parch -> int8 (ordinal-coded small integers)")
    log("types", "downcast Pclass/SibSp/Parch to int8",
        "they are bounded small integers; keeps memory and dtype intent explicit")

    # Step 3 -- standardise categoricals
    print("\n  step 3 categoricals:")
    for frame in (X_tr, X_te):
        frame["Sex"] = frame["Sex"].str.strip().str.lower()
        frame["Embarked"] = frame["Embarked"].str.strip().str.upper()
    print(f"          Sex     -> {sorted(X_tr['Sex'].dropna().unique())}")
    print(f"          Embarked-> {sorted(X_tr['Embarked'].dropna().unique())}")
    log("standardise", "trim + case-normalise Sex and Embarked",
        "prevents 'male'/'Male ' splitting into two one-hot columns")

    # Step 4 -- missing values, per the skill's strategy table
    print("\n  step 4 missing values (all statistics from TRAIN ONLY):")
    age_median = float(X_tr["Age"].median())
    emb_mode = X_tr["Embarked"].mode().iloc[0]
    fare_median = float(X_tr["Fare"].median())
    print(f"          train Age median   = {age_median}")
    print(f"          train Embarked mode= {emb_mode}")
    print(f"          train Fare median  = {fare_median}")
    print(f"          (test Age median would have been {X_te['Age'].median()} -- deliberately NOT used)")

    for frame in (X_tr, X_te):
        # informative missingness -> impute AND keep an indicator (EDA proved this predicts)
        frame["Age_was_missing"] = frame["Age"].isna().astype("int8")
        frame["CabinKnown"] = frame["Cabin"].notna().astype("int8")
        frame["Age"] = frame["Age"].fillna(age_median)
        frame["Embarked"] = frame["Embarked"].fillna(emb_mode)
        frame["Fare"] = frame["Fare"].fillna(fare_median)
    log("impute Age", f"median {age_median} from train + `Age_was_missing` indicator",
        "19.9% missing, right-skewed -> median is robust; missingness may carry signal")
    log("impute Embarked", f"mode '{emb_mode}' from train",
        "only 2 rows missing; mode imputation is immaterial either way")
    log("Cabin", "kept only as the binary `CabinKnown`; raw column dropped",
        "77.1% missing so the raw value is unusable, but EDA showed a +0.367 survival "
        "delta between recorded and missing -- the indicator is the signal")

    # Step 5 -- outliers: winsorise at TRAIN percentiles, never delete
    lo, hi = X_tr["Fare"].quantile([0.01, 0.99])
    n_clipped = int((X_tr["Fare"] > hi).sum() + (X_tr["Fare"] < lo).sum())
    for frame in (X_tr, X_te):
        frame["Fare"] = frame["Fare"].clip(lo, hi)
    print(f"\n  step 5 outliers: winsorised Fare to train p1={lo:.2f}, p99={hi:.2f} "
          f"({n_clipped} train rows clipped)")
    log("outliers", f"winsorise Fare to train [{lo:.2f}, {hi:.2f}]",
        "first-class fares up to 512 are real signal, not errors -- cap, never delete")

    # Step 6 -- validate
    print("\n  step 6 validation:")
    keep = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked",
            "Age_was_missing", "CabinKnown", "Name", "Ticket", "Cabin"]
    for nm, frame in [("train", X_tr), ("test", X_te)]:
        nulls = frame[keep].isna().sum()
        nulls = nulls[nulls > 0].drop(labels=["Cabin"], errors="ignore")
        assert nulls.empty, f"unexpected nulls in {nm}: {nulls.to_dict()}"
        assert frame["Age"].between(0, 100).all()
        assert (frame["Fare"] >= 0).all()
        print(f"          {nm}: schema OK, no nulls in modelling columns, ranges valid")
    assert len(X_tr) + len(X_te) == len(df), "row count changed"
    print(f"          row count preserved: {len(X_tr)} + {len(X_te)} = {len(df)}")
    return X_tr, X_te, y_tr, y_te


# --------------------------------------------------------- feature-engineering
TITLE_MAP = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
    "Dr": "Officer", "Rev": "Officer", "Col": "Officer", "Major": "Officer",
    "Capt": "Officer", "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
    "Don": "Royalty", "Dona": "Royalty", "Lady": "Royalty", "Sir": "Royalty",
    "the Countess": "Royalty", "Jonkheer": "Royalty",
}


def target_encode_oof(train: pd.DataFrame, col: str, y: pd.Series,
                      n_splits: int = 5, smoothing: float = 10.0):
    """The feature-engineering skill's cross-fitted target encoder.

    Returns (oof values for train, mapping learned on all of train for test).
    Each train row is encoded by folds that exclude it -- so the encoding never
    sees that row's own label.
    """
    oof = np.zeros(len(train))
    prior = y.mean()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for tr_idx, val_idx in kf.split(train):
        agg = y.iloc[tr_idx].groupby(train[col].iloc[tr_idx], observed=True).agg(["mean", "count"])
        smooth = (agg["mean"] * agg["count"] + prior * smoothing) / (agg["count"] + smoothing)
        oof[val_idx] = train[col].iloc[val_idx].map(smooth).fillna(prior).to_numpy()
    full = y.groupby(train[col], observed=True).agg(["mean", "count"])
    full_map = (full["mean"] * full["count"] + prior * smoothing) / (full["count"] + smoothing)
    return oof, full_map, prior


def engineer(X_tr, X_te, y_tr):
    banner("feature-engineering", "Phase 3 - Data Preparation",
           "express the signal without letting the target or the test set leak in")

    for frame in (X_tr, X_te):
        # -- text feature: honorific carries age/sex/status jointly
        frame["Title"] = (frame["Name"].str.extract(r",\s*([^\.]+)\.", expand=False)
                          .str.strip().map(TITLE_MAP).fillna("Rare"))
        # -- aggregation features
        frame["FamilySize"] = frame["SibSp"] + frame["Parch"] + 1
        frame["IsAlone"] = (frame["FamilySize"] == 1).astype("int8")
        # -- ratio feature with domain meaning: a ticket often covers a group
        frame["FarePerPerson"] = frame["Fare"] / frame["FamilySize"]
        # -- skew correction (Fare skew ~4.8 before, see EDA)
        frame["FareLog"] = np.log1p(frame["Fare"])
        # -- structural feature from a mostly-missing column
        frame["Deck"] = frame["Cabin"].str[0].fillna("U")
        # -- ordinal map that preserves order
        frame["AgeBand"] = pd.cut(frame["Age"], bins=[0, 12, 18, 35, 60, 100],
                                  labels=[0, 1, 2, 3, 4]).astype("int8")

    print(f"  Title       -> {X_tr['Title'].value_counts().to_dict()}")
    print(f"  FamilySize  -> mean {X_tr['FamilySize'].mean():.2f}, "
          f"IsAlone share {X_tr['IsAlone'].mean():.1%}")
    print(f"  Fare skew   -> {X_tr['Fare'].skew():.2f} raw / {X_tr['FareLog'].skew():.2f} after log1p")
    print(f"  Deck        -> {X_tr['Deck'].value_counts().to_dict()}")

    # -- leakage-safe target encoding on the highest-cardinality categorical
    print("\n  cross-fitted (out-of-fold) target encoding of Ticket prefix:")
    for frame in (X_tr, X_te):
        frame["TicketPrefix"] = (frame["Ticket"].str.replace(r"[\./]", "", regex=True)
                                 .str.split().str[0].where(lambda s: ~s.str.isdigit(), "NUM"))
    card = X_tr["TicketPrefix"].nunique()
    oof, full_map, prior = target_encode_oof(X_tr, "TicketPrefix", y_tr)
    X_tr["TicketPrefix_te"] = oof
    X_te["TicketPrefix_te"] = X_te["TicketPrefix"].map(full_map).fillna(prior).to_numpy()
    print(f"          cardinality {card} -> one numeric column")
    print(f"          prior (train survival rate) = {prior:.4f}")

    # Prove the cross-fitting matters: naive in-fold encoding correlates far
    # more strongly with the target than the honest out-of-fold version.
    naive = X_tr.groupby("TicketPrefix", observed=True)["TicketPrefix"].transform("size")
    naive_enc = X_tr["TicketPrefix"].map(y_tr.groupby(X_tr["TicketPrefix"], observed=True).mean())
    r_naive = float(np.corrcoef(naive_enc.fillna(prior), y_tr)[0, 1])
    r_oof = float(np.corrcoef(oof, y_tr)[0, 1])
    print(f"          corr(naive in-fold encoding, y) = {r_naive:.4f}   <- inflated by leakage")
    print(f"          corr(out-of-fold encoding,  y) = {r_oof:.4f}   <- honest")
    print(f"          leakage inflation: {r_naive - r_oof:+.4f}")
    log("target encoding", "cross-fitted KFold(5) with smoothing=10 on TicketPrefix",
        f"in-fold encoding correlates {r_naive:.3f} with y vs {r_oof:.3f} out-of-fold -- "
        f"the difference is pure leakage that would collapse in production")

    drop = ["Name", "Ticket", "Cabin", "PassengerId", "TicketPrefix"]
    X_tr_f = X_tr.drop(columns=drop)
    X_te_f = X_te.drop(columns=drop)
    print(f"\n  dropped identifiers: {drop}")
    print(f"  final feature set ({X_tr_f.shape[1]} columns): {list(X_tr_f.columns)}")
    return X_tr_f, X_te_f, {"naive_corr": r_naive, "oof_corr": r_oof, "prior": prior,
                            "ticket_prefix_cardinality": int(card)}


def main() -> None:
    seed_everything()
    df = pd.read_csv(TITANIC)
    X_tr, X_te, y_tr, y_te = clean(df)
    X_tr_f, X_te_f, te_stats = engineer(X_tr, X_te, y_tr)

    X_tr_f.assign(**{TARGET: y_tr}).to_csv(PROC / "titanic_train.csv", index=False)
    X_te_f.assign(**{TARGET: y_te}).to_csv(PROC / "titanic_test.csv", index=False)
    write_json({"decisions": DECISIONS, "target_encoding": te_stats,
                "train_rows": len(X_tr_f), "test_rows": len(X_te_f),
                "features": list(X_tr_f.columns)},
               "p3a_titanic_prep_decisions.json")

    banner("data-cleaning + feature-engineering", "Phase 3 - Data Preparation", "hand-off")
    print(f"  train {X_tr_f.shape}  test {X_te_f.shape}")
    print(f"  {len(DECISIONS)} cleaning decisions logged -> outputs/tables/p3a_titanic_prep_decisions.json")
    print("  NOTE: these files exist for inspection. The Phase 4 model refits every")
    print("  transform inside a sklearn Pipeline so CV folds never share statistics.")


if __name__ == "__main__":
    main()
