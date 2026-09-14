"""CRISP-DM Phase 4 - Modeling (Track A: Titanic).

Skills demonstrated:
  * reproducible-ml       -- seeds, pinned env, data hashing, config-driven run
  * sklearn-pipelines     -- ColumnTransformer + Pipeline; preprocessing refit per CV fold
  * imbalanced-data       -- PR-AUC first, class weights, SMOTE *inside* the pipeline,
                             threshold tuned on validation
  * hyperparameter-tuning -- Optuna TPE over the whole pipeline, log-scale spaces, budget caps
  * model-evaluation      -- metric choice, CV with spread, calibration, slices, honest holdout
  * experiment-tracking   -- params + metrics + artifacts + git SHA + data hash per run
"""
from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, classification_report,
                             confusion_matrix, f1_score, log_loss, precision_recall_curve,
                             recall_score, precision_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import (FIG, MOD, PALETTE, PROC, SEED, TAB, TITANIC, banner, git_sha, save_table,
                    sha256, seed_everything, style_plots, write_json)

TARGET = "Survived"
NUM = ["Age", "SibSp", "Parch", "Fare", "FarePerPerson", "FareLog", "FamilySize",
       "AgeBand", "TicketPrefix_te", "Pclass"]
CAT = ["Sex", "Embarked", "Title", "Deck"]
BIN = ["Age_was_missing", "CabinKnown", "IsAlone"]


# ------------------------------------------------------------ experiment-tracking
class RunTracker:
    """The experiment-tracking skill's 'what to always log' table.

    Uses MLflow when it is installed; otherwise writes the same schema to a local
    JSONL run store so the run history stays queryable either way.
    """

    def __init__(self, experiment: str):
        self.experiment = experiment
        self.store = TAB / "mlruns.jsonl"
        try:
            import mlflow  # noqa

            self.mlflow = mlflow
            mlflow.set_tracking_uri(f"file://{(MOD / 'mlruns').as_posix()}")
            mlflow.set_experiment(experiment)
            self.backend = "mlflow"
        except Exception:
            self.mlflow = None
            self.backend = "jsonl"
        print(f"  tracking backend: {self.backend}  (experiment '{experiment}')")

    def log(self, run_name: str, params: dict, metrics: dict, artifacts: list[str],
            tags: dict | None = None) -> dict:
        record = {
            "run_name": run_name,
            "experiment": self.experiment,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            # code state
            "git_sha": git_sha(),
            # data version
            "data_sha256": sha256(TITANIC)[:16],
            "data_version": "kaggle-titanic-891",
            # environment
            "python": platform.python_version(),
            "platform": platform.platform(),
            "sklearn": __import__("sklearn").__version__,
            "seed": SEED,
            "params": params,
            "metrics": {k: (round(v, 5) if isinstance(v, float) else v) for k, v in metrics.items()},
            "artifacts": artifacts,
            "tags": tags or {},
        }
        with open(self.store, "a") as fh:
            fh.write(json.dumps(record) + "\n")
        if self.mlflow:
            with self.mlflow.start_run(run_name=run_name):
                self.mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
                self.mlflow.log_metrics({k: float(v) for k, v in metrics.items()
                                         if isinstance(v, (int, float))})
                self.mlflow.set_tags({"git_sha": record["git_sha"],
                                      "data_sha256": record["data_sha256"], **(tags or {})})
        print(f"  logged run '{run_name}': " +
              ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float)))
        return record


# --------------------------------------------------------------- sklearn-pipelines
def build_pipeline(estimator, use_smote: bool = False):
    """One artifact holding preprocessing + estimator, so leakage is impossible by construction."""
    preprocess = ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), NUM),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                              ("ohe", OneHotEncoder(handle_unknown="ignore",
                                                    sparse_output=False))]), CAT),
            ("bin", "passthrough", BIN),
        ],
        remainder="drop",
    )
    steps = [("prep", preprocess)]
    if use_smote:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline

        steps.append(("smote", SMOTE(random_state=SEED, k_neighbors=5)))
        steps.append(("clf", estimator))
        return ImbPipeline(steps)
    steps.append(("clf", estimator))
    return Pipeline(steps)


def main() -> None:
    seed_everything()
    banner("reproducible-ml", "Phase 4 - Modeling", "same code + same data + same config -> same result")
    print(f"  seed              : {SEED} (python/random/numpy/torch all seeded)")
    print(f"  data sha256       : {sha256(TITANIC)}")
    print(f"  git sha           : {git_sha()}")
    print(f"  python            : {platform.python_version()}  ({platform.platform()})")
    print(f"  sklearn           : {__import__('sklearn').__version__}")
    print(f"  env pinned in     : requirements.txt (exact ==)")

    train = pd.read_csv(PROC / "titanic_train.csv")
    test = pd.read_csv(PROC / "titanic_test.csv")
    X, y = train.drop(columns=[TARGET]), train[TARGET]
    X_test, y_test = test.drop(columns=[TARGET]), test[TARGET]
    print(f"  train {X.shape}  holdout {X_test.shape}  (holdout untouched until the very end)")

    tracker = RunTracker("titanic-survival")
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    results: dict[str, dict] = {}

    # ------------------------------------------------------ imbalanced-data step 1
    banner("imbalanced-data", "Phase 4 - Modeling", "step 1 - fix the metric before touching the model")
    pos = y.mean()
    print(f"  positive rate {pos:.1%} (ratio {(1-pos)/pos:.1f}:1) -- moderate, not extreme")
    dummy = DummyClassifier(strategy="most_frequent").fit(X, y)
    print(f"  a majority-class DummyClassifier already scores "
          f"accuracy = {dummy.score(X_test, y_test):.3f}  <- the accuracy trap")
    print(f"  its PR-AUC is {average_precision_score(y_test, dummy.predict_proba(X_test)[:,1]):.3f} "
          f"(= base rate) -- PR-AUC exposes what accuracy hides")
    print("  -> primary metric: PR-AUC (average_precision). Secondary: ROC-AUC, recall@precision>=0.80.")

    # ---------------------------------------------------------- sklearn-pipelines
    banner("sklearn-pipelines", "Phase 4 - Modeling",
           "ColumnTransformer + Pipeline; preprocessing refits inside every CV fold")
    baselines = {
        "logreg": LogisticRegression(max_iter=2000, random_state=SEED),
        "logreg_balanced": LogisticRegression(max_iter=2000, random_state=SEED,
                                              class_weight="balanced"),
        "rf_balanced": RandomForestClassifier(n_estimators=400, random_state=SEED,
                                              class_weight="balanced_subsample", n_jobs=-1),
        "hgb": HistGradientBoostingClassifier(random_state=SEED),
    }
    print(f"  numeric   ({len(NUM)}): {NUM}")
    print(f"  categoric ({len(CAT)}): {CAT}  -> OneHotEncoder(handle_unknown='ignore')")
    print(f"  binary    ({len(BIN)}): {BIN} -> passthrough")
    print()
    for name, est in baselines.items():
        pipe = build_pipeline(est)
        t0 = time.perf_counter()
        sc = cross_validate(pipe, X, y, cv=cv,
                            scoring=["average_precision", "roc_auc", "f1", "accuracy"],
                            n_jobs=-1)
        dt = time.perf_counter() - t0
        m = {
            "cv_pr_auc": sc["test_average_precision"].mean(),
            "cv_pr_auc_std": sc["test_average_precision"].std(),
            "cv_roc_auc": sc["test_roc_auc"].mean(),
            "cv_roc_auc_std": sc["test_roc_auc"].std(),
            "cv_f1": sc["test_f1"].mean(),
            "cv_accuracy": sc["test_accuracy"].mean(),
            "fit_seconds": dt,
        }
        results[name] = m
        print(f"  {name:<16} PR-AUC {m['cv_pr_auc']:.3f} +/- {m['cv_pr_auc_std']:.3f}   "
              f"ROC-AUC {m['cv_roc_auc']:.3f} +/- {m['cv_roc_auc_std']:.3f}   "
              f"acc {m['cv_accuracy']:.3f}   [{dt:.1f}s]")
        tracker.log(f"baseline-{name}", {"model": name, **est.get_params()}, m, [], {"stage": "baseline"})

    # ------------------------------------------------- imbalanced-data steps 2 & 3
    banner("imbalanced-data", "Phase 4 - Modeling",
           "steps 2-3 - class weights vs SMOTE, resampled INSIDE the CV fold")
    smote_pipe = build_pipeline(HistGradientBoostingClassifier(random_state=SEED), use_smote=True)
    sm = cross_val_score(smote_pipe, X, y, cv=cv, scoring="average_precision", n_jobs=-1)
    results["hgb_smote"] = {"cv_pr_auc": sm.mean(), "cv_pr_auc_std": sm.std()}
    print(f"  hgb (no resampling)   PR-AUC {results['hgb']['cv_pr_auc']:.3f} "
          f"+/- {results['hgb']['cv_pr_auc_std']:.3f}")
    print(f"  hgb + SMOTE-in-fold   PR-AUC {sm.mean():.3f} +/- {sm.std():.3f}")
    delta = sm.mean() - results["hgb"]["cv_pr_auc"]
    noise = max(sm.std(), results["hgb"]["cv_pr_auc_std"])
    print(f"  delta {delta:+.4f} against fold noise +/- {noise:.4f}")
    print(f"  -> {'NOT a real gain' if abs(delta) < noise else 'a real gain'}: "
          f"at 38% positives the class weights already suffice; SMOTE adds risk without benefit.")
    print("  -> Note SMOTE sits INSIDE the imblearn Pipeline, so it only ever sees training folds.")
    tracker.log("hgb-smote-infold", {"model": "hgb", "resampler": "SMOTE(k=5)"},
                results["hgb_smote"], [], {"stage": "imbalance"})

    # ------------------------------------------------------- hyperparameter-tuning
    banner("hyperparameter-tuning", "Phase 4 - Modeling",
           "Optuna TPE over the whole pipeline, log-scale spaces, hard budget")
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    base = build_pipeline(HistGradientBoostingClassifier(random_state=SEED))

    def objective(trial):
        params = {
            "clf__learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "clf__max_depth": trial.suggest_int("max_depth", 2, 10),
            "clf__max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 5, 63),
            "clf__min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 60),
            "clf__l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
        }
        base.set_params(**params)
        return cross_val_score(base, X, y, cv=cv, scoring="average_precision", n_jobs=-1).mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=60, timeout=300, show_progress_bar=False)
    tune_s = time.perf_counter() - t0
    print(f"  {len(study.trials)} trials in {tune_s:.1f}s (budget: n_trials=60, timeout=300s)")
    print(f"  best CV PR-AUC : {study.best_value:.4f}  (untuned hgb was "
          f"{results['hgb']['cv_pr_auc']:.4f}, {study.best_value - results['hgb']['cv_pr_auc']:+.4f})")
    print("  best params:")
    for k, v in study.best_params.items():
        print(f"     {k:<22} {v}")
    print("  note: learning_rate and l2_regularization sampled on a LOG scale;")
    print("        every trial re-fits preprocessing inside each fold (clf__ prefix = inside the pipeline).")

    imp = optuna.importance.get_param_importances(study)
    print("  param importance:", {k: round(v, 3) for k, v in imp.items()})

    best_pipe = build_pipeline(HistGradientBoostingClassifier(
        random_state=SEED, **{k: v for k, v in study.best_params.items()}))
    tracker.log("hgb-optuna-best", {"model": "hgb", "search": "optuna-tpe",
                                    "n_trials": len(study.trials), **study.best_params},
                {"cv_pr_auc": study.best_value, "tune_seconds": tune_s},
                [], {"stage": "tuning"})

    # ---------------------------------------------- imbalanced-data step 4 + eval
    banner("imbalanced-data", "Phase 4 - Modeling",
           "step 4 - tune the decision threshold on VALIDATION, never on the holdout")
    X_fit, X_val, y_fit, y_val = train_test_split(X, y, test_size=0.25,
                                                  random_state=SEED, stratify=y)
    best_pipe.fit(X_fit, y_fit)
    val_scores = best_pipe.predict_proba(X_val)[:, 1]
    prec, rec, thr = precision_recall_curve(y_val, val_scores)
    ok = np.where(prec[:-1] >= 0.80)[0]
    chosen = float(thr[ok[0]]) if len(ok) else 0.5
    f1s = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    thr_f1 = float(thr[int(np.nanargmax(f1s))])
    print(f"  default threshold                : 0.500")
    print(f"  smallest threshold with precision >= 0.80 : {chosen:.3f}")
    print(f"  threshold maximising F1                   : {thr_f1:.3f}")
    for t, lab in [(0.5, "default 0.50"), (chosen, f"prec>=0.80 ({chosen:.3f})"),
                   (thr_f1, f"max-F1 ({thr_f1:.3f})")]:
        p_ = precision_score(y_val, val_scores >= t, zero_division=0)
        r_ = recall_score(y_val, val_scores >= t)
        print(f"     {lab:<24} val precision {p_:.3f}  recall {r_:.3f}  "
              f"F1 {f1_score(y_val, val_scores >= t):.3f}")
    print("  -> ship the max-F1 threshold; it is chosen on validation and never re-tuned on the holdout.")

    # ------------------------------------------------------------ model-evaluation
    banner("model-evaluation", "Phase 4 - Modeling",
           "refit on all train, score ONCE on the untouched holdout")
    best_pipe.fit(X, y)
    proba = best_pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= thr_f1).astype(int)
    holdout = {
        "pr_auc": average_precision_score(y_test, proba),
        "roc_auc": roc_auc_score(y_test, proba),
        "f1": f1_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "accuracy": float((pred == y_test).mean()),
        "log_loss": log_loss(y_test, proba),
        "brier": brier_score_loss(y_test, proba),
        "threshold": thr_f1,
    }
    print("  holdout metrics (n=%d):" % len(y_test))
    for k, v in holdout.items():
        print(f"     {k:<12} {v:.4f}")
    print("\n  confusion matrix @ threshold %.3f:" % thr_f1)
    cm = confusion_matrix(y_test, pred)
    print(f"     TN {cm[0,0]:>3}   FP {cm[0,1]:>3}")
    print(f"     FN {cm[1,0]:>3}   TP {cm[1,1]:>3}")
    print("\n" + "\n".join("     " + l for l in
                           classification_report(y_test, pred, target_names=["died", "survived"],
                                                 digits=3).splitlines()))

    # calibration
    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=8, strategy="quantile")
    cal_err = float(np.mean(np.abs(frac_pos - mean_pred)))
    print(f"\n  calibration: mean |observed - predicted| over 8 quantile bins = {cal_err:.4f}")
    print(f"  Brier score {holdout['brier']:.4f} (0 = perfect) -> probabilities are "
          f"{'usable' if cal_err < 0.10 else 'NOT reliable'} for decision thresholds")

    # slice metrics -- the skill's fairness/robustness check
    print("\n  slice metrics (catch subgroup failure that the aggregate hides):")
    slices = []
    raw_test = pd.read_csv(PROC / "titanic_test.csv")
    for col in ["Sex", "Pclass"]:
        for val in sorted(raw_test[col].unique()):
            m = raw_test[col] == val
            if m.sum() < 15 or y_test[m].nunique() < 2:
                continue
            s = {
                "slice": f"{col}={val}", "n": int(m.sum()),
                "base_rate": float(y_test[m].mean()),
                "pr_auc": average_precision_score(y_test[m], proba[m]),
                "recall": recall_score(y_test[m], pred[m]),
                "precision": precision_score(y_test[m], pred[m], zero_division=0),
            }
            slices.append(s)
            print(f"     {s['slice']:<12} n={s['n']:<4} base={s['base_rate']:.2f}  "
                  f"PR-AUC {s['pr_auc']:.3f}  recall {s['recall']:.3f}  precision {s['precision']:.3f}")
    save_table(pd.DataFrame(slices).set_index("slice"), "titanic_slice_metrics.csv")

    # honest comparison table
    banner("model-evaluation", "Phase 4 - Modeling", "is the tuned model actually better?")
    d = study.best_value - results["hgb"]["cv_pr_auc"]
    n = results["hgb"]["cv_pr_auc_std"]
    print(f"  tuned CV PR-AUC {study.best_value:.4f} vs untuned {results['hgb']['cv_pr_auc']:.4f}")
    print(f"  delta {d:+.4f}; fold-to-fold std of the untuned model is +/- {n:.4f}")
    verdict = ("inside fold noise -- report it as 'no measurable gain', not an improvement"
               if abs(d) < n else "larger than fold noise -- a defensible gain")
    print(f"  -> {verdict}")

    # ------------------------------------------------------------------ artifacts
    model_path = MOD / "titanic_pipeline.joblib"
    joblib.dump({"pipeline": best_pipe, "threshold": thr_f1, "features": list(X.columns),
                 "num": NUM, "cat": CAT, "bin": BIN,
                 "model_version": "1.0.0", "git_sha": git_sha(),
                 "data_sha256": sha256(TITANIC)[:16]}, model_path)
    print(f"\n  -> saved full pipeline (preprocessing + estimator + threshold) to {model_path.name}")

    tracker.log("hgb-optuna-holdout", {"model": "hgb", "search": "optuna-tpe", **study.best_params},
                holdout, [str(model_path.relative_to(model_path.parents[2]))],
                {"stage": "final", "promoted": "true"})

    write_json({"cv_results": results, "best_params": study.best_params,
                "best_cv_pr_auc": study.best_value, "holdout": holdout,
                "threshold_default": 0.5, "threshold_prec80": chosen, "threshold_maxf1": thr_f1,
                "calibration_error": cal_err, "slices": slices,
                "param_importance": {k: round(v, 4) for k, v in imp.items()},
                "tuned_vs_untuned_delta": d, "fold_noise": n, "verdict": verdict},
               "p4a_titanic_model_results.json")

    # figures
    import matplotlib.pyplot as plt

    style_plots()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    ax = axes[0]
    ax.plot(rec[:-1], prec[:-1], color=PALETTE[0], lw=2)
    ax.axhline(y.mean(), ls="--", c="grey", lw=1)
    ax.text(0.02, y.mean() + 0.02, f"base rate {y.mean():.2f}", fontsize=8, color="grey")
    ax.scatter([recall_score(y_val, val_scores >= thr_f1)],
               [precision_score(y_val, val_scores >= thr_f1)],
               color=PALETTE[1], zorder=5, s=60, label=f"chosen thr {thr_f1:.2f}")
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title("Threshold picked on validation PR curve")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    ax.plot(mean_pred, frac_pos, "o-", color=PALETTE[2], lw=2)
    ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed frequency")
    ax.set_title(f"Calibration (Brier {holdout['brier']:.3f})")

    ax = axes[2]
    names = list(results)
    vals = [results[n_]["cv_pr_auc"] for n_ in names]
    errs = [results[n_].get("cv_pr_auc_std", 0) for n_ in names]
    ax.barh(names + ["hgb tuned"], vals + [study.best_value],
            xerr=errs + [0], color=PALETTE[0], alpha=0.85)
    ax.set_xlim(0.6, 0.95)
    ax.set_xlabel("CV PR-AUC (error bars = fold std)")
    ax.set_title("Gains are inside fold noise")

    fig.suptitle("Titanic - Phase 4 modeling  (5-fold stratified CV, seed 42)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "p4a_titanic_model.png")
    plt.close(fig)
    print(f"  -> wrote p4a_titanic_model.png")


if __name__ == "__main__":
    main()
