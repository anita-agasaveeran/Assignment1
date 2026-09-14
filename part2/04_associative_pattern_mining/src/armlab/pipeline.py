"""End-to-end CRISP-DM pipeline.

Phase 1 Business Understanding  -> reports/business_understanding.md (static)
Phase 2 Data Understanding      -> data_profile.json
Phase 3 Data Preparation        -> preparation.json
Phase 4 Modeling                -> itemsets.json, rules.json, autoresearch.json
Phase 5 Evaluation              -> evaluation.json, benchmark.json, properties.json
Phase 6 Deployment              -> model_card.json + the FastAPI service
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from armlab.autoresearch import hillclimb, space
from armlab.autoresearch import objective as objective_mod
from armlab.autoresearch.objective import TERM_DOC, Score, score as score_fn, weights
from armlab.config import Config, provenance
from armlab.data import acquire, prepare, profile as profiling
from armlab.data.prepare import BasketDataset
from armlab.evaluation import holdout as holdout_eval
from armlab.metrics.interestingness import glossary
from armlab.metrics.properties import property_matrix
from armlab.mining import rules as rules_mod
from armlab.mining.apriori import apriori, eclat
from armlab.mining.fpgrowth import fpgrowth
from armlab.stats import significance as sig


def _log(msg: str) -> None:
    print(f"  {msg}", flush=True)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, (frozenset, set)):
        return sorted(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    return obj


def write_artifact(cfg: Config, name: str, payload: Any) -> Path:
    out = cfg.paths.resolve("artifacts") / name
    out.write_text(json.dumps(_jsonable(payload), indent=2))
    return out


# ---------------------------------------------------------------------------
class ItemsetCache:
    """Memoises FP-Growth by (min_count, max_len) across AutoResearch trials.

    Trials that differ only in confidence/lift thresholds do not need re-mining.
    The *recorded* wall clock of the original mine is replayed on a hit so the
    objective's cost term still reflects what the configuration really costs.
    """

    def __init__(self, transactions: list[list[int]]) -> None:
        self.tx = transactions
        self.store: dict[tuple[int, int], tuple[dict, Any, float]] = {}
        self.hits = 0
        self.misses = 0

    def get(self, min_count: int, max_len: int):
        key = (min_count, max_len)
        if key in self.store:
            self.hits += 1
            return (*self.store[key], True)
        self.misses += 1
        t0 = time.perf_counter()
        itemsets, stats = fpgrowth(self.tx, min_count, max_len)
        secs = time.perf_counter() - t0
        self.store[key] = (itemsets, stats, secs)
        return itemsets, stats, secs, False


# ---------------------------------------------------------------------------
def run(cfg: Config, autoresearch: bool = True,
        run_null_baseline: bool = True,
        on_progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    log = on_progress or _log
    t_pipeline = time.perf_counter()
    summary: dict[str, Any] = {"provenance": provenance(cfg)}

    # ---------------- Phase 2: Data Understanding --------------------------
    log("Phase 2/6  Data Understanding: loading raw transaction log")
    raw = acquire.load_raw(cfg)
    log(f"           {len(raw):,} rows x {raw.shape[1]} columns")
    prof = profiling.profile(raw, cfg)
    write_artifact(cfg, "data_profile.json", prof)
    log(f"           {prof['n_invoices']:,} invoices, {prof['n_items']:,} items, "
        f"{prof['n_customers']:,} customers, {prof['span_days']} days")

    # ---------------- Phase 3: Data Preparation ----------------------------
    log("Phase 3/6  Data Preparation: cleaning + basket construction")
    cleaned, drop_log = prepare.clean(raw, cfg)
    ds = prepare.build_baskets(cleaned, cfg)
    train, test = prepare.split(ds, cfg)
    log(f"           {ds.n_transactions:,} baskets, {ds.n_items:,} items, "
        f"avg size {ds.meta['avg_basket_size']:.2f}")
    log(f"           split[{cfg.data.split_strategy}] explore={train.n_transactions:,} "
        f"holdout={test.n_transactions:,}")

    prep_payload = {
        "drop_log": drop_log,
        "prep_log": ds.meta["prep_log"],
        "dataset": {
            "n_transactions": ds.n_transactions, "n_items": ds.n_items,
            "avg_basket_size": ds.meta["avg_basket_size"],
            "density": float(ds.meta["avg_basket_size"] / max(ds.n_items, 1)),
        },
        "split": {
            "strategy": cfg.data.split_strategy,
            "holdout_fraction": cfg.data.holdout_fraction,
            "explore_n": train.n_transactions, "holdout_n": test.n_transactions,
            "explore_start": str(train.timestamps[0]) if train.n_transactions else None,
            "explore_end": str(train.timestamps[-1]) if train.n_transactions else None,
            "holdout_start": str(test.timestamps[0]) if test.n_transactions else None,
            "holdout_end": str(test.timestamps[-1]) if test.n_transactions else None,
            "rationale": prepare.split.__doc__,
        },
        "top_items": [
            {"item_id": int(i), "code": ds.item_codes[i], "label": ds.item_labels[i],
             "count": int(ds.item_counts[i]),
             "support": float(ds.item_counts[i] / ds.n_transactions)}
            for i in range(min(30, ds.n_items))
        ],
    }
    write_artifact(cfg, "preparation.json", prep_payload)
    summary["preparation"] = prep_payload["dataset"] | prep_payload["split"]

    # Holdout incidence matrix, computed once and shared by every trial.
    hmat = holdout_eval.incidence_matrix(test, ds.n_items)
    cache = ItemsetCache(train.transactions)

    def evaluate_params(params: dict[str, Any]) -> tuple[Score, dict[str, Any]]:
        trial_cfg = cfg.with_mining(**params)
        min_count = max(1, int(np.ceil(trial_cfg.mining.min_support * train.n_transactions)))
        itemsets, mstats, mine_secs, hit = cache.get(min_count,
                                                     trial_cfg.mining.max_itemset_len)
        t0 = time.perf_counter()
        res = rules_mod.generate_rules(train, trial_cfg, itemsets=itemsets,
                                       stats=mstats, run_significance=True)
        rep = holdout_eval.evaluate(res.rules, test, trial_cfg, ds.n_items, mat=hmat)
        secs = mine_secs + (time.perf_counter() - t0)
        sc = score_fn(res.rules, rep, secs, cfg.autoresearch,
                      diagnostics={"min_count": min_count,
                                   "n_itemsets": len(itemsets),
                                   "n_candidate_rules": res.n_candidate_rules})
        return sc, {"cache_hit": hit, "mine_seconds": round(mine_secs, 4),
                    "coverage": round(rep["coverage"], 5),
                    "replication_rate": round(rep["replication_rate"], 5)}

    # ---------------- Phase 4: Modeling ------------------------------------
    log("Phase 4/6  Modeling: baseline mine at hand-set thresholds")
    baseline_params = {
        "min_support": cfg.mining.min_support,
        "min_confidence": cfg.mining.min_confidence,
        "min_lift": cfg.mining.min_lift,
        "max_antecedent_len": cfg.mining.max_antecedent_len,
        "max_itemset_len": cfg.mining.max_itemset_len,
    }
    t_base = time.perf_counter()
    base_score, _ = evaluate_params(baseline_params)
    base_seconds = time.perf_counter() - t_base
    # Calibrate the cost term to this dataset before the search starts, then
    # re-score the baseline on the same scale so the comparison is apples to
    # apples.
    objective_mod.set_runtime_budget(5.0 * base_seconds)
    base_score, _ = evaluate_params(baseline_params)
    log(f"           cost term calibrated: runtime budget "
        f"{objective_mod.RUNTIME_BUDGET_S:.2f}s (5x baseline {base_seconds:.2f}s)")
    log(f"           baseline objective={base_score.value:.4f} "
        f"rules={base_score.diagnostics.get('n_rules', 0)} "
        f"validated={base_score.diagnostics.get('n_validated', 0)}")

    ar_payload: dict[str, Any] = {"enabled": False}
    champion_params = dict(baseline_params)

    if autoresearch and cfg.autoresearch.enabled:
        log(f"Phase 4/6  AutoResearch: hill climbing, budget "
            f"{cfg.autoresearch.max_evaluations} evaluations, "
            f"{cfg.autoresearch.restarts} restarts")
        result = hillclimb.search(
            evaluate_params, cfg.autoresearch, baseline_params,
            ledger_path=cfg.paths.resolve("artifacts") / "autoresearch_ledger.jsonl",
            progress=lambda m: log("           " + m))
        champion_params = result.best_params
        log(f"           champion objective={result.best_score:.4f} "
            f"(+{result.improvement_over_seed:.4f} over seed) in "
            f"{result.seconds:.1f}s")
        ar_payload = {
            "enabled": True,
            "algorithm": "stochastic hill climbing + adaptive step + tabu + random restarts",
            "algorithm_docstring": hillclimb.__doc__,
            "config": cfg.autoresearch.__dict__,
            "space": space.describe(),
            "objective": {
                "weights": weights(cfg.autoresearch),
                "terms": TERM_DOC,
                "target_rules": objective_mod.TARGET_RULES,
                "runtime_budget_seconds": objective_mod.RUNTIME_BUDGET_S,
            },
            "seed_params": baseline_params,
            "seed_score": result.seed_score,
            "best_params": result.best_params,
            "best_score": result.best_score,
            "improvement_over_seed": result.improvement_over_seed,
            "improvement_pct": (100 * result.improvement_over_seed / result.seed_score
                                if result.seed_score else 0.0),
            "n_evaluations": result.n_evaluations,
            "n_restarts_used": result.n_restarts_used,
            "seconds": result.seconds,
            "convergence": result.convergence,
            "trials": result.ledger.to_dict(),
            "cache": {"hits": cache.hits, "misses": cache.misses},
        }
    write_artifact(cfg, "autoresearch.json", ar_payload)

    # ---------------- Champion model ---------------------------------------
    log("Phase 4/6  Champion refit + full rule generation")
    champ_cfg = cfg.with_mining(**champion_params)
    min_count = max(1, int(np.ceil(champ_cfg.mining.min_support * train.n_transactions)))
    itemsets, mstats, mine_secs, _ = cache.get(min_count, champ_cfg.mining.max_itemset_len)
    champ = rules_mod.generate_rules(train, champ_cfg, itemsets=itemsets, stats=mstats)
    hold_report = holdout_eval.evaluate(champ.rules, test, champ_cfg, ds.n_items, mat=hmat)
    champ_rules = holdout_eval.attach(champ.rules, hold_report)
    champ_score = score_fn(champ.rules, hold_report, mine_secs + champ.seconds_rules,
                           cfg.autoresearch)
    log(f"           {len(champ_rules):,} rules | "
        f"{hold_report['n_validated']:,} holdout-validated | "
        f"coverage {hold_report['coverage']:.1%}")

    write_artifact(cfg, "rules.json", {
        "params": champion_params,
        "min_count": min_count,
        "n_rules": int(len(champ_rules)),
        "diagnostics": champ.diagnostics,
        "objective": champ_score.to_dict(),
        "rules": rules_mod.humanise(champ_rules, train),
    })

    # Frequent itemsets artifact
    lattice = sorted(itemsets.items(), key=lambda kv: -kv[1])
    write_artifact(cfg, "itemsets.json", {
        "min_support": champ_cfg.mining.min_support,
        "min_count": min_count,
        "n_itemsets": len(itemsets),
        "per_level": mstats.per_level,
        "algorithm_stats": mstats.__dict__,
        "top": [
            {"items": [train.label(i) for i in sorted(s)],
             "codes": [train.item_codes[i] for i in sorted(s)],
             "size": len(s), "count": int(c),
             "support": float(c / train.n_transactions)}
            for s, c in lattice[:300] if len(s) >= 2
        ],
    })

    # ---------------- Phase 5: Evaluation ----------------------------------
    log("Phase 5/6  Evaluation: algorithm benchmark, null baseline, measure properties")
    bench = benchmark_algorithms(train, min_count, champ_cfg.mining.max_itemset_len)
    for b in bench["runs"]:
        log(f"           {b['algorithm']:<9} {b['seconds']:.3f}s "
            f"{b['itemsets_found']:,} itemsets  agree={b['agrees_with_fpgrowth']}")

    null_payload: dict[str, Any] = {"enabled": False}
    if run_null_baseline:
        null_payload = null_baseline(train, champ_cfg, ds.n_items, hmat, test,
                                     len(champ_rules))
        log(f"           swap-randomised null: {null_payload['n_rules_null']:,} rules "
            f"vs {len(champ_rules):,} real "
            f"(empirical FDR {null_payload['empirical_fdr']:.3f})")

    props = property_matrix()
    write_artifact(cfg, "properties.json",
                   props | {"glossary": glossary()})

    evaluation_payload = {
        "baseline": {"params": baseline_params, "objective": base_score.to_dict()},
        "champion": {"params": champion_params, "objective": champ_score.to_dict()},
        "holdout": {k: v for k, v in hold_report.items() if k != "per_rule"},
        "null_baseline": null_payload,
        "benchmark": bench,
        "significance": champ.diagnostics.get("significance", {}),
        "lift_calibration": lift_calibration(champ_rules),
        "sensitivity": sensitivity_curves(train, test, cfg, ds.n_items, hmat,
                                          champion_params, cache),
    }
    write_artifact(cfg, "evaluation.json", evaluation_payload)
    write_artifact(cfg, "network.json", rule_network(champ_rules, train))

    # ---------------- Phase 6: Deployment ----------------------------------
    log("Phase 6/6  Deployment: model card + run summary")
    card = model_card(cfg, champion_params, champ_rules, hold_report, prof,
                      champ_score, ar_payload, null_payload)
    write_artifact(cfg, "model_card.json", card)

    summary |= {
        "phases_completed": 6,
        "seconds_total": time.perf_counter() - t_pipeline,
        "n_rules": int(len(champ_rules)),
        "n_validated": hold_report["n_validated"],
        "coverage": hold_report["coverage"],
        "baseline_objective": base_score.value,
        "champion_objective": champ_score.value,
        "champion_params": champion_params,
        "autoresearch_enabled": bool(ar_payload.get("enabled")),
    }
    write_artifact(cfg, "run.json", summary)
    log(f"Done in {summary['seconds_total']:.1f}s -> "
        f"{cfg.paths.resolve('artifacts')}")
    return summary


# ---------------------------------------------------------------------------
def benchmark_algorithms(train, min_count: int, max_len: int) -> dict[str, Any]:
    """FP-Growth vs Apriori vs Eclat: same input, same thresholds, same answer."""
    runs = []
    ref: dict | None = None
    for name, fn in (("fpgrowth", fpgrowth), ("apriori", apriori), ("eclat", eclat)):
        try:
            sets, st = fn(train.transactions, min_count, max_len)
        except Exception as exc:                       # pragma: no cover
            runs.append({"algorithm": name, "error": str(exc)})
            continue
        if ref is None:
            ref = sets
            agrees = True
        else:
            agrees = (sets == ref)
        runs.append({
            "algorithm": name, "seconds": round(st.seconds, 4),
            "itemsets_found": st.itemsets_found,
            "candidates_generated": st.candidates_generated,
            "nodes_created": st.nodes_created,
            "conditional_trees": st.conditional_trees,
            "passes": st.passes, "max_depth": st.max_depth,
            "agrees_with_fpgrowth": bool(agrees),
        })
    fastest = min((r for r in runs if "seconds" in r), key=lambda r: r["seconds"],
                  default=None)
    return {"min_count": min_count, "max_itemset_len": max_len, "runs": runs,
            "fastest": fastest["algorithm"] if fastest else None,
            "all_agree": all(r.get("agrees_with_fpgrowth") for r in runs
                             if "seconds" in r),
            "note": "All three are exact algorithms, so identical output is a "
                    "correctness requirement, not a coincidence. Divergence "
                    "would mean a bug in the from-scratch FP-Growth."}


def null_baseline(train, champ_cfg: Config, n_items: int, hmat, test,
                  n_real_rules: int) -> dict[str, Any]:
    """Mine the same thresholds on margin-preserving randomised data."""
    surrogate = sig.swap_randomisation_null(train.transactions, n_items, seed=7)
    ds_null = BasketDataset(
        transactions=surrogate, item_labels=train.item_labels,
        item_codes=train.item_codes, item_counts=train.item_counts,
        basket_ids=train.basket_ids, timestamps=train.timestamps, split="null")
    res = rules_mod.generate_rules(ds_null, champ_cfg, run_significance=True)
    n_null = int(len(res.rules))
    n_null_sig = int(res.rules["significant_searchspace"].sum()) if n_null else 0
    return {
        "enabled": True,
        "method": "swap randomisation preserving basket sizes and item "
                  "frequencies (Gionis et al., ACM TKDD 2007)",
        "n_swaps_performed": int(getattr(sig.swap_randomisation_null, "last_swaps", 0)),
        "n_rules_real": int(n_real_rules),
        "n_rules_null": n_null,
        "n_significant_null": n_null_sig,
        "empirical_fdr": float(n_null / n_real_rules) if n_real_rules else 0.0,
        "interpretation": "Rules found on data whose item frequencies and basket "
                          "sizes match the real data but whose co-occurrence "
                          "structure has been destroyed are, by construction, "
                          "false discoveries. The ratio is a distribution-free "
                          "check on the analytic FDR.",
    }


def lift_calibration(rules: pd.DataFrame, bins: int = 8) -> list[dict[str, Any]]:
    """Do rules that promise lift X actually deliver lift X out of sample?"""
    if len(rules) == 0 or "holdout_lift" not in rules:
        return []
    tr = np.clip(rules["lift"].to_numpy(), 0, 50)
    ho = np.clip(np.asarray(rules["holdout_lift"], dtype=float), 0, 50)
    edges = np.quantile(tr, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    out = []
    for i in range(len(edges) - 1):
        m = (tr >= edges[i]) & (tr <= edges[i + 1] if i == len(edges) - 2 else tr < edges[i + 1])
        if m.sum() == 0:
            continue
        out.append({"bin": i, "explore_lift_mean": float(tr[m].mean()),
                    "holdout_lift_mean": float(ho[m].mean()),
                    "n": int(m.sum()),
                    "range": [float(edges[i]), float(edges[i + 1])]})
    return out


def sensitivity_curves(train, test, cfg: Config, n_items: int, hmat,
                       champion: dict[str, Any], cache: ItemsetCache) -> dict[str, Any]:
    """One-at-a-time sweeps around the champion -- the dashboard's tornado view.

    Local sensitivity is what tells a reviewer whether the champion sits on a
    plateau (robust) or a spike (an overfit artefact of the search).
    """
    out: dict[str, Any] = {}
    sweeps = {
        "min_support": np.round(np.geomspace(0.005, 0.08, 9), 5).tolist(),
        "min_confidence": np.round(np.linspace(0.05, 0.85, 9), 4).tolist(),
        "min_lift": np.round(np.linspace(1.0, 3.5, 9), 4).tolist(),
    }
    for dim, values in sweeps.items():
        pts = []
        for v in values:
            params = dict(champion); params[dim] = v
            tcfg = cfg.with_mining(**params)
            mc = max(1, int(np.ceil(tcfg.mining.min_support * train.n_transactions)))
            itemsets, st, ms, _ = cache.get(mc, tcfg.mining.max_itemset_len)
            res = rules_mod.generate_rules(train, tcfg, itemsets=itemsets, stats=st)
            rep = holdout_eval.evaluate(res.rules, test, tcfg, n_items, mat=hmat)
            sc = score_fn(res.rules, rep, ms + res.seconds_rules, cfg.autoresearch)
            pts.append({"value": float(v), "objective": sc.value,
                        "n_rules": int(len(res.rules)),
                        "n_validated": int(sc.diagnostics.get("n_validated", 0)),
                        "coverage": rep["coverage"],
                        "seconds": round(ms + res.seconds_rules, 4)})
        out[dim] = pts
    return out


def rule_network(rules: pd.DataFrame, ds, max_rules: int = 120) -> dict[str, Any]:
    """Item graph for the dashboard's network view."""
    if len(rules) == 0:
        return {"nodes": [], "edges": []}
    sel = rules.head(max_rules)
    deg: dict[int, float] = {}
    edges = []
    for r in sel.itertuples(index=False):
        for a in r.antecedent:
            for b in r.consequent:
                w = float(min(r.lift, 25.0))
                edges.append({"source": int(a), "target": int(b), "lift": w,
                              "confidence": float(r.confidence),
                              "support": float(r.support),
                              "validated": bool(getattr(r, "validated", False))})
                deg[a] = deg.get(a, 0) + w
                deg[b] = deg.get(b, 0) + w
    nodes = [{"id": int(i), "label": ds.label(i), "code": ds.item_codes.get(i, ""),
              "weight": round(w, 3),
              "support": float(ds.item_counts[i] / ds.n_transactions)}
             for i, w in sorted(deg.items(), key=lambda kv: -kv[1])]
    return {"nodes": nodes, "edges": edges, "n_rules_shown": int(len(sel))}


def model_card(cfg: Config, params: dict[str, Any], rules: pd.DataFrame,
               hold: dict[str, Any], prof: dict[str, Any], sc: Score,
               ar: dict[str, Any], null: dict[str, Any]) -> dict[str, Any]:
    """Model card (Mitchell et al., FAT* 2019) adapted to an unsupervised
    pattern-mining artefact."""
    return {
        "model_details": {
            "name": "ARMLab market-basket rule set",
            "version": cfg.fingerprint(),
            "type": "Association rule set (unsupervised pattern mining)",
            "algorithm": "FP-Growth (Han, Pei & Yin, SIGMOD 2000), from scratch",
            "hyperparameters": params,
            "tuning": ("stochastic hill climbing with random restarts"
                       if ar.get("enabled") else "hand-set"),
            "provenance": provenance(cfg),
        },
        "intended_use": {
            "primary": "Cross-sell and bundle recommendations, planogram "
                       "adjacency, and basket-builder promotions for an online "
                       "gift retailer.",
            "users": "Merchandising analysts; a recommender service consuming the "
                     "rules as a cold-start co-purchase prior.",
            "out_of_scope": [
                "Causal claims -- these are associations from observational "
                "transaction data with no intervention.",
                "Individual-level targeting: rules are population-level and the "
                "dataset's CustomerID is missing on ~25% of rows.",
                "Any market outside the training period/geography without refit.",
            ],
        },
        "data": {
            "source": "Online Retail (UCI ML Repository 352; Kaggle "
                      "carrie1/ecommerce-data), Chen, Sain & Guo 2012",
            "period": [prof["date_min"], prof["date_max"]],
            "n_invoices_raw": prof["n_invoices"],
            "n_items_raw": prof["n_items"],
            "countries": prof["n_countries"],
            "known_issues": [g for g in prof["gates"] if g["status"] == "warn"],
        },
        "metrics": {
            "objective_value": sc.value,
            "objective_terms": sc.terms,
            "n_rules": int(len(rules)),
            "n_holdout_validated": hold["n_validated"],
            "replication_rate": hold["replication_rate"],
            "holdout_coverage": hold["coverage"],
            "firing_precision": hold["firing_precision"],
            "lift_rank_spearman": hold["lift_rank_spearman"],
            "empirical_fdr_vs_null": null.get("empirical_fdr"),
        },
        "ethical_considerations": [
            "Association rules can encode sensitive inferences when items are "
            "proxies for protected attributes; this catalogue is giftware, but "
            "the same pipeline on pharmacy or grocery data would need review.",
            "Rules used for dynamic pricing can produce discriminatory outcomes "
            "even when no protected attribute is present in the data.",
        ],
        "caveats": [
            "Support thresholds systematically exclude long-tail products; the "
            "rule set is silent about ~%d%% of the catalogue by design."
            % int(100 * (1 - min(1.0, 200 / max(prof["n_items"], 1)))),
            "The holdout is temporal, so seasonal drift is measured but not "
            "corrected. Rules mined on Jan-Sep are tested on Oct-Dec, which "
            "spans the retailer's Christmas peak.",
            "Wholesale-sized baskets were removed; rules do not describe B2B "
            "purchasing.",
        ],
    }
