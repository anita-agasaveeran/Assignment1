"""End-to-end CRISP-DM run: executes all six phases and emits one run record.

The output of ``run_study`` is a single JSON document that is the sole input to the
dashboard. Nothing in the dashboard is hand-written from a number seen in a
terminal - if it is on a panel, it came from here, which is what makes the whole
thing reproducible with one command.
"""
from __future__ import annotations

import json
import platform
import sys
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn

from . import (autoresearch as ar, config, data, deploy, evaluate, models,
               prepare, profiling, research)
from . import search_space as ss
from .metrics import silhouette_profile

warnings.filterwarnings("ignore")


def _log(msg: str, t0: float) -> None:
    print(f"[{time.perf_counter()-t0:7.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Success criteria
# --------------------------------------------------------------------------- #
def check_criteria(champion_metrics: dict, ref_space: dict, stability_res: dict,
                   profiles: list[dict], surrogate: dict, latency: dict) -> list[dict]:
    """Evaluate the Phase-1 charter against the Phase-5 results.

    Judged on the *reference-space* silhouette, not the modelling-space one. Grading
    a model in the space that was tuned to make it look good is how a project passes
    its own exam.
    """
    crit = {c["id"]: c for c in config.BUSINESS_CHARTER["success_criteria"]}
    jac = stability_res.get("per_cluster_jaccard", {})
    min_jac = min((v["mean"] for v in jac.values()), default=0.0)
    min_share = min((p["share"] for p in profiles), default=0.0)
    p95 = latency["measurements"][-1]["per_record_ms_p95"]
    sil_ref = ref_space.get("silhouette", float("nan"))

    rows = [
        ("SC-1", sil_ref >= 0.25, f"silhouette = {sil_ref:.3f} in the reference space "
                                  f"({champion_metrics['silhouette']:.3f} in the modelling space)"),
        ("SC-2", min_jac >= 0.60, f"weakest segment Jaccard = {min_jac:.3f} "
                                  f"(over {len(jac)} segments)"),
        ("SC-3", stability_res.get("ari_mean", 0) >= 0.60,
         f"mean bootstrap ARI = {stability_res.get('ari_mean')} "
         f"(95% CI {stability_res.get('ari_ci95')})"),
        ("SC-4", min_share >= 0.03, f"smallest segment holds {100*min_share:.1f}% of the book"),
        ("SC-5", surrogate["fidelity"] >= 0.85,
         f"depth-{surrogate['depth']} surrogate reproduces {100*surrogate['fidelity']:.1f}% of assignments"),
        ("SC-6", p95 < 5.0, f"p95 scoring latency = {p95:.3f} ms/record at batch 1000"),
    ]
    return [{"id": i, "criterion": crit[i]["criterion"], "rationale": crit[i]["rationale"],
             "passed": bool(ok), "evidence": ev} for i, ok, ev in rows]


# --------------------------------------------------------------------------- #
# Main study
# --------------------------------------------------------------------------- #
def run_study(rc: config.RunConfig | None = None, out_name: str = "run.json") -> dict:
    rc = rc or config.RunConfig()
    sc, ec = rc.search, rc.evaluation
    t0 = time.perf_counter()
    started = datetime.now(timezone.utc)

    # ---------------------------------------------------------- Phase 1 + 2 --
    _log("Phase 2: acquiring data", t0)
    raw, provenance = data.load_raw()
    prof = data.profile(raw)
    issues = data.audit(raw)

    _log("Phase 3: fixed cleaning + feature engineering", t0)
    clean, ledger = prepare.fixed_clean(raw)
    transforms = prepare.transform_effect(clean, prepare.feature_view("raw17"))

    # Clustering tendency, on the neutral reference space so it is not a property
    # of a preprocessing choice we later search over.
    ref_pre = prepare.build_preprocessor(
        dict(imputer="median", winsorize="none", transform="yeojohnson",
             scaler="standard", reducer="none"))
    X_ref = np.asarray(ref_pre.fit_transform(clean[prepare.feature_view("raw17")]), dtype=float)
    tendency = data.clustering_tendency(X_ref, seed=sc.seed)
    _log(f"  Hopkins = {tendency['hopkins']} ({tendency['interpretation']})", t0)

    # ------------------------------------------------------------- Phase 4 --
    search_frame = clean.sample(min(sc.search_subsample, len(clean)), random_state=sc.seed)
    _log(f"Phase 4: calibrating objective on n={len(search_frame)}", t0)
    calib, calib_probes = ar.calibrate(search_frame, sc, n=18)

    _log(f"Phase 4: hill climbing ({sc.max_evals} evals, {sc.restarts} restarts)", t0)
    ev_hc = ar.Evaluator(search_frame, calib, sc, objective=rc.objective, seed=sc.seed)
    hc = ar.hill_climb(ev_hc, sc)
    _log(f"  HC best = {hc['best_objective']} in {hc['wall_seconds']}s", t0)

    _log("Phase 4: random-search control arm (identical budget)", t0)
    ev_rs = ar.Evaluator(search_frame, calib, sc, objective=rc.objective, seed=sc.seed)
    rs = ar.random_search(ev_rs, sc, budget=ev_hc.n_calls)
    _log(f"  RS best = {rs['best_objective']} in {rs['wall_seconds']}s", t0)

    all_trials = hc["trials"] + rs["trials"]
    budget = max(len(hc["trials"]), len(rs["trials"]))
    board = ar.leaderboard(all_trials, top=25)

    _log("Phase 4: refitting the top finalists on the full book", t0)
    finals = evaluate.finalist_refit(clean, board, calib, sc, top_n=12, seed=sc.seed)
    champion_cfg = finals["champion"]["config"]
    champion_source = finals["champion"]["optimiser"]
    _log(f"  {finals['n_feasible_on_full']}/{finals['n_refit']} finalists feasible on full data", t0)

    _log("Phase 4: stratified subsample -> full-book transfer study", t0)
    transfer = evaluate.transfer_study(clean, all_trials, calib, sc, n_probe=26, seed=sc.seed)
    _log(f"  transfer rho(full range) = {transfer.get('spearman_full_range')}, "
         f"rho(top band) = {transfer.get('spearman_top_band')}", t0)

    # ------------------------------------------------------------- Phase 5 --
    _log("Phase 5: refitting champion on the full book", t0)
    fit = evaluate.fit_pipeline(clean, champion_cfg, seed=sc.seed)
    from .metrics import internal_indices, score_partition
    X_eval_full, _ = prepare.reference_space(clean, seed=sc.seed)
    champ = score_partition(X_eval_full, fit["labels"], calib, objective=rc.objective,
                            search_cfg=sc, seed=sc.seed)
    champ["self_silhouette"] = internal_indices(fit["X"], fit["labels"], seed=sc.seed)["silhouette"]
    champ["model_space_dims"] = int(fit["X"].shape[1])
    _log(f"  full-data: k={champ['k']} sil={champ['silhouette']} obj={champ['objective']}", t0)

    ref_space = evaluate.reference_space_indices(clean, fit["labels"], seed=sc.seed)
    # Silhouette diagnostic is drawn in the neutral space too, so the per-cluster
    # picture matches the headline number rather than a friendlier projection.
    sil_prof = silhouette_profile(X_eval_full, fit["labels"], seed=sc.seed)

    _log(f"Phase 5: stability ({ec.bootstrap_rounds} resampling replicates)", t0)
    stab = evaluate.stability(clean, champion_cfg, ec, fit["labels"], fit_cap=4000)
    _log(f"  ARI = {stab.get('ari_mean')} +/- {stab.get('ari_std')}", t0)

    _log("Phase 5: gap statistic", t0)
    gap = evaluate.gap_statistic(fit["X"], ec.k_sweep, b_refs=ec.gap_b_refs, seed=sc.seed)

    _log("Phase 5: k-sweep", t0)
    # HDBSCAN chooses its own k, so the sweep runs on the best k-parameterised
    # finalist instead - the panel is about the choice of k, and a model that does
    # not take k cannot answer it.
    # The sweep holds the champion's *preprocessing* fixed and clusters with KMeans,
    # which honours every requested k. Birch and HDBSCAN do not: Birch caps the
    # partition at the size of its CF-tree, so sweeping it produced seven identical
    # rows labelled k=6..12. The champion's own achieved k is reported separately.
    sweep_cfg = dict(champion_cfg, algorithm="kmeans", n_init=10, init="k-means++")
    sweep = evaluate.k_sweep(clean, sweep_cfg, calib, sc, ec, stability_rounds=6)

    _log("Phase 5: baselines + algorithm shootout", t0)
    base = evaluate.baselines(clean, calib, sc, champion_k=int(champ["k"]), seed=sc.seed)
    shootout = evaluate.algorithm_shootout(clean, champion_cfg, calib, sc, seed=sc.seed)

    # What did the search actually buy? Compared against the strongest hand-built
    # pipeline, not against the weakest, and reported either way.
    best_base = max((b for b in base if "objective" in b), key=lambda b: b["objective"], default=None)
    naive = next((b for b in base if b["name"] == "Textbook default"), None)
    search_value = {
        "champion_objective": champ["objective"],
        "best_baseline": best_base["name"] if best_base else None,
        "best_baseline_objective": best_base["objective"] if best_base else None,
        "gain_over_best_baseline": round(champ["objective"] - best_base["objective"], 4) if best_base else None,
        "gain_over_textbook_default": round(champ["objective"] - naive["objective"], 4) if naive else None,
        "verdict": None,
    }
    if best_base:
        g = search_value["gain_over_best_baseline"]
        search_value["verdict"] = (
            "The search clears every hand-built pipeline by a clear margin."
            if g > 0.02 else
            "The search beats the best hand-built pipeline, but by less than the spread "
            "between adjacent leaderboard entries - the honest claim is that it found a "
            "good pipeline automatically, not a better one than an expert would."
            if g > 0 else
            "A hand-built pipeline matched or beat the search champion on the full book. "
            "The search's value here is coverage and reproducibility, not peak score.")

    # ------------------------------------------------------ interpretation --
    _log("Phase 5: segment profiling + surrogate + projection", t0)
    profiles = profiling.name_segments(profiling.segment_profiles(clean, fit["labels"]))
    surrogate = profiling.surrogate_rules(clean, fit["labels"], depth=ec.surrogate_depth, seed=sc.seed)
    proj = profiling.projection(fit["X"], fit["labels"], seed=sc.seed)
    economics = profiling.unit_economics(clean, fit["labels"])
    margins = fit["assigner"].decision_margin(fit["X"])
    agree = np.array(stab.get("per_point_agreement_raw")) if stab.get("per_point_agreement_raw") else None
    boundary = profiling.boundary_accounts(margins, agree, fit["labels"])

    # ------------------------------------------------ charter compliance --
    _log("Phase 5: grading the shortlist against the full charter", t0)
    shortlist = [f["config"] for f in finals["finalists"] if "error" not in f][:6]
    shortlist += [dict(champion_cfg, algorithm="kmeans", k=k, n_init=10, init="k-means++")
                  for k in (3, 4, 5, 6)]
    compliance = evaluate.charter_compliance(clean, shortlist, calib, sc, ec, rounds=12, seed=sc.seed)
    _log(f"  {compliance['n_compliant']}/{compliance['n_evaluated']} candidates satisfy every criterion", t0)

    # Snapshot the champion's evidence before anything is swapped, so the report can
    # show both the objective-optimal model and whatever actually ships.
    champ_jac = {int(k): v["mean"] for k, v in stab.get("per_cluster_jaccard", {}).items()}
    champion_charter = {
        "SC-1": ref_space.get("silhouette", 0) >= 0.25,
        "SC-2": min(champ_jac.values(), default=0.0) >= 0.60,
        "SC-3": stab.get("ari_mean", 0) >= 0.60,
        "SC-4": min((p["share"] for p in profiles), default=0.0) >= sc.min_cluster_frac,
        "SC-5": surrogate["fidelity"] >= 0.85,
    }
    champion_snapshot = {
        "pipeline": ss.describe(champion_cfg), "metrics": champ, "reference_space": ref_space,
        "ari_mean": stab.get("ari_mean"), "per_cluster_jaccard": champ_jac,
        "min_share": round(min((p["share"] for p in profiles), default=0.0), 4),
        "surrogate_fidelity": surrogate["fidelity"], "checks": champion_charter,
        "failed": [k for k, v in champion_charter.items() if not v],
    }
    # SC-6 (latency) is measured in Phase 6 and is met by every candidate by three
    # orders of magnitude, so the switch decision rests on the five available now.
    deployed_cfg, deployed_fit, deployed_role = champion_cfg, fit, "objective-optimal champion"
    if compliance["recommended"] and not all(champion_charter.values()):
        deployed_cfg = compliance["recommended"]["config"]
        deployed_fit = evaluate.fit_pipeline(clean, deployed_cfg, seed=sc.seed)
        deployed_role = ("charter-compliant recommendation" if compliance["n_compliant"]
                         else "most reproducible candidate, pending a charter amendment")
        _log(f"  champion fails the charter; deploying {ss.describe(deployed_cfg)}", t0)

    # Full evaluation of whatever is actually being shipped.
    if deployed_cfg is not champion_cfg:
        dep_metrics = score_partition(X_eval_full, deployed_fit["labels"], calib,
                                      objective=rc.objective, search_cfg=sc, seed=sc.seed)
        dep_metrics["self_silhouette"] = internal_indices(
            deployed_fit["X"], deployed_fit["labels"], seed=sc.seed)["silhouette"]
        dep_metrics["model_space_dims"] = int(deployed_fit["X"].shape[1])
        dep_ref = evaluate.reference_space_indices(clean, deployed_fit["labels"], seed=sc.seed)
        dep_stab = evaluate.stability(clean, deployed_cfg, ec, deployed_fit["labels"], fit_cap=4000)
        dep_sil = silhouette_profile(X_eval_full, deployed_fit["labels"], seed=sc.seed)
        profiles = profiling.name_segments(profiling.segment_profiles(clean, deployed_fit["labels"]))
        surrogate = profiling.surrogate_rules(clean, deployed_fit["labels"],
                                              depth=ec.surrogate_depth, seed=sc.seed)
        proj = profiling.projection(deployed_fit["X"], deployed_fit["labels"], seed=sc.seed)
        economics = profiling.unit_economics(clean, deployed_fit["labels"])
        margins = deployed_fit["assigner"].decision_margin(deployed_fit["X"])
        agree = (np.array(dep_stab["per_point_agreement_raw"])
                 if dep_stab.get("per_point_agreement_raw") else None)
        boundary = profiling.boundary_accounts(margins, agree, deployed_fit["labels"])
        champ, ref_space, stab, sil_prof = dep_metrics, dep_ref, dep_stab, dep_sil

    # ------------------------------------------------------------- Phase 6 --
    _log("Phase 6: scoring artifact, latency, drift", t0)
    personas = {int(p["cluster"]): p["persona"] for p in profiles}
    artifact = deploy.build_artifact(
        deployed_fit, deployed_cfg, personas,
        trained_on={**provenance, "date": started.date().isoformat(), "n_rows": int(len(clean))},
        reference_frame=clean)
    artifact_path = artifact.save()
    latency = deploy.latency_benchmark(artifact, raw)

    rng = np.random.default_rng(sc.seed)
    holdout = raw.iloc[rng.choice(len(raw), len(raw) // 3, replace=False)]
    drift_healthy = deploy.drift_report(artifact, raw, holdout, label="random hold-third (negative control)")
    drift_shift = deploy.drift_report(artifact, raw, deploy.synthetic_shift(raw, seed=sc.seed),
                                      label="synthetic shifted book (positive control)")

    criteria = check_criteria(champ, ref_space, stab, profiles, surrogate, latency)
    card = deploy.model_card(
        champion={"pipeline": ss.describe(deployed_cfg), "metrics": champ},
        stability_res=stab, ref_space=ref_space, latency=latency, personas=profiles,
        provenance={**provenance, "date": started.date().isoformat()},
        search_summary={"space_size": ss.space_size(), "unique_evaluations": ev_hc.n_calls + ev_rs.n_calls},
        criteria_results=criteria)

    # ------------------------------------------------------------- outputs --
    scored = artifact.score(raw)
    scored["assignment_agreement"] = agree if agree is not None else np.nan
    scored_path = config.ARTIFACTS / "scored_customers.csv"
    scored.to_csv(scored_path, index=False)

    run = {
        "meta": {
            "generated_at": started.isoformat(),
            "wall_seconds": round(time.perf_counter() - t0, 1),
            "seed": config.RANDOM_SEED,
            "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                         "numpy": np.__version__, "pandas": pd.__version__,
                         "platform": platform.platform()},
            "run_config": {"search": sc.to_dict(),
                           "evaluation": {"bootstrap_rounds": ec.bootstrap_rounds,
                                          "bootstrap_frac": ec.bootstrap_frac,
                                          "gap_b_refs": ec.gap_b_refs,
                                          "k_sweep": list(ec.k_sweep),
                                          "surrogate_depth": ec.surrogate_depth},
                           "objective": rc.objective},
            "artifacts": {"model": str(artifact_path.name), "scored": str(scored_path.name)},
        },
        "phase1_business": config.BUSINESS_CHARTER,
        "phase2_data": {"provenance": provenance, "profile": prof, "quality_issues": issues,
                        "clustering_tendency": tendency},
        "phase3_prepare": {"ledger": ledger, "transform_effect": transforms,
                           "feature_views": {v: len(prepare.feature_view(v))
                                             for v in ["raw17", "engineered", "core8", "behaviour"]}},
        "phase4_model": {
            "space": ss.space_size(),
            "objective": {"name": rc.objective, "weights": {k: v for k, v in
                          __import__("ccseg.metrics", fromlist=["x"]).COMPOSITE_WEIGHTS.items()},
                          "calibration": calib.to_dict(),
                          "infeasible_shift": 1.0},
            "calibration_probes": calib_probes,
            "hill_climb": {k: v for k, v in hc.items() if k != "trials"},
            "random_search": {k: v for k, v in rs.items() if k != "trials"},
            "trials": all_trials,
            "anytime": {"hill_climb": ar.anytime_curve(hc["trials"], budget),
                        "random_search": ar.anytime_curve(rs["trials"], budget)},
            "leaderboard": board[:15],
            "finalist_refit": finals,
            "transfer_study": transfer,
            "dimension_importance": ar.dimension_importance(all_trials),
            "optimism_analysis": ar.optimism_analysis(all_trials),
            "evaluation_space": {
                "spec": prepare.REFERENCE_SPEC, "view": prepare.REFERENCE_VIEW,
                "why": ("Every candidate is scored on the geometry its labels induce in one "
                        "fixed space that no candidate can influence. Scoring a candidate in "
                        "the space it chose for itself turns the search into a search for a "
                        "flattering projection - see the optimism panel."),
            },
            "evaluator_hill_climb": ev_hc.stats(),
            "evaluator_random_search": ev_rs.stats(),
            "champion": {"source": champion_source, "config": ss.canonical(champion_cfg),
                         "pipeline": ss.describe(champion_cfg),
                         "family": models.ALGO_FAMILY.get(champion_cfg["algorithm"]),
                         "search_objective": hc["best_objective"] if champion_source == "hill_climb"
                         else rs["best_objective"],
                         # The champion's *own* metrics. What actually ships may differ - see
                         # phase5_evaluate.deployed - and conflating the two is how a report ends
                         # up describing one model with another model's numbers.
                         "full_data_metrics": champion_snapshot["metrics"],
                         "inductive_fidelity": round(fit["assigner"].fidelity_, 4)},
        },
        "phase5_evaluate": {
            "reference_space": ref_space,
            "silhouette_profile": sil_prof,
            "stability": {k: v for k, v in stab.items() if k != "per_point_agreement_raw"},
            "gap_statistic": gap,
            "k_sweep": sweep,
            "k_sweep_pipeline": ss.describe(sweep_cfg),
            "k_sweep_note": ("Champion preprocessing held fixed; clustered with KMeans because "
                             "it honours every requested k. The champion algorithm "
                             f"({champion_cfg['algorithm']}) achieved k={champ['k']}."),
            "baselines": base,
            "algorithm_shootout": shootout,
            "success_criteria": criteria,
            "champion_snapshot": champion_snapshot,
            "charter_compliance": compliance,
            "deployed": {"role": deployed_role, "pipeline": ss.describe(deployed_cfg),
                         "config": ss.canonical(deployed_cfg),
                         "differs_from_champion": ss.config_key(deployed_cfg) != ss.config_key(champion_cfg),
                         "inductive_fidelity": round(deployed_fit["assigner"].fidelity_, 4),
                         "metrics": champ},
            "search_value": search_value,
        },
        "segments": {"profiles": profiles, "surrogate": surrogate, "projection": proj,
                     "economics": economics, "economics_assumptions": profiling.VALUE_ASSUMPTIONS,
                     "boundary_accounts": boundary},
        "phase6_deploy": {"model_card": card, "latency": latency,
                          "drift": [drift_healthy, drift_shift],
                          "reference_stats": {k: v for k, v in artifact.reference_stats.items()
                                              if k != "feature_quantiles"}},
        "research": {"citations": research.CITATIONS, "topics": research.TOPICS},
    }

    out = config.ARTIFACTS / out_name
    out.write_text(json.dumps(run, indent=1, default=_jsonable))
    _log(f"wrote {out} ({out.stat().st_size/1e6:.2f} MB)", t0)
    return run


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


if __name__ == "__main__":
    sys.exit(0 if run_study() else 1)
