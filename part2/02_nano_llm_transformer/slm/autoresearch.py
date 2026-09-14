"""AutoResearch: greedy hill climbing over a registry of published techniques.

The premise: every candidate move is a claim from a paper, and a trial is an
attempt to reproduce that claim at our scale, on our data. The study starts from
a deliberately dated baseline (2019-era GPT-2: LayerNorm, learned absolute
positions, GELU FFN, full multi-head attention, untied embeddings, AdamW+cosine)
and climbs. A successful climb *rediscovers* the modern recipe from the
literature instead of assuming it.

Methodology, and why each piece is there
----------------------------------------
* **Noise floor first.** Before any candidate is judged, the baseline is trained
  with `n_seeds` different seeds and the standard deviation of val loss is
  recorded. Without it, "intervention X improved loss by 0.004" is unfalsifiable.
  Acceptance requires an improvement of at least `z_accept` * sigma.
  (Methodology in the spirit of Wortsman et al. 2023, arXiv:2309.14322.)
* **Paired trials.** Every candidate inside a round uses the same seed and the
  same fixed eval batches as the incumbent, so the comparison is paired and the
  variance being tested against is seed variance, not batch variance.
* **Low-fidelity proxy.** Trials run on a smaller/shorter proxy configuration,
  which is the successive-halving idea from Hyperband (arXiv:1603.06560) reduced
  to its simplest usable form. The champion is then retrained at full scale.
* **Null results are kept.** Rejected trials stay in the database with their
  measured delta. The interesting output of a study is as much "these three
  papers did not reproduce at 4M parameters" as "these six did".

Greedy hill climbing is deliberately simple and has a known failure mode: it
cannot see interactions between techniques that only pay off together. The
report states the accepted order, which is the axis along which that bias acts.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import time
from dataclasses import asdict

import yaml

from .config import RunConfig, apply_overrides, save_config
from .registry import BASELINE_OVERRIDES, PAPERS, SEARCH_SPACE, SEARCH_SPACE_BY_KEY, paper_url
from .tracking import Tracker
from .train import train


def proxy_config(base_corpus: str, steps: int, seq_len: int, batch_size: int,
                 n_layer: int, d_model: int, n_head: int) -> RunConfig:
    """Small, fast configuration used for every trial."""
    cfg = RunConfig(name="proxy", phase="autoresearch")
    cfg.data.corpus = base_corpus
    cfg.model.n_layer, cfg.model.d_model, cfg.model.n_head = n_layer, d_model, n_head
    cfg.model.max_seq_len = seq_len
    cfg.train.seq_len = seq_len
    cfg.train.batch_size = batch_size
    cfg.train.steps = steps
    cfg.train.warmup_steps = max(20, steps // 20)
    cfg.train.eval_every = max(50, steps // 3)
    cfg.train.eval_iters = 20
    cfg.train.log_every = 25
    cfg.train.sample_every = 0
    cfg.train.save_checkpoints = False      # trials are throwaway; keep the disk clean
    cfg = apply_overrides(cfg, BASELINE_OVERRIDES)
    return cfg


def _run_trial(cfg: RunConfig, tracker: Tracker, seed: int) -> dict:
    c = copy.deepcopy(cfg)
    c.train.seed = seed
    t0 = time.perf_counter()
    try:
        res = train(c, tracker=tracker, quiet=True)
        res["wall_s"] = time.perf_counter() - t0
        return res
    except Exception as e:  # a diverged/failed trial is data, not a crash
        return dict(run_id=None, status=f"error:{type(e).__name__}",
                    best_val_loss=float("inf"), wall_s=time.perf_counter() - t0,
                    tokens_per_s=0.0, n_params_nonemb=0, peak_mem_mb=0.0, error=str(e))


def run_study(args: argparse.Namespace) -> dict:
    tracker = Tracker(os.path.join(args.out_dir, "tracking.db"))
    study = args.study
    base = proxy_config(args.corpus, args.steps, args.seq_len, args.batch_size,
                        args.n_layer, args.d_model, args.n_head)

    candidates = [iv for iv in SEARCH_SPACE if iv.key not in set(args.exclude or [])]
    study_cfg = dict(
        study=study, proxy=base.to_dict(), n_seeds=args.n_seeds, z_accept=args.z_accept,
        min_delta=args.min_delta, max_rounds=args.max_rounds,
        candidates=[c.key for c in candidates],
        baseline_overrides=BASELINE_OVERRIDES,
        method="greedy hill climbing with paired trials and a seed-variance noise floor",
    )
    tracker.upsert_study(study, study_cfg, "running")
    t_study = time.perf_counter()
    print(f"\n{'='*78}\nAutoResearch study '{study}'")
    print(f"  proxy: L{base.model.n_layer} d{base.model.d_model} "
          f"T{base.train.seq_len} B{base.train.batch_size} x {base.train.steps} steps")
    print(f"  {len(candidates)} candidate interventions, up to {args.max_rounds} rounds")
    print(f"{'='*78}\n")

    # ---------------------------------------------------------- noise floor
    print(f"[round 0] noise floor: baseline x {args.n_seeds} seeds")
    seed_losses: list[float] = []
    for i in range(args.n_seeds):
        seed = args.seed + i
        r = _run_trial(base, tracker, seed)
        seed_losses.append(r["best_val_loss"])
        tracker.add_trial(study=study, round=0, run_id=r.get("run_id"),
                          intervention="__baseline__", papers=[], overrides={},
                          base_config=base.to_dict(), status=r["status"], accepted=0,
                          val_loss=r["best_val_loss"], baseline_loss=None, delta=None,
                          z_score=None, seed=seed, wall_s=r["wall_s"],
                          tokens_per_s=r.get("tokens_per_s"),
                          n_params=r.get("n_params_nonemb"),
                          peak_mem_mb=r.get("peak_mem_mb"), finished_at=time.time(),
                          meta=dict(role="noise_floor"))
        print(f"    seed {seed}: val {r['best_val_loss']:.4f}  ({r['wall_s']:.0f}s)")

    sigma = statistics.stdev(seed_losses) if len(seed_losses) > 1 else 0.0
    incumbent_loss = statistics.mean(seed_losses)
    threshold = max(args.min_delta, args.z_accept * sigma)
    print(f"  baseline mean {incumbent_loss:.4f}  sigma {sigma:.4f}  "
          f"accept if improvement > {threshold:.4f}\n")

    incumbent = copy.deepcopy(base)
    accepted: list[dict] = []
    remaining = {c.key for c in candidates}
    history: list[dict] = []

    # ------------------------------------------------------------- climbing
    for rnd in range(1, args.max_rounds + 1):
        if not remaining:
            break
        seed = args.seed + 100 * rnd            # paired within a round
        print(f"[round {rnd}] incumbent val {incumbent_loss:.4f} | "
              f"{len(remaining)} candidates | seed {seed}")

        # re-measure the incumbent at this round's seed so the comparison is paired
        inc_r = _run_trial(incumbent, tracker, seed)
        inc_loss_paired = inc_r["best_val_loss"]
        tracker.add_trial(study=study, round=rnd, run_id=inc_r.get("run_id"),
                          intervention="__incumbent__", papers=[], overrides={},
                          base_config=incumbent.to_dict(), status=inc_r["status"],
                          accepted=0, val_loss=inc_loss_paired, baseline_loss=None,
                          delta=None, z_score=None, seed=seed, wall_s=inc_r["wall_s"],
                          tokens_per_s=inc_r.get("tokens_per_s"),
                          n_params=inc_r.get("n_params_nonemb"),
                          peak_mem_mb=inc_r.get("peak_mem_mb"), finished_at=time.time(),
                          meta=dict(role="paired_reference"))
        print(f"    incumbent@seed {inc_loss_paired:.4f}")

        results = []
        for key in sorted(remaining):
            iv = SEARCH_SPACE_BY_KEY[key]
            if iv.depends_on and not all(d in {a['key'] for a in accepted} for d in iv.depends_on):
                continue                       # e.g. rope_theta only matters once rope is in
            trial_cfg = apply_overrides(incumbent, iv.on)
            tid = tracker.add_trial(
                study=study, round=rnd, run_id=None, intervention=key,
                papers=iv.papers, overrides=iv.on, base_config=incumbent.to_dict(),
                status="running", accepted=0, seed=seed,
                meta=dict(title=iv.title, category=iv.category, claim=iv.claim,
                          expect=iv.expect, cost=iv.cost, risk=iv.risk,
                          paper_records=iv.paper_records()))
            r = _run_trial(trial_cfg, tracker, seed)
            delta = r["best_val_loss"] - inc_loss_paired
            z = delta / sigma if sigma > 0 else 0.0
            tracker.update_trial(tid, run_id=r.get("run_id"), status=r["status"],
                                 val_loss=r["best_val_loss"], baseline_loss=inc_loss_paired,
                                 delta=delta, z_score=z, wall_s=r["wall_s"],
                                 tokens_per_s=r.get("tokens_per_s"),
                                 n_params=r.get("n_params_nonemb"),
                                 peak_mem_mb=r.get("peak_mem_mb"), finished_at=time.time())
            results.append(dict(key=key, trial_id=tid, delta=delta, z=z,
                                val_loss=r["best_val_loss"], status=r["status"],
                                tokens_per_s=r.get("tokens_per_s", 0.0),
                                n_params=r.get("n_params_nonemb", 0), wall_s=r["wall_s"]))
            flag = "**" if delta < -threshold else ("  " if delta < 0 else "xx")
            print(f"    {flag} {key:16s} val {r['best_val_loss']:.4f} "
                  f"delta {delta:+.4f} (z={z:+.2f}) {r.get('tokens_per_s',0):,.0f} tok/s "
                  f"[{r['wall_s']:.0f}s]")

        results = [r for r in results if r["val_loss"] < float("inf")]
        if not results:
            print("    no valid trials; stopping\n"); break
        best = min(results, key=lambda r: r["delta"])
        history.append(dict(round=rnd, incumbent_loss=inc_loss_paired,
                            results=results, best=best["key"]))

        if best["delta"] < -threshold:
            iv = SEARCH_SPACE_BY_KEY[best["key"]]
            incumbent = apply_overrides(incumbent, iv.on)
            incumbent_loss = best["val_loss"]
            remaining.discard(best["key"])
            accepted.append(dict(key=best["key"], round=rnd, delta=best["delta"],
                                 z=best["z"], val_loss=best["val_loss"],
                                 title=iv.title, papers=iv.papers, claim=iv.claim,
                                 expect=iv.expect, category=iv.category))
            tracker.update_trial(best["trial_id"], accepted=1)
            print(f"  -> ACCEPT {best['key']} ({best['delta']:+.4f}); "
                  f"incumbent now {incumbent_loss:.4f}\n")
        else:
            print(f"  -> no candidate beat the {threshold:.4f} threshold; "
                  f"hill climb converged\n")
            break

    # -------------------------------------------------- champion validation
    print("[final] validating champion vs baseline across seeds")
    val_seeds = [args.seed + 900 + i for i in range(args.n_seeds)]
    champ_losses, base_losses = [], []
    for s in val_seeds:
        champ_losses.append(_run_trial(incumbent, tracker, s)["best_val_loss"])
        base_losses.append(_run_trial(base, tracker, s)["best_val_loss"])
    champ_mean = statistics.mean(champ_losses)
    base_mean = statistics.mean(base_losses)
    pooled = statistics.stdev(champ_losses + base_losses) if len(champ_losses) > 1 else sigma
    effect = (base_mean - champ_mean) / pooled if pooled > 0 else 0.0

    ckpt_dir = os.path.join(args.out_dir, "autoresearch", study)
    os.makedirs(ckpt_dir, exist_ok=True)
    save_config(incumbent, os.path.join(ckpt_dir, "champion.yaml"))

    tested = {r["key"] for h in history for r in h["results"]}
    accepted_keys = {a["key"] for a in accepted}
    # One verdict per rejected technique: its result in the *last* round it was
    # tested, i.e. against the strongest incumbent it faced.
    last_seen: dict[str, dict] = {}
    for h in history:
        for r in h["results"]:
            if r["key"] not in accepted_keys:
                iv = SEARCH_SPACE_BY_KEY[r["key"]]
                last_seen[r["key"]] = dict(
                    key=r["key"], round=h["round"], delta=r["delta"], z=r["z"],
                    title=iv.title, papers=iv.papers, claim=iv.claim, expect=iv.expect,
                    category=iv.category, rounds_tested=last_seen.get(r["key"], {}).get("rounds_tested", 0) + 1,
                    cleared_threshold=bool(r["delta"] < -threshold))
    rejected = sorted(last_seen.values(), key=lambda r: r["delta"])
    converged = not remaining or len(history) < args.max_rounds

    summary = dict(
        study=study, method=study_cfg["method"],
        noise_sigma=sigma, accept_threshold=threshold,
        baseline_mean=base_mean, champion_mean=champ_mean,
        improvement=base_mean - champ_mean,
        improvement_pct=round(100 * (base_mean - champ_mean) / base_mean, 2),
        effect_size_cohens_d=round(effect, 2),
        baseline_seed_losses=base_losses, champion_seed_losses=champ_losses,
        accepted=accepted, rejected=rejected, n_tested=len(tested),
        rounds_run=len(history), converged=converged,
        still_significant=[r["key"] for r in rejected if r["cleared_threshold"]],
        champion_config=incumbent.to_dict(),
        champion_yaml=os.path.join(ckpt_dir, "champion.yaml"),
        wall_minutes=round((time.perf_counter() - t_study) / 60, 1),
    )
    tracker.upsert_study(study, study_cfg, "finished", summary)
    with open(os.path.join(ckpt_dir, "study.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*78}")
    print(f"champion: baseline {base_mean:.4f} -> {champ_mean:.4f} "
          f"({summary['improvement_pct']:+.2f}%, Cohen's d {effect:.2f})")
    print(f"accepted ({len(accepted)}): {[a['key'] for a in accepted]}")
    print(f"rejected ({len(rejected)}): {[r['key'] for r in rejected]}")
    if summary["still_significant"]:
        print(f"budget-limited: {summary['still_significant']} still cleared the bar in the "
              f"final round; more rounds would likely have accepted them")
    print(f"total {summary['wall_minutes']} min -> {ckpt_dir}/champion.yaml")
    print(f"{'='*78}\n")
    tracker.close()
    return summary


def main() -> None:
    ap = argparse.ArgumentParser("slm.autoresearch")
    ap.add_argument("--study", default="hillclimb-v1")
    ap.add_argument("--corpus", default="data/processed/stories")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--z-accept", type=float, default=1.0,
                    help="accept only if improvement exceeds this many baseline sigmas")
    ap.add_argument("--min-delta", type=float, default=0.005)
    ap.add_argument("--max-rounds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--exclude", nargs="*", default=[])
    a = ap.parse_args()
    run_study(a)


if __name__ == "__main__":
    main()
