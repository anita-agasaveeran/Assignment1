"""Command line entry point:  python -m ccseg.cli run [--quick]"""
from __future__ import annotations

import argparse

from . import config
from .report import run_study


def main() -> int:
    p = argparse.ArgumentParser(prog="ccseg", description="CRISP-DM credit-card segmentation study")
    p.add_argument("command", choices=["run"], nargs="?", default="run")
    p.add_argument("--quick", action="store_true", help="small budget, for a smoke test")
    p.add_argument("--evals", type=int, default=None, help="override the evaluation budget")
    p.add_argument("--objective", default="composite",
                   choices=["composite", "silhouette", "calinski_harabasz", "davies_bouldin"])
    p.add_argument("--strategy", default="stochastic", choices=["stochastic", "steepest"])
    p.add_argument("--out", default="run.json")
    a = p.parse_args()

    rc = config.RunConfig(objective=a.objective)
    rc.search.strategy = a.strategy
    if a.quick:
        rc.search.max_evals, rc.search.restarts, rc.search.search_subsample = 60, 3, 1200
        rc.evaluation.bootstrap_rounds, rc.evaluation.gap_b_refs = 8, 5
    if a.evals:
        rc.search.max_evals = a.evals
    run_study(rc, out_name=a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
