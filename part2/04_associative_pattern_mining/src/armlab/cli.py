"""ARMLab command line."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from armlab import pipeline
from armlab.config import Config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="armlab",
        description="Associative pattern mining on the Online Retail dataset, "
                    "run as a CRISP-DM pipeline with AutoResearch hill climbing.")
    ap.add_argument("command", choices=["run", "serve", "properties", "profile"],
                    help="run: full pipeline | serve: dashboard + API | "
                         "properties: print the measure property matrix | "
                         "profile: data-understanding report only")
    ap.add_argument("-c", "--config", default=None, help="path to experiment.yaml")
    ap.add_argument("--no-autoresearch", action="store_true",
                    help="skip hill climbing, mine at the configured thresholds")
    ap.add_argument("--no-null-baseline", action="store_true",
                    help="skip the swap-randomisation false-discovery baseline")
    ap.add_argument("--evaluations", type=int, default=None,
                    help="override autoresearch.max_evaluations")
    ap.add_argument("--restarts", type=int, default=None)
    ap.add_argument("--min-support", type=float, default=None)
    ap.add_argument("--country", default=None,
                    help="restrict to one country, e.g. 'United Kingdom'")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8931)
    args = ap.parse_args(argv)

    cfg = Config.load(args.config)
    if args.evaluations is not None:
        cfg = replace(cfg, autoresearch=replace(cfg.autoresearch,
                                                max_evaluations=args.evaluations))
    if args.restarts is not None:
        cfg = replace(cfg, autoresearch=replace(cfg.autoresearch, restarts=args.restarts))
    if args.min_support is not None:
        cfg = cfg.with_mining(min_support=args.min_support)
    if args.country:
        cfg = cfg.with_data(countries=[args.country])

    if args.command == "run":
        pipeline.run(cfg, autoresearch=not args.no_autoresearch,
                     run_null_baseline=not args.no_null_baseline)
        return 0

    if args.command == "profile":
        from armlab.data import acquire, profile as profiling
        raw = acquire.load_raw(cfg)
        prof = profiling.profile(raw, cfg)
        pipeline.write_artifact(cfg, "data_profile.json", prof)
        print(json.dumps({k: v for k, v in prof.items()
                          if k not in ("columns", "basket_size_hist")}, indent=2)[:4000])
        return 0

    if args.command == "properties":
        from armlab.metrics.properties import property_matrix
        pm = property_matrix()
        heads = [p["key"] for p in pm["properties"]]
        print(f"{'measure':26s}" + "".join(f"{h:>5}" for h in heads))
        for r in pm["measures"]:
            print(f"{r['key']:26s}" + "".join(f"{'Y' if r[h] else '.':>5}" for h in heads))
        print("\n" + pm["method"])
        return 0

    if args.command == "serve":
        import uvicorn
        from armlab.api.server import create_app
        app = create_app(cfg)
        print(f"ARMLab dashboard -> http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
