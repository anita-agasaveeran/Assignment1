#!/usr/bin/env python3
"""Inject a run record into the dashboard template.

The dashboard has exactly one data source: artifacts/run.json. Nothing on the page
is written by hand from a number seen in a terminal, so re-running the study and
re-running this script is the whole update path.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "dashboard" / "template.html"
DEFAULT_RUN = ROOT / "artifacts" / "run.json"
OUT = ROOT / "dashboard" / "index.html"

# Fields carried in the run record for downstream analysis but never read by the
# page. Dropping them keeps the published artifact small without touching the
# archived run record.
DROP = [
    ("phase5_evaluate", "stability", "per_point_agreement_raw"),
    ("phase5_evaluate", "stability", "ari_values"),
    ("phase2_data", "profile", "features", "*", "medians"),
]


def finite(node):
    """Replace NaN / Infinity with null.

    ``json.dumps`` happily emits bare ``NaN``, which every JSON parser including
    the browser's rejects. Silhouette is legitimately NaN for a degenerate
    partition, so these do occur in a real run record and the page must not die
    on one.
    """
    if isinstance(node, dict):
        return {k: finite(v) for k, v in node.items()}
    if isinstance(node, list):
        return [finite(v) for v in node]
    if isinstance(node, float) and not math.isfinite(node):
        return None
    return node


def prune(run: dict) -> dict:
    for path in DROP:
        node = run
        try:
            for key in path[:-1]:
                node = node[key]
            node.pop(path[-1], None)
        except (KeyError, TypeError):
            pass
    # Trial records keep only what the trial scatter and the importance analysis read.
    keep = {"i", "t", "optimiser", "restart", "move", "step_kind", "accepted",
            "config", "key", "objective", "incumbent", "best_so_far", "elapsed_s", "error", "metrics"}
    run["phase4_model"]["trials"] = [{k: v for k, v in t.items() if k in keep}
                                     for t in run["phase4_model"]["trials"]]
    return run


def main(argv: list[str]) -> int:
    src = Path(argv[1]) if len(argv) > 1 else DEFAULT_RUN
    if not src.exists():
        print(f"no run record at {src} - run `python -m ccseg.cli run` first", file=sys.stderr)
        return 1
    run = finite(prune(json.loads(src.read_text())))
    html = TEMPLATE.read_text()
    # Split on '<' so the embedded JSON can never terminate the host <script> block.
    payload = json.dumps(run, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    OUT.write_text(html.replace("__RUN_DATA__", payload))
    print(f"{OUT.relative_to(ROOT)}  ({OUT.stat().st_size / 1e6:.2f} MB)  from {src.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
