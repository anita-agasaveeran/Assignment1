"""Shared paths, seeding and plot styling for the CRISP-DM skill demonstration.

Implements the `reproducible-ml` skill's Pillar 1 (seed everything) and the
project layout it prescribes (immutable data/raw, derived data/processed,
code in scripts/, artifacts in outputs/).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"
REP = OUT / "reports"
MOD = OUT / "models"
for _d in (PROC, FIG, TAB, REP, MOD):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
TITANIC = RAW / "titanic.csv"
RETAIL = RAW / "online_retail.csv"
WAREHOUSE = PROC / "retail.db"


def seed_everything(seed: int = SEED) -> None:
    """`reproducible-ml` Pillar 1 — seed every RNG we might touch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def sha256(path: Path) -> str:
    """`reproducible-ml` Pillar 3 — hash the dataset so runs are keyed to data."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> str:
    """`experiment-tracking` — record code state with every run."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
        return f"{sha}{'-dirty' if dirty else ''}"
    except Exception:
        return "unknown"


def style_plots():
    """`visualization-builder` steps 3-4: professional, accessible, low-chrome defaults."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", font_scale=0.95)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.2,
        "font.family": "sans-serif",
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
    })
    # Okabe-Ito: colourblind-safe, per the skill's accessibility requirement.
    return ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"]


PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"]


def banner(skill: str, phase: str, text: str = "") -> None:
    line = "=" * 78
    print(f"\n{line}\n[CRISP-DM {phase}]  skill: {skill}\n{('  ' + text) if text else ''}{line}")


def save_table(df, name: str):
    p = TAB / name
    df.to_csv(p, index=True)
    print(f"  -> wrote {p.relative_to(ROOT)}")
    return p


def write_json(obj, name: str):
    p = TAB / name
    p.write_text(json.dumps(obj, indent=2, default=str))
    print(f"  -> wrote {p.relative_to(ROOT)}")
    return p


def write_report(text: str, name: str):
    p = REP / name
    p.write_text(text)
    print(f"  -> wrote {p.relative_to(ROOT)}")
    return p
