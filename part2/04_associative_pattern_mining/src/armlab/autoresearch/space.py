"""The AutoResearch search space.

Hill climbing needs a metric space.  Mining hyperparameters are not one: support
is meaningful on a log scale, `max_antecedent_len` is a small integer, and lift
is linear.  Every dimension is therefore mapped to a normalised [0,1] coordinate
where a step of 0.1 means roughly "one comparably sized move" in each, and
decoded back only at evaluation time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Dimension:
    name: str
    low: float
    high: float
    kind: str = "float"          # float | log | int
    rationale: str = ""

    def decode(self, u: float) -> Any:
        u = float(np.clip(u, 0.0, 1.0))
        if self.kind == "log":
            lo, hi = math.log(self.low), math.log(self.high)
            return float(math.exp(lo + u * (hi - lo)))
        if self.kind == "int":
            return int(round(self.low + u * (self.high - self.low)))
        return float(self.low + u * (self.high - self.low))

    def encode(self, v: float) -> float:
        if self.kind == "log":
            lo, hi = math.log(self.low), math.log(self.high)
            return float(np.clip((math.log(max(v, 1e-9)) - lo) / (hi - lo), 0, 1))
        span = self.high - self.low
        return float(np.clip((v - self.low) / span, 0, 1)) if span else 0.0


SPACE: list[Dimension] = [
    Dimension("min_support", 0.004, 0.12, "log",
              "Log scale: the itemset count collapses super-linearly as support "
              "falls, so 0.5%->1% is a far bigger move than 10%->10.5%."),
    Dimension("min_confidence", 0.05, 0.90, "float",
              "Linear. Below ~0.05 rules are noise; above ~0.9 only tautologies "
              "and near-duplicate product variants survive."),
    Dimension("min_lift", 1.0, 4.0, "float",
              "Lift 1.0 is independence -- the floor. Above ~4 on retail data "
              "only rare-item artefacts remain."),
    Dimension("max_antecedent_len", 1, 3, "int",
              "Longer antecedents fire on fewer baskets and multiply the "
              "hypothesis count; 3 is the practical ceiling for merchandising."),
    Dimension("max_itemset_len", 2, 4, "int",
              "Caps the FP-Growth recursion depth; dominates runtime."),
]

NAMES = [d.name for d in SPACE]
INDEX = {d.name: i for i, d in enumerate(SPACE)}


def decode(u: np.ndarray) -> dict[str, Any]:
    params = {d.name: d.decode(u[i]) for i, d in enumerate(SPACE)}
    # Repair: an antecedent cannot be longer than the itemset it comes from.
    params["max_antecedent_len"] = int(min(params["max_antecedent_len"],
                                           params["max_itemset_len"] - 1))
    params["max_antecedent_len"] = max(1, params["max_antecedent_len"])
    return params


def encode(params: dict[str, Any]) -> np.ndarray:
    return np.array([d.encode(params[d.name]) for d in SPACE], dtype=float)


def describe() -> list[dict[str, Any]]:
    return [{"name": d.name, "low": d.low, "high": d.high, "scale": d.kind,
             "rationale": d.rationale} for d in SPACE]


def key(u: np.ndarray, resolution: int = 24) -> tuple:
    """Discretised coordinate used for the tabu set."""
    return tuple(int(round(x * resolution)) for x in np.clip(u, 0, 1))
