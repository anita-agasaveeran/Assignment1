"""The AutoResearch objective function.

Association rule mining has no natural loss, so "tune min_support until the
output looks nice" is the norm.  That is not a defensible methodology.  We make
the target explicit and multi-objective, following the framing of Ghosh & Nath
(*Multi-objective rule mining using genetic algorithms*, Information Sciences
163:123-133, 2004), who argue support/confidence alone cannot express what a
useful rule set is.

Every term is normalised to [0,1] with higher = better, so the weighted sum is
itself in [0,1] and the dashboard can render an honest stacked decomposition.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from armlab.config import AutoResearchConfig

TARGET_RULES = 500          # saturation point for the yield term
RUNTIME_BUDGET_S = 12.0     # a config slower than this earns no cost credit


def set_runtime_budget(seconds: float) -> None:
    """Calibrate the cost term against this dataset.

    A fixed budget is meaningless: on a 200-basket toy set every configuration
    is instant and the term is a constant 1.0, contributing nothing to the
    search.  The pipeline calibrates it to a small multiple of the baseline
    configuration's wall clock so `cost` actually discriminates.
    """
    global RUNTIME_BUDGET_S
    RUNTIME_BUDGET_S = float(max(0.25, seconds))


def _saturate(n: float, target: float) -> float:
    """Diminishing returns: the 500th rule is worth far less than the 5th."""
    return float(min(1.0, math.log1p(max(n, 0.0)) / math.log1p(target)))


@dataclass
class Score:
    value: float
    terms: dict[str, float]
    weighted: dict[str, float]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "terms": self.terms,
                "weighted": self.weighted, "diagnostics": self.diagnostics}


TERM_DOC = {
    "validated_yield": {
        "label": "Validated yield",
        "definition": "log-saturated count of rules that are simultaneously "
                      "(a) replicated on the temporal holdout, (b) significant "
                      "under search-space-corrected FDR, and (c) productive.",
        "citation": "Webb, Discovering Significant Patterns, Mach. Learn. 68 (2007); "
                    "Benjamini & Hochberg, JRSS-B 57 (1995)",
        "why": "Counting raw rules rewards threshold-lowering. Counting rules that "
               "survive an untouched holdout and a multiplicity correction does not.",
    },
    "effect_size": {
        "label": "Effect size",
        "definition": "median Kulczynski of validated rules, blended with a "
                      "bounded transform of median lift, 1 - 1/lift.",
        "citation": "Wu, Chen & Han, ACM TKDD 4(3) (2010); Brin et al., SIGMOD 1997",
        "why": "Kulczynski is null-invariant, so it does not inflate when the "
               "catalogue is padded with unrelated products; the lift term keeps "
               "the objective sensitive to genuinely surprising pairs.",
    },
    "coverage": {
        "label": "Holdout coverage",
        "definition": "share of holdout baskets on which at least one rule fires.",
        "citation": "Hahsler, Grun & Hornik, J. Stat. Soft. 14(15) (2005)",
        "why": "A rule set that is precise but never applicable cannot drive a "
               "recommendation surface.",
    },
    "parsimony": {
        "label": "Parsimony",
        "definition": "penalises long antecedents and near-duplicate rules that "
                      "share a consequent.",
        "citation": "Occam / MDL framing, Rissanen 1978; redundancy notion from "
                    "Zaki, ACM TKDD 9(3) (2004) on non-redundant rules",
        "why": "Two hundred variations of one insight is one insight with a "
               "reporting problem.",
    },
    "cost": {
        "label": "Compute economy",
        "definition": "1 - normalised wall-clock of the mine+score cycle.",
        "citation": "operational, not from the literature",
        "why": "The champion configuration has to be re-mineable nightly; a "
               "40x slower config needs to be 40x better to win.",
    },
}


def score(rules: pd.DataFrame, holdout_report: dict[str, Any],
          seconds: float, cfg: AutoResearchConfig,
          diagnostics: dict[str, Any] | None = None) -> Score:
    diag: dict[str, Any] = dict(diagnostics or {})
    n = 0 if rules is None else len(rules)

    if n == 0:
        terms = {k: 0.0 for k in ("validated_yield", "effect_size", "coverage",
                                  "parsimony", "cost")}
        terms["cost"] = float(max(0.0, 1.0 - seconds / RUNTIME_BUDGET_S))
        weighted = {k: _w(cfg, k) * v for k, v in terms.items()}
        return Score(sum(weighted.values()), terms, weighted,
                     diag | {"n_rules": 0, "n_validated": 0,
                             "reason": "no rules cleared the thresholds"})

    per = holdout_report.get("per_rule") or {}
    validated = np.asarray(per.get("validated", np.zeros(n, dtype=bool)))
    if validated.size != n:
        validated = np.zeros(n, dtype=bool)
    productive = rules["productive"].to_numpy() if "productive" in rules else np.ones(n, bool)
    sig_ss = (rules["significant_searchspace"].to_numpy()
              if "significant_searchspace" in rules else np.zeros(n, bool))
    keep = validated & productive & sig_ss
    n_val = int(keep.sum())

    # --- 1. validated yield -------------------------------------------------
    t_yield = _saturate(n_val, TARGET_RULES)

    # --- 2. effect size -----------------------------------------------------
    if n_val:
        kul = float(np.median(rules.loc[keep, "kulczynski"]))
        lift = float(np.median(np.clip(rules.loc[keep, "lift"], 1.0, 1e6)))
        t_effect = float(np.clip(0.5 * kul + 0.5 * (1.0 - 1.0 / max(lift, 1e-9)), 0, 1))
    else:
        kul = lift = 0.0
        t_effect = 0.0

    # --- 3. coverage --------------------------------------------------------
    t_cov = float(np.clip(holdout_report.get("coverage", 0.0), 0, 1))

    # --- 4. parsimony -------------------------------------------------------
    if n_val:
        mean_ante = float(rules.loc[keep, "antecedent_len"].mean())
        cons = rules.loc[keep, "consequent"]
        redundancy = 1.0 - (cons.astype(str).nunique() / max(n_val, 1))
    else:
        mean_ante, redundancy = 3.0, 1.0
    len_pen = (mean_ante - 1.0) / 2.0                      # 1 -> 0, 3 -> 1
    t_pars = float(np.clip(1.0 - 0.5 * len_pen - 0.5 * redundancy, 0, 1))

    # --- 5. compute economy -------------------------------------------------
    t_cost = float(np.clip(1.0 - seconds / RUNTIME_BUDGET_S, 0, 1))

    terms = {"validated_yield": t_yield, "effect_size": t_effect,
             "coverage": t_cov, "parsimony": t_pars, "cost": t_cost}
    weighted = {k: _w(cfg, k) * v for k, v in terms.items()}
    diag |= {
        "n_rules": int(n), "n_validated": n_val,
        "median_kulczynski": round(kul, 5), "median_lift": round(lift, 5),
        "mean_antecedent_len": round(mean_ante, 3),
        "consequent_redundancy": round(float(redundancy), 4),
        "seconds": round(seconds, 4),
        "replication_rate": round(float(holdout_report.get("replication_rate", 0.0)), 4),
        "lift_rank_spearman": round(float(holdout_report.get("lift_rank_spearman", 0.0)), 4),
    }
    return Score(float(sum(weighted.values())), terms, weighted, diag)


def _w(cfg: AutoResearchConfig, term: str) -> float:
    return float(getattr(cfg, f"w_{term}"))


def weights(cfg: AutoResearchConfig) -> dict[str, float]:
    return {k: _w(cfg, k) for k in TERM_DOC}
