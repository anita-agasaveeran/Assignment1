"""Typed configuration + deterministic run provenance.

Every artifact this project writes carries the hash of the config that produced
it, so a reviewer can tell at a glance whether two result sets are comparable.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PathsConfig:
    raw: str = "data/raw"
    interim: str = "data/interim"
    processed: str = "data/processed"
    artifacts: str = "artifacts"
    reports: str = "reports"

    def resolve(self, key: str) -> Path:
        p = ROOT / getattr(self, key)
        p.mkdir(parents=True, exist_ok=True)
        return p


@dataclass(frozen=True)
class DataConfig:
    """CRISP-DM Phase 3 (Data Preparation) knobs.

    Defaults follow the cleaning recipe that is standard for the UCI/Kaggle
    Online Retail dataset: drop credit notes, non-positive quantities and the
    service StockCodes that are not products.
    """
    source_name: str = "online_retail"
    source_file: str = "Online Retail.xlsx"
    source_url: str = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
    kaggle_ref: str = "carrie1/ecommerce-data"

    basket_key: str = "InvoiceNo"
    item_key: str = "StockCode"
    label_key: str = "Description"

    drop_cancellations: bool = True
    drop_non_positive_quantity: bool = True
    drop_non_positive_price: bool = True
    drop_service_codes: bool = True
    countries: list[str] = field(default_factory=list)  # empty = all

    min_basket_size: int = 2
    max_basket_size: int = 200
    min_item_support_count: int = 30

    # Temporal holdout: retail demand drifts, so a random split leaks the
    # future into the training set.  Webb (2007) requires an *independent*
    # holdout for the significance test to be honest.
    split_strategy: str = "temporal"   # temporal | random
    holdout_fraction: float = 0.30
    random_seed: int = 20260901


@dataclass(frozen=True)
class MiningConfig:
    """CRISP-DM Phase 4 (Modeling) knobs -- these are what AutoResearch tunes."""
    algorithm: str = "fpgrowth"        # fpgrowth | apriori
    min_support: float = 0.02
    min_confidence: float = 0.30
    min_lift: float = 1.05
    min_leverage: float = 0.0
    max_itemset_len: int = 4
    max_antecedent_len: int = 3
    top_k: int = 500


@dataclass(frozen=True)
class StatsConfig:
    alpha: float = 0.05
    correction: str = "bh"             # bh | bonferroni | none
    min_holdout_count: int = 5


@dataclass(frozen=True)
class AutoResearchConfig:
    """Stochastic hill climbing with random restarts (Selman & Gomes, 2006)."""
    enabled: bool = True
    max_evaluations: int = 60
    restarts: int = 3
    neighbours_per_step: int = 4
    initial_step: float = 0.5
    step_decay: float = 0.75
    min_step: float = 0.05
    patience: int = 6
    tabu_size: int = 40
    seed: int = 20260901
    # Objective weights -- see armlab.autoresearch.objective for the citation
    # attached to each term.
    w_validated_yield: float = 0.35
    w_effect_size: float = 0.25
    w_coverage: float = 0.20
    w_parsimony: float = 0.10
    w_cost: float = 0.10


@dataclass(frozen=True)
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    mining: MiningConfig = field(default_factory=MiningConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    autoresearch: AutoResearchConfig = field(default_factory=AutoResearchConfig)

    # ---- serialisation -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        """Stable 12-char hash of the whole config -- the run identity."""
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def with_mining(self, **kw: Any) -> "Config":
        return replace(self, mining=replace(self.mining, **kw))

    def with_data(self, **kw: Any) -> "Config":
        return replace(self, data=replace(self.data, **kw))

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        if path is None:
            path = ROOT / "config" / "experiment.yaml"
        path = Path(path)
        if not path.exists():
            return cls()
        raw = yaml.safe_load(path.read_text()) or {}
        return cls(
            paths=PathsConfig(**raw.get("paths", {})),
            data=DataConfig(**raw.get("data", {})),
            mining=MiningConfig(**raw.get("mining", {})),
            stats=StatsConfig(**raw.get("stats", {})),
            autoresearch=AutoResearchConfig(**raw.get("autoresearch", {})),
        )


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return "unknown"


def provenance(cfg: Config) -> dict[str, Any]:
    """Everything needed to reproduce a run, embedded in every artifact."""
    return {
        "run_fingerprint": cfg.fingerprint(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "config": cfg.to_dict(),
    }
