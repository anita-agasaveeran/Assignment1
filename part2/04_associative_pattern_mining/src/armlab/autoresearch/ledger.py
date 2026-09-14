"""Append-only experiment ledger.

Every configuration AutoResearch evaluates -- accepted, rejected, or wasted on a
restart -- is written here before the next one starts.  If the process dies at
evaluation 41, the first 40 are still on disk and still analysable.  This is the
provenance record the dashboard's Search Trajectory panel reads.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Trial:
    index: int
    restart: int
    iteration: int
    kind: str                       # seed | neighbour | restart
    params: dict[str, Any]
    coords: list[float]
    score: float
    terms: dict[str, float]
    weighted: dict[str, float]
    diagnostics: dict[str, Any]
    accepted: bool = False
    is_best_so_far: bool = False
    step_size: float = 0.0
    seconds: float = 0.0
    parent: int | None = None
    cache_hit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Ledger:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else None
        self.trials: list[Trial] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("")

    def record(self, trial: Trial) -> Trial:
        self.trials.append(trial)
        if self.path:
            with self.path.open("a") as fh:
                fh.write(json.dumps(trial.to_dict(), default=str) + "\n")
        return trial

    @property
    def best(self) -> Trial | None:
        return max(self.trials, key=lambda t: t.score) if self.trials else None

    def to_dict(self) -> list[dict[str, Any]]:
        return [t.to_dict() for t in self.trials]
