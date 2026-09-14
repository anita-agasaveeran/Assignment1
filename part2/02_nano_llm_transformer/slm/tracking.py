"""Experiment tracking: a single SQLite file is the source of truth.

Why SQLite and not a hosted tracker: the whole project must run offline on one
laptop, the dashboard needs cheap indexed reads, and a grader/reviewer should be
able to open `runs/tracking.db` and reproduce every number without our code.

Schema
------
runs        one row per training run (config JSON, status, summary metrics)
metrics     long-format (run_id, step, key, value) time series
events      human-readable log lines / lifecycle markers
artifacts   files a run produced (checkpoints, profiles, samples)
samples     generated text snapshots, for qualitative tracking over training
trials      one row per AutoResearch hill-climbing trial
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterable

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, name TEXT, phase TEXT, status TEXT,
  fingerprint TEXT, config JSON, env JSON, tags JSON, notes TEXT,
  created_at REAL, updated_at REAL, finished_at REAL,
  n_params INTEGER, n_params_emb INTEGER, summary JSON, parent_run TEXT
);
CREATE TABLE IF NOT EXISTS metrics (
  run_id TEXT, step INTEGER, wall REAL, key TEXT, value REAL
);
CREATE INDEX IF NOT EXISTS idx_metrics ON metrics(run_id, key, step);
CREATE TABLE IF NOT EXISTS events (
  run_id TEXT, ts REAL, level TEXT, message TEXT, data JSON
);
CREATE INDEX IF NOT EXISTS idx_events ON events(run_id, ts);
CREATE TABLE IF NOT EXISTS artifacts (
  run_id TEXT, kind TEXT, path TEXT, meta JSON, ts REAL
);
CREATE TABLE IF NOT EXISTS samples (
  run_id TEXT, step INTEGER, prompt TEXT, completion TEXT, params JSON, ts REAL
);
CREATE INDEX IF NOT EXISTS idx_samples ON samples(run_id, step);
CREATE TABLE IF NOT EXISTS trials (
  trial_id TEXT PRIMARY KEY, study TEXT, round INTEGER, run_id TEXT,
  intervention TEXT, papers JSON, overrides JSON, base_config JSON,
  status TEXT, accepted INTEGER, val_loss REAL, baseline_loss REAL,
  delta REAL, z_score REAL, seed INTEGER, wall_s REAL,
  tokens_per_s REAL, n_params INTEGER, peak_mem_mb REAL,
  created_at REAL, finished_at REAL, meta JSON
);
CREATE INDEX IF NOT EXISTS idx_trials ON trials(study, round);
CREATE TABLE IF NOT EXISTS studies (
  study TEXT PRIMARY KEY, status TEXT, config JSON, created_at REAL,
  updated_at REAL, summary JSON
);
"""


def _json(x: Any) -> str:
    return json.dumps(x, default=str)


class Tracker:
    """Thin, thread-safe-enough writer around the SQLite run store."""

    def __init__(self, db_path: str = "runs/tracking.db"):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self.run_id: str | None = None
        self._buf: list[tuple] = []

    # ---------------------------------------------------------------- runs
    def start_run(self, name: str, phase: str, config: dict, env: dict,
                  tags: Iterable[str] = (), notes: str = "",
                  fingerprint: str = "", parent_run: str | None = None) -> str:
        rid = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        now = time.time()
        self.conn.execute(
            "INSERT INTO runs (run_id,name,phase,status,fingerprint,config,env,tags,notes,"
            "created_at,updated_at,parent_run) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (rid, name, phase, "running", fingerprint, _json(config), _json(env),
             _json(list(tags)), notes, now, now, parent_run))
        self.conn.commit()
        self.run_id = rid
        return rid

    def set_params(self, n_params: int, n_params_emb: int) -> None:
        self.conn.execute("UPDATE runs SET n_params=?, n_params_emb=?, updated_at=? WHERE run_id=?",
                          (n_params, n_params_emb, time.time(), self.run_id))
        self.conn.commit()

    def finish_run(self, status: str = "finished", summary: dict | None = None) -> None:
        self.flush()
        now = time.time()
        self.conn.execute("UPDATE runs SET status=?, summary=?, finished_at=?, updated_at=? WHERE run_id=?",
                          (status, _json(summary or {}), now, now, self.run_id))
        self.conn.commit()

    # ------------------------------------------------------------- metrics
    def log(self, step: int, metrics: dict[str, float], flush: bool = False) -> None:
        wall = time.time()
        for k, v in metrics.items():
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            self._buf.append((self.run_id, int(step), wall, k, fv))
        # Flush on every call: a commit per log interval costs ~1 ms against a
        # ~650 ms training step, and it is what makes the dashboard live rather
        # than 200 steps behind.
        self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        self.conn.executemany("INSERT INTO metrics VALUES (?,?,?,?,?)", self._buf)
        self.conn.commit()
        self._buf.clear()

    def event(self, message: str, level: str = "info", data: dict | None = None) -> None:
        self.conn.execute("INSERT INTO events VALUES (?,?,?,?,?)",
                          (self.run_id, time.time(), level, message, _json(data or {})))
        self.conn.commit()

    def artifact(self, kind: str, path: str, meta: dict | None = None) -> None:
        self.conn.execute("INSERT INTO artifacts VALUES (?,?,?,?,?)",
                          (self.run_id, kind, path, _json(meta or {}), time.time()))
        self.conn.commit()

    def sample(self, step: int, prompt: str, completion: str, params: dict | None = None) -> None:
        self.conn.execute("INSERT INTO samples VALUES (?,?,?,?,?,?)",
                          (self.run_id, int(step), prompt, completion, _json(params or {}), time.time()))
        self.conn.commit()

    # -------------------------------------------------------------- trials
    def upsert_study(self, study: str, config: dict, status: str = "running",
                     summary: dict | None = None) -> None:
        now = time.time()
        self.conn.execute(
            "INSERT INTO studies (study,status,config,created_at,updated_at,summary) VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(study) DO UPDATE SET status=excluded.status, updated_at=excluded.updated_at, "
            "summary=excluded.summary",
            (study, status, _json(config), now, now, _json(summary or {})))
        self.conn.commit()

    def add_trial(self, **kw: Any) -> str:
        tid = kw.pop("trial_id", None) or uuid.uuid4().hex[:10]
        cols = ["trial_id", "study", "round", "run_id", "intervention", "papers", "overrides",
                "base_config", "status", "accepted", "val_loss", "baseline_loss", "delta",
                "z_score", "seed", "wall_s", "tokens_per_s", "n_params", "peak_mem_mb",
                "created_at", "finished_at", "meta"]
        kw.setdefault("created_at", time.time())
        for j in ("papers", "overrides", "base_config", "meta"):
            if j in kw and not isinstance(kw[j], str):
                kw[j] = _json(kw[j])
        vals = [tid] + [kw.get(c) for c in cols[1:]]
        self.conn.execute(f"INSERT INTO trials ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", vals)
        self.conn.commit()
        return tid

    def update_trial(self, trial_id: str, **kw: Any) -> None:
        for j in ("papers", "overrides", "base_config", "meta"):
            if j in kw and not isinstance(kw[j], str):
                kw[j] = _json(kw[j])
        sets = ",".join(f"{k}=?" for k in kw)
        self.conn.execute(f"UPDATE trials SET {sets} WHERE trial_id=?", [*kw.values(), trial_id])
        self.conn.commit()

    def close(self) -> None:
        self.flush()
        self.conn.close()


@contextmanager
def run_context(tracker: Tracker, **kw: Any):
    rid = tracker.start_run(**kw)
    try:
        yield rid
    except KeyboardInterrupt:
        tracker.event("interrupted by user", "warn")
        tracker.finish_run("cancelled")
        raise
    except Exception as e:  # noqa: BLE001
        tracker.event(f"{type(e).__name__}: {e}", "error")
        tracker.finish_run("failed")
        raise


def read_db(db_path: str = "runs/tracking.db") -> sqlite3.Connection:
    """Read-only-ish connection used by the dashboard API."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn
