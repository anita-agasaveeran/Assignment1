"""CRISP-DM phase 6: Deployment. Dashboard API + chat serving.

One FastAPI process serves both the analyst-facing dashboard (read-only views
over runs/tracking.db and the data profiles) and the chat endpoint backed by the
same `LM` object the offline evaluation uses.

The model is loaded lazily and swapped under a lock so the dashboard stays
responsive while training runs in another process. All DB reads use a read-only
connection, so opening the dashboard can never corrupt or lock a live run's
writes (SQLite is in WAL mode).
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slm.infer import LM, latest_checkpoint            # noqa: E402
from slm.registry import SEARCH_SPACE, papers_for_dashboard  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.environ.get("SLM_RUNS", os.path.join(ROOT, "runs"))
DB_PATH = os.path.join(RUNS_DIR, "tracking.db")
CORPUS = os.environ.get("SLM_CORPUS", os.path.join(ROOT, "data/processed/stories"))
CHAT_DATA = os.environ.get("SLM_CHAT", os.path.join(ROOT, "data/processed/chat"))

app = FastAPI(title="CRISP-LM Dashboard", version="0.1.0")

_lm: LM | None = None
_lm_path: str | None = None
_lm_lock = threading.Lock()


def db() -> sqlite3.Connection:
    if not os.path.exists(DB_PATH):
        raise HTTPException(404, "no tracking database yet - train a run first")
    c = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def rows(cur) -> list[dict]:
    return [dict(r) for r in cur.fetchall()]


def jload(s: Any, default=None):
    if not s:
        return default
    try:
        return json.loads(s)
    except (TypeError, json.JSONDecodeError):
        return default


def get_lm(path: str | None = None) -> LM:
    global _lm, _lm_path
    with _lm_lock:
        target = path or _lm_path or latest_checkpoint(RUNS_DIR, "sft") or latest_checkpoint(RUNS_DIR)
        if not target:
            raise HTTPException(503, "no checkpoint available yet")
        if _lm is None or target != _lm_path:
            _lm = LM(target)
            _lm_path = target
        return _lm


# ------------------------------------------------------------------ health
@app.get("/api/health")
def health() -> dict:
    ck = latest_checkpoint(RUNS_DIR, "sft") or latest_checkpoint(RUNS_DIR)
    return dict(ok=True, db=os.path.exists(DB_PATH), checkpoint=ck,
                loaded=_lm_path, corpus=os.path.exists(f"{CORPUS}.meta.json"))


# -------------------------------------------------------------- overview
@app.get("/api/overview")
def overview() -> dict:
    out: dict = dict(generated_at=time.time())
    if os.path.exists(DB_PATH):
        c = db()
        runs = rows(c.execute(
            "SELECT run_id,name,phase,status,created_at,finished_at,n_params,summary,tags "
            "FROM runs ORDER BY created_at DESC"))
        for r in runs:
            r["summary"] = jload(r["summary"], {})
            r["tags"] = jload(r["tags"], [])
        out["n_runs"] = len(runs)
        out["runs_by_phase"] = {}
        for r in runs:
            out["runs_by_phase"][r["phase"]] = out["runs_by_phase"].get(r["phase"], 0) + 1
        out["recent_runs"] = runs[:12]
        best = [r for r in runs if r["phase"] == "pretrain"
                and isinstance(r["summary"].get("best_val_loss"), (int, float))
                and r["summary"]["best_val_loss"] < 1e9]
        out["best_pretrain"] = min(best, key=lambda r: r["summary"]["best_val_loss"]) if best else None
        sft = [r for r in runs if r["phase"] == "sft" and r["summary"].get("best_val_loss")]
        out["best_sft"] = min(sft, key=lambda r: r["summary"]["best_val_loss"]) if sft else None
        out["total_gpu_seconds"] = sum(
            (r["summary"].get("elapsed_s") or 0) for r in runs)
        out["total_tokens_trained"] = sum(
            (r["summary"].get("tokens_seen") or 0) for r in runs)
        st = rows(c.execute("SELECT study,status,summary,created_at FROM studies"))
        for s in st:
            s["summary"] = jload(s["summary"], {})
        out["studies"] = st
        c.close()
    meta = f"{CORPUS}.meta.json"
    if os.path.exists(meta):
        out["corpus"] = json.load(open(meta))
    out["n_papers"] = len(papers_for_dashboard())
    out["n_interventions"] = len(SEARCH_SPACE)
    return out


# ------------------------------------------------------------------ runs
@app.get("/api/runs")
def list_runs(phase: str | None = None, limit: int = 200) -> list[dict]:
    c = db()
    q = "SELECT * FROM runs"
    args: list = []
    if phase:
        q += " WHERE phase=?"; args.append(phase)
    q += " ORDER BY created_at DESC LIMIT ?"; args.append(limit)
    out = rows(c.execute(q, args))
    for r in out:
        for k in ("config", "env", "summary", "tags"):
            r[k] = jload(r[k], {} if k != "tags" else [])
    c.close()
    return out


@app.get("/api/runs/{run_id}")
def run_detail(run_id: str) -> dict:
    c = db()
    r = c.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
    if not r:
        raise HTTPException(404, "run not found")
    d = dict(r)
    for k in ("config", "env", "summary", "tags"):
        d[k] = jload(d[k], {} if k != "tags" else [])
    keys = [x["key"] for x in rows(c.execute(
        "SELECT DISTINCT key FROM metrics WHERE run_id=?", (run_id,)))]
    series: dict[str, list] = {}
    for k in keys:
        series[k] = [[m["step"], m["value"]] for m in rows(c.execute(
            "SELECT step,value FROM metrics WHERE run_id=? AND key=? ORDER BY step", (run_id, k)))]
    d["metrics"] = series
    d["events"] = rows(c.execute(
        "SELECT ts,level,message,data FROM events WHERE run_id=? ORDER BY ts", (run_id,)))
    for e in d["events"]:
        e["data"] = jload(e["data"], {})
    d["samples"] = rows(c.execute(
        "SELECT step,prompt,completion,params FROM samples WHERE run_id=? ORDER BY step", (run_id,)))
    for s in d["samples"]:
        s["params"] = jload(s["params"], {})
    c.close()
    ev = os.path.join(RUNS_DIR, run_id, "eval.json")
    if os.path.exists(ev):
        d["eval"] = json.load(open(ev))
    pl = os.path.join(RUNS_DIR, run_id, "pos_loss.json")
    if os.path.exists(pl):
        d["pos_loss"] = json.load(open(pl))
    return d


@app.get("/api/compare")
def compare(run_ids: str = Query(..., description="comma separated run ids"),
            key: str = "val_loss") -> dict:
    c = db()
    out = {}
    for rid in [r.strip() for r in run_ids.split(",") if r.strip()]:
        out[rid] = [[m["step"], m["value"]] for m in rows(c.execute(
            "SELECT step,value FROM metrics WHERE run_id=? AND key=? ORDER BY step", (rid, key)))]
    c.close()
    return dict(key=key, series=out)


# ------------------------------------------------------------------- data
@app.get("/api/data-profile")
def data_profile() -> dict:
    p = f"{CORPUS}.profile.json"
    if not os.path.exists(p):
        raise HTTPException(404, "no data profile - run slm.data prepare-pretrain")
    return json.load(open(p))


@app.get("/api/sft-profile")
def sft_profile() -> dict:
    p = f"{CHAT_DATA}.profile.json"
    if not os.path.exists(p):
        raise HTTPException(404, "no SFT profile - run slm.chat build-sft")
    return json.load(open(p))


# ---------------------------------------------------------- autoresearch
@app.get("/api/studies")
def studies() -> list[dict]:
    c = db()
    out = rows(c.execute("SELECT * FROM studies ORDER BY created_at DESC"))
    for s in out:
        s["config"] = jload(s["config"], {})
        s["summary"] = jload(s["summary"], {})
    c.close()
    return out


@app.get("/api/studies/{study}")
def study_detail(study: str) -> dict:
    c = db()
    s = c.execute("SELECT * FROM studies WHERE study=?", (study,)).fetchone()
    if not s:
        raise HTTPException(404, "study not found")
    d = dict(s)
    d["config"] = jload(d["config"], {})
    d["summary"] = jload(d["summary"], {})
    tr = rows(c.execute("SELECT * FROM trials WHERE study=? ORDER BY round, created_at", (study,)))
    for t in tr:
        for k in ("papers", "overrides", "base_config", "meta"):
            t[k] = jload(t[k], {})
    d["trials"] = tr
    c.close()
    return d


@app.get("/api/papers")
def papers() -> dict:
    return dict(papers=papers_for_dashboard(),
                interventions=[dict(
                    key=i.key, title=i.title, category=i.category, on=i.on, off=i.off,
                    papers=i.papers, claim=i.claim, expect=i.expect, cost=i.cost,
                    risk=i.risk, depends_on=i.depends_on,
                    paper_records=i.paper_records()) for i in SEARCH_SPACE])


# ------------------------------------------------------------------ model
@app.get("/api/model-card")
def model_card() -> dict:
    ck = latest_checkpoint(RUNS_DIR, "sft") or latest_checkpoint(RUNS_DIR)
    if not ck:
        raise HTTPException(404, "no checkpoint")
    lm = get_lm(ck)
    m = lm.model
    card = dict(
        checkpoint=ck, run_id=lm.run_id, step=lm.step, val_loss=lm.val_loss,
        config=lm.cfg.to_dict(),
        params_total=sum(p.numel() for p in m.parameters()),
        params_non_embedding=m.num_params(True),
        param_breakdown=m.param_breakdown(),
        flops_per_token=m.flops_per_token(),
        kv_cache_mb_full_ctx=m.kv_cache_bytes(1, lm.cfg.model.max_seq_len) / 1e6,
        vocab_size=lm.tok.n_vocab,
        tokenizer_stats={k: v for k, v in lm.tok.train_stats.items() if k != "merge_log"},
        device=str(lm.device),
    )
    sd = os.path.join(os.path.dirname(ck), "summary.json")
    if os.path.exists(sd):
        card["training_summary"] = json.load(open(sd))
    ev = os.path.join(os.path.dirname(ck), "eval.json")
    if os.path.exists(ev):
        card["evaluation"] = json.load(open(ev))
    return card


@app.get("/api/checkpoints")
def checkpoints() -> list[dict]:
    out = []
    if os.path.isdir(RUNS_DIR):
        for d in sorted(os.listdir(RUNS_DIR), reverse=True):
            p = os.path.join(RUNS_DIR, d, "best.pt")
            if os.path.exists(p):
                s = os.path.join(RUNS_DIR, d, "summary.json")
                meta = json.load(open(s)) if os.path.exists(s) else {}
                cfgp = os.path.join(RUNS_DIR, d, "config.yaml")
                phase = "unknown"
                if os.path.exists(cfgp):
                    import yaml
                    phase = (yaml.safe_load(open(cfgp)) or {}).get("phase", "unknown")
                out.append(dict(run_id=d, path=p, phase=phase,
                                size_mb=round(os.path.getsize(p) / 1e6, 1),
                                mtime=os.path.getmtime(p),
                                best_val_loss=meta.get("best_val_loss")))
    return out


# ------------------------------------------------------------------- chat
class ChatReq(BaseModel):
    messages: list[dict]
    system: str | None = None
    max_new_tokens: int = 220
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 0
    repetition_penalty: float = 1.05
    checkpoint: str | None = None


@app.post("/api/chat")
def chat(req: ChatReq):
    lm = get_lm(req.checkpoint)
    from slm.chat import DEFAULT_SYSTEM
    system = req.system if req.system is not None else DEFAULT_SYSTEM

    def gen():
        try:
            for ev in lm.chat_stream(
                    req.messages, system=system, max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature, top_p=req.top_p, top_k=req.top_k,
                    repetition_penalty=req.repetition_penalty):
                yield f"data: {json.dumps(ev)}\n\n"
        except Exception as e:  # surface errors in-stream so the UI can show them
            yield f"data: {json.dumps(dict(error=f'{type(e).__name__}: {e}'))}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


class CompleteReq(BaseModel):
    prompt: str
    max_new_tokens: int = 160
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 0
    repetition_penalty: float = 1.0
    checkpoint: str | None = None


@app.post("/api/complete")
def complete(req: CompleteReq):
    lm = get_lm(req.checkpoint)

    def gen():
        try:
            for ev in lm.stream(req.prompt, max_new_tokens=req.max_new_tokens,
                                temperature=req.temperature, top_p=req.top_p,
                                top_k=req.top_k, repetition_penalty=req.repetition_penalty,
                                allow_special=False):
                yield f"data: {json.dumps(ev)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps(dict(error=f'{type(e).__name__}: {e}'))}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


# ------------------------------------------------------------------ static
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


class NoCacheStatic(StaticFiles):
    """A local analysis dashboard changes underfoot; never let the browser cache it."""

    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return resp


app.mount("/static", NoCacheStatic(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"),
                        headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
