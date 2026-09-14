"""FastAPI service: the dashboard's backend and the rule set's inference surface.

CRISP-DM Phase 6 (Deployment) is where most academic projects stop at "we could
deploy this".  Here the rule set is actually served: `/api/recommend` takes a
partial basket and returns ranked next-item suggestions with the evidence
(support, confidence, lift, holdout status) attached to each one, which is the
form a recommender service or a merchandiser's tool would consume.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from armlab.config import Config

ARTIFACTS = {
    "run": "run.json",
    "profile": "data_profile.json",
    "preparation": "preparation.json",
    "itemsets": "itemsets.json",
    "rules": "rules.json",
    "autoresearch": "autoresearch.json",
    "evaluation": "evaluation.json",
    "properties": "properties.json",
    "network": "network.json",
    "model_card": "model_card.json",
}


class _Store:
    """mtime-aware artifact cache, so a re-run is picked up without a restart."""

    def __init__(self, cfg: Config) -> None:
        self.dir = cfg.paths.resolve("artifacts")
        self._cache: dict[str, tuple[float, Any]] = {}

    def path(self, name: str) -> Path:
        return self.dir / ARTIFACTS[name]

    def get(self, name: str) -> Any:
        if name not in ARTIFACTS:
            raise HTTPException(404, f"unknown artifact '{name}'")
        p = self.path(name)
        if not p.exists():
            raise HTTPException(
                503, f"artifact '{name}' not built yet -- run `armlab run` first")
        mtime = p.stat().st_mtime
        hit = self._cache.get(name)
        if hit and hit[0] == mtime:
            return hit[1]
        data = json.loads(p.read_text())
        self._cache[name] = (mtime, data)
        return data

    def available(self) -> dict[str, bool]:
        return {k: self.path(k).exists() for k in ARTIFACTS}


class RecommendRequest(BaseModel):
    items: list[str] = Field(..., description="Product descriptions or StockCodes "
                                              "already in the basket")
    top_k: int = Field(10, ge=1, le=100)
    validated_only: bool = Field(True, description="Only use rules that replicated "
                                                   "on the temporal holdout")
    min_confidence: float = Field(0.0, ge=0.0, le=1.0)


def create_app(cfg: Config | None = None) -> FastAPI:
    cfg = cfg or Config.load()
    store = _Store(cfg)
    app = FastAPI(title="ARMLab", version=cfg.fingerprint(),
                  description="Associative pattern mining -- CRISP-DM pipeline, "
                              "AutoResearch hill climbing, and rule serving.")

    # Web assets ship inside the package so the service is self-contained.
    dashboard = Path(__file__).resolve().parents[1] / "web"

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "config_fingerprint": cfg.fingerprint(),
                "artifacts": store.available(), "served_at": time.time()}

    @app.get("/api/manifest")
    def manifest() -> dict[str, Any]:
        av = store.available()
        out: dict[str, Any] = {"artifacts": av, "config": cfg.to_dict()}
        if av.get("run"):
            out["run"] = store.get("run")
        return out

    for key in ARTIFACTS:
        if key == "rules":
            continue

        def _make(k: str):
            def handler() -> Any:
                return store.get(k)
            return handler

        app.add_api_route(f"/api/{key.replace('_', '-')}", _make(key),
                          methods=["GET"], name=f"get_{key}")

    @app.get("/api/rules")
    def rules(
        limit: int = Query(200, ge=1, le=5000),
        offset: int = Query(0, ge=0),
        sort: str = Query("leverage"),
        desc: bool = Query(True),
        min_lift: float = Query(0.0),
        min_confidence: float = Query(0.0),
        min_support: float = Query(0.0),
        validated_only: bool = Query(False),
        significant_only: bool = Query(False),
        productive_only: bool = Query(False),
        antecedent_len: int = Query(0, description="0 = any"),
        search: str = Query(""),
    ) -> dict[str, Any]:
        payload = store.get("rules")
        rows: list[dict] = payload["rules"]
        q = search.strip().lower()

        def keep(r: dict) -> bool:
            if (r.get("lift") or 0) < min_lift: return False
            if (r.get("confidence") or 0) < min_confidence: return False
            if (r.get("support") or 0) < min_support: return False
            if validated_only and not r.get("validated"): return False
            if significant_only and not r.get("significant_searchspace"): return False
            if productive_only and not r.get("productive"): return False
            if antecedent_len and r.get("antecedent_len") != antecedent_len: return False
            if q and q not in (r.get("rule") or "").lower(): return False
            return True

        filtered = [r for r in rows if keep(r)]
        if sort in (filtered[0] if filtered else {}):
            filtered.sort(key=lambda r: (r.get(sort) is None, r.get(sort) or 0),
                          reverse=desc)
        return {
            "params": payload.get("params"),
            "objective": payload.get("objective"),
            "diagnostics": payload.get("diagnostics"),
            "total": len(rows), "matched": len(filtered),
            "offset": offset, "limit": limit,
            "rules": filtered[offset: offset + limit],
        }

    @app.post("/api/recommend")
    def recommend(req: RecommendRequest) -> dict[str, Any]:
        """Serve the rule set: partial basket -> ranked next items with evidence."""
        payload = store.get("rules")
        basket = {s.strip().upper() for s in req.items if s.strip()}
        if not basket:
            raise HTTPException(400, "provide at least one item")

        scored: dict[str, dict[str, Any]] = {}
        fired = []
        for r in payload["rules"]:
            ante = set(r["antecedent_items"]) | set(r["antecedent_codes"])
            ante = {a.upper() for a in ante if a}
            names = {a.upper() for a in r["antecedent_items"]}
            if not names or not names <= basket:
                continue
            if req.validated_only and not r.get("validated"):
                continue
            if (r.get("confidence") or 0) < req.min_confidence:
                continue
            fired.append(r["rule_id"])
            for item, code in zip(r["consequent_items"], r["consequent_codes"]):
                if item.upper() in basket:
                    continue
                # Rank by confidence, break ties toward higher lift; keep the
                # strongest single piece of evidence per candidate item.
                cand = scored.get(item)
                if cand is None or r["confidence"] > cand["confidence"]:
                    scored[item] = {
                        "item": item, "stock_code": code,
                        "confidence": r["confidence"], "lift": r["lift"],
                        "support": r["support"], "leverage": r["leverage"],
                        "kulczynski": r.get("kulczynski"),
                        "p_fisher": r.get("p_fisher"),
                        "holdout_confidence": r.get("holdout_confidence"),
                        "holdout_validated": bool(r.get("validated")),
                        "evidence_rule": r["rule"], "rule_id": r["rule_id"],
                    }
        ranked = sorted(scored.values(),
                        key=lambda d: (d["confidence"], d["lift"]), reverse=True)
        return {
            "basket": sorted(basket),
            "n_rules_fired": len(fired),
            "n_candidates": len(ranked),
            "recommendations": ranked[: req.top_k],
            "note": "Associations, not causal effects. Confidence is the "
                    "in-sample estimate; holdout_confidence is the same rule "
                    "measured on transactions the miner never saw.",
        }

    @app.get("/api/items")
    def items(q: str = Query("", min_length=0), limit: int = Query(30, le=200)) -> Any:
        prep = store.get("preparation")
        pool = prep.get("top_items", [])
        ql = q.strip().lower()
        hits = [i for i in pool if ql in i["label"].lower()] if ql else pool
        return {"items": hits[:limit]}

    if dashboard.exists():
        app.mount("/static", StaticFiles(directory=str(dashboard)), name="static")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(str(dashboard / "index.html"))

    @app.middleware("http")
    async def _no_store(request, call_next):
        """Artifacts are rebuilt in place by `armlab run`. Letting a browser
        heuristically cache them means the dashboard silently shows the previous
        run's numbers, which is worse than a slow reload."""
        resp = await call_next(request)
        resp.headers["Cache-Control"] = "no-store, must-revalidate"
        return resp

    @app.exception_handler(HTTPException)
    def _http_err(_, exc: HTTPException) -> JSONResponse:
        return JSONResponse({"error": exc.detail, "status": exc.status_code},
                            status_code=exc.status_code)

    return app


app = create_app()
