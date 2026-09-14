"""API contract. Skipped when artifacts are not built."""
import json
from pathlib import Path

import pytest

from armlab.config import Config

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient

CFG = Config.load()
ART = CFG.paths.resolve("artifacts")
HAS_ARTIFACTS = (ART / "rules.json").exists()
needs_artifacts = pytest.mark.skipif(not HAS_ARTIFACTS,
                                     reason="run `./run_pipeline.sh` first")


@pytest.fixture(scope="module")
def client():
    from armlab.api.server import create_app
    return TestClient(create_app(CFG))


def test_health_always_answers(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "artifacts" in r.json()


def test_unknown_artifact_is_404(client):
    assert client.get("/api/definitely-not-a-thing").status_code == 404


@needs_artifacts
@pytest.mark.parametrize("path", [
    "/api/run", "/api/profile", "/api/preparation", "/api/itemsets",
    "/api/autoresearch", "/api/evaluation", "/api/properties", "/api/network",
    "/api/model-card", "/api/manifest",
])
def test_artifact_endpoints(client, path):
    r = client.get(path)
    assert r.status_code == 200
    assert isinstance(r.json(), dict)


@needs_artifacts
def test_rules_filtering_narrows_the_set(client):
    a = client.get("/api/rules?limit=500").json()
    b = client.get("/api/rules?limit=500&min_lift=3").json()
    assert b["matched"] <= a["matched"]
    assert all(r["lift"] >= 3 for r in b["rules"])
    c = client.get("/api/rules?limit=500&validated_only=true").json()
    assert all(r["validated"] for r in c["rules"])


@needs_artifacts
def test_rules_sorting(client):
    r = client.get("/api/rules?limit=50&sort=lift&desc=true").json()["rules"]
    lifts = [x["lift"] for x in r]
    assert lifts == sorted(lifts, reverse=True)


@needs_artifacts
def test_every_rule_carries_every_measure(client):
    from armlab.metrics.interestingness import MEASURES
    r = client.get("/api/rules?limit=5").json()["rules"]
    for rule in r:
        for k in MEASURES:
            assert k in rule, f"rule is missing measure {k}"
        for k in ("p_fisher", "q_value_searchspace", "productive", "n11", "N"):
            assert k in rule


@needs_artifacts
def test_recommend_requires_a_basket(client):
    assert client.post("/api/recommend", json={"items": []}).status_code == 400


@needs_artifacts
def test_recommend_never_suggests_what_is_already_in_the_basket(client):
    rules = client.get("/api/rules?limit=1&sort=confidence").json()["rules"]
    if not rules:
        pytest.skip("no rules")
    basket = rules[0]["antecedent_items"]
    r = client.post("/api/recommend", json={"items": basket, "validated_only": False}).json()
    assert r["n_rules_fired"] >= 1
    got = {x["item"].upper() for x in r["recommendations"]}
    assert not (got & {b.upper() for b in basket})


@needs_artifacts
def test_recommend_on_unknown_item_is_empty_not_an_error(client):
    r = client.post("/api/recommend", json={"items": ["NOT A REAL PRODUCT XYZ"]})
    assert r.status_code == 200
    assert r.json()["recommendations"] == []


@needs_artifacts
def test_artifacts_are_json_serialisable_and_finite():
    """NaN/Infinity are not valid JSON; a leak here silently breaks the dashboard."""
    for p in ART.glob("*.json"):
        text = p.read_text()
        assert "NaN" not in text and "Infinity" not in text, f"{p.name} contains non-JSON floats"
        json.loads(text)
