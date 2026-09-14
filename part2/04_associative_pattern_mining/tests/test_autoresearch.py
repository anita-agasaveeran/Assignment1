"""Search space encoding and the hill climber."""
import numpy as np
import pytest

from armlab.autoresearch import hillclimb, space
from armlab.autoresearch.objective import Score
from armlab.config import AutoResearchConfig


def test_encode_decode_roundtrip():
    p = {"min_support": 0.02, "min_confidence": 0.4, "min_lift": 2.0,
         "max_antecedent_len": 2, "max_itemset_len": 4}
    back = space.decode(space.encode(p))
    for k, v in p.items():
        assert back[k] == pytest.approx(v, rel=1e-6) if isinstance(v, float) else back[k] == v


def test_decode_clamps_to_bounds():
    for u in (np.zeros(len(space.SPACE)), np.ones(len(space.SPACE)),
              np.full(len(space.SPACE), -5.0), np.full(len(space.SPACE), 5.0)):
        p = space.decode(u)
        for d in space.SPACE:
            assert d.low - 1e-9 <= p[d.name] <= d.high + 1e-9


def test_antecedent_len_never_exceeds_itemset_len():
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = space.decode(rng.random(len(space.SPACE)))
        assert p["max_antecedent_len"] <= p["max_itemset_len"] - 1
        assert p["max_antecedent_len"] >= 1


def test_hillclimb_finds_a_known_optimum():
    """A smooth synthetic objective with its peak away from the seed."""
    target = np.array([0.72, 0.31, 0.55, 0.5, 0.9])

    def evaluate(params):
        u = space.encode(params)
        val = float(np.exp(-4 * np.sum((u - target) ** 2)))
        return Score(val, {"v": val}, {"v": val}, {}), {}

    cfg = AutoResearchConfig(max_evaluations=180, restarts=2, neighbours_per_step=5,
                             patience=8, seed=1)
    seed_params = space.decode(np.zeros(len(space.SPACE)))
    res = hillclimb.search(evaluate, cfg, seed_params)

    assert res.n_evaluations <= cfg.max_evaluations
    assert res.best_score > res.seed_score
    assert res.improvement_over_seed > 0
    assert res.best_score > 0.5, "hill climber failed to approach the optimum"
    assert len(res.ledger.trials) == res.n_evaluations
    assert res.convergence[-1]["best"] == pytest.approx(res.best_score)


def test_hillclimb_is_reproducible():
    def evaluate(params):
        u = space.encode(params)
        v = float(-np.sum((u - 0.4) ** 2))
        return Score(v, {}, {}, {}), {}

    cfg = AutoResearchConfig(max_evaluations=40, restarts=1, seed=99)
    seed_params = space.decode(np.full(len(space.SPACE), 0.1))
    a = hillclimb.search(evaluate, cfg, seed_params)
    b = hillclimb.search(evaluate, cfg, seed_params)
    assert a.best_score == b.best_score
    assert a.best_params == b.best_params


def test_best_so_far_is_monotone():
    rng = np.random.default_rng(0)

    def evaluate(params):
        v = float(rng.random())
        return Score(v, {}, {}, {}), {}

    cfg = AutoResearchConfig(max_evaluations=50, restarts=1, seed=4)
    res = hillclimb.search(evaluate, cfg, space.decode(np.full(len(space.SPACE), .5)))
    bests = [c["best"] for c in res.convergence]
    assert all(b2 >= b1 for b1, b2 in zip(bests, bests[1:]))


def test_objective_weights_sum_to_one():
    from armlab.autoresearch.objective import weights
    cfg = AutoResearchConfig()
    assert sum(weights(cfg).values()) == pytest.approx(1.0)


def test_objective_is_zero_without_rules():
    import pandas as pd
    from armlab.autoresearch.objective import score
    s = score(pd.DataFrame(), {}, 0.1, AutoResearchConfig())
    assert s.terms["validated_yield"] == 0.0
    assert s.value < 0.2
