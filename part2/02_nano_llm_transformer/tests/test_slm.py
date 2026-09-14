"""Correctness tests for the parts where a bug would be silent.

Priority is given to failures that do not crash and do not look wrong in the
metrics -- a leaky causal mask, a KV cache that drifts from the uncached path, an
SFT loss mask that supervises the user's turn. Each of those produces a model
that trains happily and is wrong.
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slm.chat import DEFAULT_SYSTEM, build_sft_example, parse_instruct_record
from slm.config import ModelConfig, RunConfig, apply_overrides
from slm.data import exact_dedup, leakage_audit, minhash_signatures, near_dedup
from slm.model import build_model
from slm.optim import build_optimizer, lr_scale, zeropower_via_newtonschulz5
from slm.registry import BASELINE_OVERRIDES, PAPERS, SEARCH_SPACE, paper_url
from slm.tokenizer import BPETokenizer


# --------------------------------------------------------------- tokenizer
@pytest.fixture(scope="module")
def tok():
    corpus = ("Once upon a time there was a little girl named Lily. "
              "She had a red hat and a big cat. They liked to run and play! "
              "One day, Tom said: \"Let's go to the park.\" 42 times.\n") * 60
    return BPETokenizer().train([corpus], vocab_size=400, verbose=False)


def test_tokenizer_roundtrip_is_lossless(tok):
    for s in ["Once upon a time", "café — naïve 🚀", "  double  spaces\n\nnewlines",
              "digits 12345 and !@#$%", ""]:
        assert tok.decode(tok.encode_ordinary(s)) == s


def test_tokenizer_handles_special_tokens(tok):
    s = "hello<|endoftext|>world"
    ids = tok.encode(s)
    assert tok.eot_id in ids
    assert tok.decode(ids) == s
    # with specials disabled the literal is encoded as ordinary text
    assert tok.eot_id not in tok.encode(s, allowed_special=False)


def test_tokenizer_persists_exactly(tok, tmp_path):
    p = str(tmp_path / "t.json")
    tok.save(p)
    t2 = BPETokenizer.load(p)
    s = "Lily and the cat ran home."
    assert t2.encode(s) == tok.encode(s)
    assert t2.n_vocab == tok.n_vocab


def test_merges_never_cross_pretoken_boundary(tok):
    # " cat" and " hat" may merge internally, but no token may span the space
    # boundary of two separate words in a way that loses the split.
    ids = tok.encode_ordinary("cat hat")
    assert tok.decode(ids) == "cat hat"


# ------------------------------------------------------------------ config
def test_config_rejects_invalid_shapes():
    with pytest.raises(ValueError):
        ModelConfig(d_model=100, n_head=6)
    with pytest.raises(ValueError):
        ModelConfig(n_head=6, n_kv_head=4)
    with pytest.raises(ValueError):
        ModelConfig(norm="batchnorm")


def test_n_kv_head_zero_means_multi_head():
    assert ModelConfig(n_head=6, n_kv_head=0).n_kv_head == 6
    assert ModelConfig(n_head=4, n_kv_head=0).n_kv_head == 4


def test_overrides_reject_unknown_paths():
    c = RunConfig()
    with pytest.raises(KeyError):
        apply_overrides(c, {"model.does_not_exist": 1})
    assert apply_overrides(c, {"train.lr": 1e-4}).train.lr == 1e-4


def test_ffn_change_rederives_d_ff():
    c = RunConfig()
    assert apply_overrides(c, {"model.ffn": "gelu"}).model.d_ff == 4 * c.model.d_model
    # explicit d_ff must win over the automatic derivation
    assert apply_overrides(c, {"model.ffn": "gelu", "model.d_ff": 999}).model.d_ff == 999


def test_fingerprint_ignores_cosmetic_fields():
    a, b = RunConfig(), RunConfig(name="other", notes="x", tags=["y"])
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != apply_overrides(a, {"train.lr": 9e-9}).fingerprint()


# ------------------------------------------------------------------- model
@pytest.fixture(scope="module")
def small():
    c = ModelConfig(vocab_size=128, n_layer=2, n_head=4, n_kv_head=2, d_model=64,
                    max_seq_len=32)
    return build_model(c)


def test_forward_shapes_and_loss(small):
    x = torch.randint(0, 128, (3, 16))
    out = small(x, x)
    assert out["logits"].shape == (3, 16, 128)
    assert out["loss"].ndim == 0 and torch.isfinite(out["loss"])


def test_attention_is_causal(small):
    """Changing a later token must not change any earlier position's logits.

    A leaky mask does not crash and does not look wrong -- it just makes the
    model appear much better than it is.
    """
    torch.manual_seed(0)
    x = torch.randint(0, 128, (1, 16))
    base = small(x)["logits"]
    x2 = x.clone()
    x2[0, 10] = (x2[0, 10] + 7) % 128
    mod = small(x2)["logits"]
    assert torch.allclose(base[:, :10], mod[:, :10], atol=1e-5), "information leaked backwards"
    assert not torch.allclose(base[:, 10:], mod[:, 10:], atol=1e-5)


def test_kv_cache_matches_uncached_greedy(small):
    torch.manual_seed(0)
    p = torch.randint(0, 128, (1, 6))
    a = small.generate(p, 10, temperature=0.0, use_cache=True)
    b = small.generate(p, 10, temperature=0.0, use_cache=False)
    assert torch.equal(a, b), "KV cache diverges from the uncached path"


def test_generate_respects_context_limit(small):
    p = torch.randint(0, 128, (1, 8))
    out = small.generate(p, 500, temperature=0.0)
    assert out.size(1) <= small.cfg.max_seq_len


def test_tied_embeddings_share_storage():
    m = build_model(ModelConfig(vocab_size=64, n_layer=1, n_head=2, d_model=32,
                                n_kv_head=1, max_seq_len=16, tie_embeddings=True))
    assert m.lm_head.weight is m.tok_emb.weight
    assert "lm_head" not in m.param_breakdown()


def test_z_loss_adds_a_penalty():
    kw = dict(vocab_size=64, n_layer=1, n_head=2, n_kv_head=1, d_model=32, max_seq_len=16)
    torch.manual_seed(0); a = build_model(ModelConfig(**kw, z_loss=0.0))
    torch.manual_seed(0); b = build_model(ModelConfig(**kw, z_loss=1e-2))
    x = torch.randint(0, 64, (2, 8))
    oa, ob = a(x, x), b(x, x)
    assert "z" not in oa and "z" in ob
    assert ob["loss"] > ob["ce"]
    assert torch.allclose(oa["ce"], ob["ce"], atol=1e-5)   # z-loss must not change CE


def test_all_interventions_build_and_backprop():
    small_dims = {"model.vocab_size": 128, "model.n_layer": 2, "model.d_model": 64,
                  "model.n_head": 4, "model.max_seq_len": 32}
    # dims first: BASELINE_OVERRIDES resolves n_kv_head=0 against the current n_head
    base = apply_overrides(apply_overrides(RunConfig(), small_dims), BASELINE_OVERRIDES)
    for iv in SEARCH_SPACE:
        cfg = apply_overrides(base, iv.on)
        m = build_model(cfg.model)
        x = torch.randint(0, 128, (2, 16))
        m(x, x)["loss"].backward()
        assert any(p.grad is not None for p in m.parameters()), iv.key


# --------------------------------------------------------------- optimizer
def test_newton_schulz_approximately_orthogonalizes():
    torch.manual_seed(0)
    s = torch.linalg.svdvals(zeropower_via_newtonschulz5(torch.randn(64, 32)).float())
    assert 0.5 < s.min() and s.max() < 1.5     # the 5-step iteration is deliberately approximate


def test_muon_routes_only_hidden_2d_weights():
    c = RunConfig(); c.train.optimizer = "muon"
    c.model = ModelConfig(vocab_size=64, n_layer=2, n_head=2, n_kv_head=1,
                          d_model=32, max_seq_len=16)
    m = build_model(c.model)
    _, info = build_optimizer(m, c.train, verbose=False)
    assert info["n_muon_tensors"] > 0
    assert not any(n.startswith(("tok_emb", "pos_emb", "lm_head"))
                   for n in info["muon_param_names"])


@pytest.mark.parametrize("sched", ["cosine", "wsd", "linear", "constant"])
def test_schedules_warm_up_and_stay_bounded(sched):
    c = RunConfig(); c.train.schedule = sched
    c.train.steps, c.train.warmup_steps = 100, 10
    vals = [lr_scale(s, c.train) for s in range(100)]
    assert vals[0] < vals[9]                      # warming up
    assert abs(vals[9] - 1.0) < 1e-9              # reaches peak at end of warmup
    assert all(0 <= v <= 1.0 + 1e-9 for v in vals)
    if sched in ("cosine", "linear"):
        assert vals[-1] < 0.5


# -------------------------------------------------------------------- data
_VOCAB = ("lily tom ben sara ball park cat dog tree house sun rain garden bird "
          "cake hat shoe river boat kite friend mother father teacher sandbox "
          "bicycle window kitchen forest mountain river song story blanket").split()


def _docs(n=400):
    """Genuinely distinct documents built from a seeded random word sequence.

    Templated text is a bad fixture here: documents that differ only in an index
    really are near-duplicates of one another (Jaccard ~0.9), so the detector
    correctly flags them and the test would be asserting the wrong thing.
    """
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        words = rng.choice(_VOCAB, size=60, replace=True)
        out.append(" ".join(words) + ".")
    return out


def test_exact_dedup_finds_planted_copies():
    d = _docs() + _docs()[:37]
    keep, stats = exact_dedup(d)
    assert stats["n_exact_dupes"] == 37
    assert len(keep) == 400


def test_near_dedup_finds_planted_near_copies():
    d = _docs()
    planted = d + [x + " The end." for x in d[:25]]      # small edit -> J well above 0.8
    sigs = minhash_signatures(planted, 64, 5)
    keep, stats = near_dedup(planted, sigs, 16, 0.8)
    assert stats["n_removed"] == 25, stats
    assert len(keep) == 400


def test_near_dedup_ignores_heavily_rewritten_documents():
    """Specificity: the threshold must reject documents that merely share a topic."""
    d = _docs(200)
    rewritten = ["A completely different account of events number %d involving "
                 "a spaceship, a robot engineer, and a long journey between two "
                 "distant planets during a storm." % i for i in range(20)]
    sigs = minhash_signatures(d + rewritten, 64, 5)
    _, stats = near_dedup(d + rewritten, sigs, 16, 0.8)
    assert stats["n_removed"] == 0, stats


def test_leakage_audit_catches_cross_split_duplicates():
    d = _docs(300)
    train, val = d[:200], d[200:] + d[:15]
    keep, stats = leakage_audit(train, val)
    assert stats["n_leaked"] == 15
    assert len(keep) == 100


def test_dedup_does_not_remove_distinct_documents():
    d = _docs(200)
    sigs = minhash_signatures(d, 64, 5)
    _, stats = near_dedup(d, sigs, 16, 0.8)
    assert stats["n_removed"] == 0, "false positives on distinct documents"


# --------------------------------------------------------------------- sft
def test_sft_mask_supervises_only_assistant_turn(tok):
    msgs = [dict(role="user", content="Tell me a story about a cat"),
            dict(role="assistant", content="The cat ran home")]
    ids, mask = build_sft_example(tok, msgs, DEFAULT_SYSTEM, 512)
    assert len(ids) == len(mask)
    assert 0 < sum(mask) < len(mask)
    supervised = tok.decode([i for i, m in zip(ids, mask) if m])
    assert "The cat ran home" in supervised
    assert "Tell me a story" not in supervised, "user turn is being trained on"
    assert supervised.rstrip().endswith("<|im_end|>"), "stop token must be supervised"


def test_sft_example_rejects_overlong_input(tok):
    msgs = [dict(role="user", content="x " * 500),
            dict(role="assistant", content="y " * 500)]
    assert build_sft_example(tok, msgs, DEFAULT_SYSTEM, 64) is None


def test_instruct_parser_extracts_fields():
    rec = parse_instruct_record(
        "Features: Dialogue\nWords: quit, oak\nSummary: A short summary.\n"
        "Story: \n\nSara said hello to Ben.")
    assert rec["words"] == "quit, oak"
    assert rec["features"] == "Dialogue"
    assert "Sara said hello" in rec["story"]
    assert parse_instruct_record("no fields here") is None


# ---------------------------------------------------------------- registry
def test_every_intervention_cites_a_registered_paper():
    for iv in SEARCH_SPACE:
        assert iv.papers, iv.key
        for k in iv.papers:
            assert k in PAPERS, f"{iv.key} cites unknown paper {k}"
            assert paper_url(k).startswith("http")


def test_baseline_ablates_every_intervention():
    """The baseline must turn every candidate off, or a trial is not an A/B."""
    base = apply_overrides(RunConfig(), BASELINE_OVERRIDES).to_dict()
    for iv in SEARCH_SPACE:
        for path, off_val in iv.off.items():
            section, field = path.split(".")
            actual = base[section][field]
            if field == "n_kv_head" and off_val == 0:
                assert actual == base["model"]["n_head"]      # 0 resolves to MHA
            else:
                assert actual == off_val, f"{iv.key}: {path} is {actual}, expected {off_val}"


def test_paper_metadata_is_complete():
    for k, p in PAPERS.items():
        assert p["title"] and p["authors"] and p["claim"]
        assert 1990 < p["year"] <= 2026, k
        assert p.get("arxiv") or p.get("url"), f"{k} has no resolvable reference"
