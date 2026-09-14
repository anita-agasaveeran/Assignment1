"""Typed configuration objects.

Every architectural or optimization choice that the AutoResearch hill-climber is
allowed to toggle lives here as a named field, so a trial is fully described by a
(config, seed) pair and is therefore reproducible from the run database alone.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any


def _round_to(x: float, multiple: int) -> int:
    return int(multiple * round(x / multiple))


@dataclass
class ModelConfig:
    """Decoder-only transformer. Defaults = the modern (post-LLaMA) recipe.

    Each switchable field names the paper that introduced it; see slm/registry.py
    for the full citation and the claim we are testing.
    """
    vocab_size: int = 8192
    n_layer: int = 6
    n_head: int = 6
    n_kv_head: int = 2            # GQA groups; == n_head -> MHA, == 1 -> MQA, <=0 -> MHA
    d_model: int = 384
    d_ff: int | None = None       # None -> derived from `ffn`
    max_seq_len: int = 512

    norm: str = "rmsnorm"         # rmsnorm | layernorm
    norm_placement: str = "pre"   # pre | post
    pos: str = "rope"             # rope | learned | none
    ffn: str = "swiglu"           # swiglu | gelu
    qk_norm: bool = True          # normalize q,k before attention
    tie_embeddings: bool = True   # share input embedding with output head
    bias: bool = False            # biases in linear layers

    dropout: float = 0.0
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    init_std: float = 0.02
    scale_resid_init: bool = True  # scale residual-projection init by 1/sqrt(2L)
    z_loss: float = 0.0            # auxiliary log-Z^2 penalty
    logit_soft_cap: float = 0.0    # 0 disables
    attn_soft_cap: float = 0.0     # 0 disables

    head_dim: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        # n_kv_head <= 0 means "full multi-head", resolved relative to n_head so
        # the MHA ablation is portable across model widths.
        if self.n_kv_head <= 0:
            self.n_kv_head = self.n_head
        if self.d_model % self.n_head:
            raise ValueError(f"d_model={self.d_model} not divisible by n_head={self.n_head}")
        if self.n_head % self.n_kv_head:
            raise ValueError(f"n_head={self.n_head} not divisible by n_kv_head={self.n_kv_head}")
        if self.norm not in ("rmsnorm", "layernorm"):
            raise ValueError(f"unknown norm {self.norm!r}")
        if self.pos not in ("rope", "learned", "none"):
            raise ValueError(f"unknown pos {self.pos!r}")
        if self.ffn not in ("swiglu", "gelu"):
            raise ValueError(f"unknown ffn {self.ffn!r}")
        if self.norm_placement not in ("pre", "post"):
            raise ValueError(f"unknown norm_placement {self.norm_placement!r}")
        object.__setattr__(self, "head_dim", self.d_model // self.n_head)
        if self.d_ff is None:
            # SwiGLU has 3 matrices instead of 2, so 8/3*d keeps the parameter
            # count matched to a 4*d GELU block (Shazeer 2020, arXiv:2002.05202).
            self.d_ff = _round_to(8 / 3 * self.d_model, 64) if self.ffn == "swiglu" \
                else 4 * self.d_model


@dataclass
class DataConfig:
    corpus: str = "data/processed/stories"       # prefix -> .train.bin/.val.bin/.meta.json
    tokenizer: str = "data/processed/tokenizer.json"
    seq_len: int = 512
    # Data Preparation knobs (CRISP-DM phase 3)
    min_doc_chars: int = 64
    exact_dedup: bool = True
    near_dedup: bool = True
    near_dup_threshold: float = 0.8   # Jaccard, estimated by MinHash
    minhash_perms: int = 64
    minhash_bands: int = 16
    shingle_size: int = 5             # word-level shingles


@dataclass
class TrainConfig:
    steps: int = 2000
    batch_size: int = 32
    grad_accum: int = 1
    seq_len: int = 512

    optimizer: str = "adamw"          # adamw | muon
    lr: float = 3e-3
    muon_lr: float = 0.02
    min_lr_ratio: float = 0.1
    warmup_steps: int = 100
    schedule: str = "cosine"          # cosine | wsd | linear | constant
    wsd_decay_frac: float = 0.2
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0
    label_smoothing: float = 0.0

    device: str = "auto"              # auto | mps | cuda | cpu
    dtype: str = "bfloat16"           # bfloat16 | float32
    compile: bool = False
    seed: int = 1337

    eval_every: int = 100
    eval_iters: int = 40
    log_every: int = 10
    sample_every: int = 500
    ckpt_every: int = 0               # 0 -> only best + last
    save_checkpoints: bool = True     # False for throwaway AutoResearch proxy trials


@dataclass
class RunConfig:
    name: str = "baseline"
    phase: str = "pretrain"           # pretrain | sft
    notes: str = ""
    tags: list[str] = field(default_factory=list)
    out_dir: str = "runs"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["model"].pop("head_dim", None)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunConfig":
        d = copy.deepcopy(d)
        m = d.pop("model", {}) or {}
        m.pop("head_dim", None)
        da = d.pop("data", {}) or {}
        t = d.pop("train", {}) or {}
        known = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if k in known}
        return cls(model=ModelConfig(**m), data=DataConfig(**da), train=TrainConfig(**t), **d)

    def fingerprint(self) -> str:
        """Stable hash of everything that changes the numerical result."""
        d = self.to_dict()
        for k in ("name", "notes", "tags", "out_dir"):
            d.pop(k, None)
        blob = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]


def apply_overrides(cfg: RunConfig, overrides: dict[str, Any]) -> RunConfig:
    """Return a copy of `cfg` with dotted-path fields replaced.

    >>> apply_overrides(cfg, {"model.qk_norm": True, "train.lr": 1e-3})
    """
    d = cfg.to_dict()
    for path, value in overrides.items():
        node, *rest = path.split(".")
        cur = d
        keys = [node, *rest]
        for k in keys[:-1]:
            if k not in cur or not isinstance(cur[k], dict):
                raise KeyError(f"bad override path {path!r} (at {k!r})")
            cur = cur[k]
        if keys[-1] not in cur:
            raise KeyError(f"bad override path {path!r} (no field {keys[-1]!r})")
        cur[keys[-1]] = value
    # d_ff must be re-derived when `ffn` changes unless explicitly overridden
    if any(k.startswith("model.ffn") for k in overrides) and "model.d_ff" not in overrides:
        d["model"]["d_ff"] = None
    return RunConfig.from_dict(d)


def load_config(path: str) -> RunConfig:
    import yaml
    with open(path) as f:
        return RunConfig.from_dict(yaml.safe_load(f) or {})


def save_config(cfg: RunConfig, path: str) -> None:
    import yaml
    with open(path, "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
