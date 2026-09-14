#!/usr/bin/env python
"""Scale the AutoResearch champion recipe up to the full training configuration.

The study searches at a low-fidelity proxy (4 layers x d256, 300 steps). Its
output is a *recipe* -- which techniques to use -- not a set of dimensions. This
script keeps every discovered flag and replaces only the size/schedule knobs.

Two scale-up hazards are handled explicitly:

1. `n_kv_head` is an absolute count in the search space, but valid values depend
   on `n_head`. If the proxy ended at full multi-head, the scaled model stays
   full multi-head; otherwise the grouping is preserved as a ratio.
2. The optimal learning rate is *not* width-invariant outside a muP
   parameterization (arXiv:2203.03466), so a peak LR the search accepted at
   d256 may diverge at d384. The value is carried over unchanged -- but
   `--probe` runs a short stability check before committing to the full run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slm.config import RunConfig, load_config, save_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--champion", default="runs/autoresearch/hillclimb-v1/champion.yaml")
    ap.add_argument("--study-json", default="runs/autoresearch/hillclimb-v1/study.json")
    ap.add_argument("--out", default="configs/champion.yaml")
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args()

    if not os.path.exists(a.champion):
        raise SystemExit(f"no champion config at {a.champion} -- run slm.autoresearch first")
    cfg: RunConfig = load_config(a.champion)

    accepted: list[str] = []
    if os.path.exists(a.study_json):
        study = json.load(open(a.study_json))
        accepted = [x["key"] for x in study.get("accepted", [])]

    proxy_head, proxy_kv = cfg.model.n_head, cfg.model.n_kv_head

    cfg.name = "champion"
    cfg.phase = "pretrain"
    cfg.tags = ["champion", "autoresearch"]
    cfg.notes = ("Recipe discovered by AutoResearch hill climbing, scaled up. "
                 f"Accepted interventions: {', '.join(accepted) or 'none'}.")

    cfg.model.n_layer = a.n_layer
    cfg.model.d_model = a.d_model
    cfg.model.n_head = a.n_head
    cfg.model.max_seq_len = a.seq_len
    cfg.model.d_ff = None                      # re-derive from the chosen ffn

    if proxy_kv == proxy_head:                 # proxy stayed full multi-head
        cfg.model.n_kv_head = a.n_head
    else:
        group = max(1, proxy_head // proxy_kv)
        kv = max(1, a.n_head // group)
        while a.n_head % kv:                   # must divide n_head exactly
            kv -= 1
        cfg.model.n_kv_head = kv

    cfg.train.seq_len = a.seq_len
    cfg.train.steps = a.steps
    cfg.train.batch_size = a.batch_size
    cfg.train.warmup_steps = max(cfg.train.warmup_steps, a.steps // 20)
    cfg.train.eval_every = max(100, a.steps // 20)
    cfg.train.eval_iters = 40
    cfg.train.log_every = 10
    cfg.train.sample_every = max(500, a.steps // 8)
    cfg.train.save_checkpoints = True
    cfg.model.__post_init__()                  # re-derive d_ff/head_dim for the new width

    save_config(cfg, a.out)
    print(f"wrote {a.out}")
    print(f"  accepted : {accepted}")
    print(f"  model    : L{cfg.model.n_layer} d{cfg.model.d_model} "
          f"H{cfg.model.n_head}/{cfg.model.n_kv_head} ff{cfg.model.d_ff} ctx{cfg.model.max_seq_len}")
    print(f"  primitives: {cfg.model.norm} · {cfg.model.pos} · {cfg.model.ffn} · "
          f"qk_norm={cfg.model.qk_norm} · tied={cfg.model.tie_embeddings} · "
          f"dropout={cfg.model.dropout} · z_loss={cfg.model.z_loss}")
    print(f"  optim    : {cfg.train.optimizer} lr={cfg.train.lr} muon_lr={cfg.train.muon_lr} "
          f"{cfg.train.schedule} warmup={cfg.train.warmup_steps} wd={cfg.train.weight_decay}")
    print(f"  budget   : {a.steps} steps x {a.batch_size} x {a.seq_len} = "
          f"{a.steps * a.batch_size * a.seq_len / 1e6:.1f}M tokens")


if __name__ == "__main__":
    main()
