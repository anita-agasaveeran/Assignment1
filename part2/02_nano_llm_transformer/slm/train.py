"""CRISP-DM phase 4: Modeling. The training loop.

Design choices that exist specifically to make AutoResearch trials trustworthy:

* Evaluation batches are drawn from a fixed RNG that is independent of the run
  seed, so two trials are scored on *identical* held-out windows. Without this,
  a 0.01 difference in val loss is indistinguishable from batch luck.
* Throughput, MFU, memory and gradient norms are logged on the same step grid as
  loss, so a "win" that is really just a slower run is visible immediately.
* Every run writes its full config, environment and git state to the tracker
  before the first step, so any row in the dashboard can be re-run from the DB.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time

import numpy as np
import torch

from .config import RunConfig, load_config, save_config
from .model import SLM, build_model
from .optim import build_optimizer, clip_grad_norm, lr_scale
from .tokenizer import BPETokenizer
from .tracking import Tracker
from .utils import device_memory_mb, env_info, measure_peak_flops, pick_device, set_seed, sync

EVAL_SEED = 20240601          # fixed across all runs on purpose


# --------------------------------------------------------------------- data
class BinDataset:
    """Memory-mapped uint16 token stream sampled as random fixed-length windows."""

    def __init__(self, path: str, seq_len: int):
        self.path, self.seq_len = path, seq_len
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        if len(self.data) < seq_len + 1:
            raise ValueError(f"{path}: {len(self.data)} tokens < seq_len+1")

    def __len__(self) -> int:
        return len(self.data)

    def batch(self, batch_size: int, rng: np.random.Generator, device: torch.device):
        ix = rng.integers(0, len(self.data) - self.seq_len - 1, size=batch_size)
        x = np.stack([self.data[i:i + self.seq_len] for i in ix]).astype(np.int64)
        y = np.stack([self.data[i + 1:i + 1 + self.seq_len] for i in ix]).astype(np.int64)
        xt = torch.from_numpy(x).to(device, non_blocking=True)
        yt = torch.from_numpy(y).to(device, non_blocking=True)
        return xt, yt

    def fixed_batches(self, batch_size: int, n_batches: int, device: torch.device):
        """Deterministic eval set -- identical for every run, every seed."""
        rng = np.random.default_rng(EVAL_SEED)
        return [self.batch(batch_size, rng, device) for _ in range(n_batches)]


def git_state() -> dict:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=5).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                    text=True, timeout=5).stdout.strip())
        return dict(commit=sha, dirty=dirty)
    except Exception:
        return dict(commit=None, dirty=None)


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model: SLM, batches, device: torch.device, autocast_ctx,
             bytes_per_token: float = 0.0, collect_stats: bool = False) -> dict:
    model.eval()
    tot_ce, tot_n = 0.0, 0
    correct = top5 = 0
    conf_sum = 0.0
    logit_absmax = 0.0
    lse_sum = 0.0
    pos_loss = None
    for x, y in batches:
        with autocast_ctx:
            out = model(x, y, collect_stats=collect_stats)
        logits = out["logits"].float()
        ce = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.reshape(-1), reduction="none")
        n = y.numel()
        tot_ce += float(ce.sum()); tot_n += n
        probs = torch.softmax(logits, dim=-1)
        top = probs.topk(5, dim=-1)
        pred = top.indices[..., 0]
        correct += int((pred == y).sum())
        top5 += int((top.indices == y.unsqueeze(-1)).any(-1).sum())
        conf_sum += float(top.values[..., 0].sum())
        logit_absmax = max(logit_absmax, float(logits.abs().max()))
        lse_sum += float(torch.logsumexp(logits, dim=-1).sum())
        pl = ce.view(y.shape).mean(0)
        pos_loss = pl if pos_loss is None else pos_loss + pl
    model.train()

    ce_mean = tot_ce / max(tot_n, 1)
    res = dict(
        val_loss=ce_mean,
        val_ppl=float(math.exp(min(ce_mean, 20))),
        val_acc=correct / max(tot_n, 1),
        val_acc_top5=top5 / max(tot_n, 1),
        val_confidence=conf_sum / max(tot_n, 1),
        val_logit_absmax=logit_absmax,
        val_logZ=lse_sum / max(tot_n, 1),
    )
    if bytes_per_token > 0:
        # scale-free loss unit: comparable across tokenizers (arXiv:2010.14701)
        res["val_bits_per_byte"] = ce_mean / math.log(2) / bytes_per_token
    if pos_loss is not None:
        pl = (pos_loss / len(batches)).tolist()
        res["_pos_loss"] = pl
    return res


# ------------------------------------------------------------------- train
def train(cfg: RunConfig, tracker: Tracker | None = None, quiet: bool = False,
          progress_cb=None) -> dict:
    device = pick_device(cfg.train.device)
    set_seed(cfg.train.seed)
    use_amp = cfg.train.dtype == "bfloat16" and device.type in ("cuda", "mps")
    autocast_ctx = (torch.autocast(device_type=device.type, dtype=torch.bfloat16)
                    if use_amp else torch.autocast(device_type="cpu", enabled=False))

    meta_path = f"{cfg.data.corpus}.meta.json"
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    if meta.get("vocab_size") and meta["vocab_size"] != cfg.model.vocab_size:
        cfg.model.vocab_size = meta["vocab_size"]
    bpt = float(meta.get("bytes_per_token_val", 0.0))

    train_ds = BinDataset(f"{cfg.data.corpus}.train.bin", cfg.train.seq_len)
    val_ds = BinDataset(f"{cfg.data.corpus}.val.bin", cfg.train.seq_len)

    model = build_model(cfg.model).to(device)
    if cfg.train.compile:
        model = torch.compile(model)
    opt, opt_info = build_optimizer(model, cfg.train, verbose=not quiet)

    n_params = model.num_params(non_embedding=True)
    n_total = sum(p.numel() for p in model.parameters())
    fpt = model.flops_per_token(cfg.train.seq_len)
    # Measure the roofline in the dtype training actually uses, or MFU is
    # nonsense (a bf16 roofline against fp32 CPU training reports >100%).
    peak_flops = measure_peak_flops(device, torch.bfloat16 if use_amp else torch.float32)
    tokens_per_step = cfg.train.batch_size * cfg.train.grad_accum * cfg.train.seq_len
    total_tokens = tokens_per_step * cfg.train.steps

    env = env_info(device) | dict(git=git_state(), peak_flops_measured=peak_flops,
                                  amp=use_amp, dtype=cfg.train.dtype)
    own_tracker = tracker is None
    tracker = tracker or Tracker(os.path.join(cfg.out_dir, "tracking.db"))
    run_id = tracker.start_run(cfg.name, cfg.phase, cfg.to_dict(), env, cfg.tags,
                               cfg.notes, cfg.fingerprint())
    tracker.set_params(n_total, n_total - n_params)
    run_dir = os.path.join(cfg.out_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    save_config(cfg, os.path.join(run_dir, "config.yaml"))

    # Chinchilla accounting stated up front, not discovered later (arXiv:2203.15556)
    chinchilla_tokens = 20 * n_params
    budget = dict(
        n_params_nonemb=n_params, n_params_total=n_total, flops_per_token=fpt,
        tokens_per_step=tokens_per_step, total_train_tokens=total_tokens,
        epochs_over_corpus=round(total_tokens / max(len(train_ds), 1), 3),
        chinchilla_optimal_tokens=chinchilla_tokens,
        chinchilla_ratio=round(total_tokens / max(chinchilla_tokens, 1), 3),
        total_flops=fpt * total_tokens, peak_flops_measured=peak_flops,
        corpus_tokens=len(train_ds), val_tokens=len(val_ds),
    )
    tracker.event("budget", data=budget)
    if not quiet:
        print(f"[train] run {run_id} on {device} | {n_total/1e6:.2f}M params "
              f"({n_params/1e6:.2f}M non-emb)")
        print(f"[train] {total_tokens/1e6:.1f}M tokens = {budget['epochs_over_corpus']}x corpus, "
              f"{budget['chinchilla_ratio']:.2f}x Chinchilla-optimal")
        print(f"[train] {fpt/1e6:.1f} MFLOP/token, measured peak {peak_flops/1e12:.2f} TFLOP/s")

    eval_batches = val_ds.fixed_batches(cfg.train.batch_size, cfg.train.eval_iters, device)
    train_eval_batches = train_ds.fixed_batches(cfg.train.batch_size,
                                                max(4, cfg.train.eval_iters // 4), device)
    rng = np.random.default_rng(cfg.train.seed)

    tok = None
    tok_path = cfg.data.tokenizer
    if os.path.exists(tok_path):
        tok = BPETokenizer.load(tok_path)
    sample_prompts = ["Once upon a time", "Lily and Tom went to the", "The little dog"]

    best_val = float("inf")
    best_step = 0
    hist: list[dict] = []
    t_start = time.perf_counter()
    tokens_seen = 0
    model.train()
    status = "finished"

    try:
        for step in range(cfg.train.steps):
            t0 = time.perf_counter()
            scale = lr_scale(step, cfg.train)
            opt.set_lr_scale(scale)

            loss_acc = ce_acc = z_acc = 0.0
            for micro in range(cfg.train.grad_accum):
                x, y = train_ds.batch(cfg.train.batch_size, rng, device)
                with autocast_ctx:
                    out = model(x, y, label_smoothing=cfg.train.label_smoothing)
                loss = out["loss"] / cfg.train.grad_accum
                loss.backward()
                loss_acc += out["loss"].detach().item()
                ce_acc += out.get("ce", out["loss"]).detach().item()
                z = out.get("z")
                z_acc += z.detach().item() if z is not None else 0.0
            loss_acc /= cfg.train.grad_accum
            ce_acc /= cfg.train.grad_accum
            z_acc /= cfg.train.grad_accum

            gnorm = clip_grad_norm(model, cfg.train.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            sync(device)

            dt = time.perf_counter() - t0
            tokens_seen += tokens_per_step
            tps = tokens_per_step / dt
            mfu = (fpt * tps / peak_flops) if (peak_flops and math.isfinite(peak_flops)
                                               and peak_flops > 0) else 0.0

            if not math.isfinite(loss_acc):
                tracker.event(f"non-finite loss at step {step}", "error")
                status = "diverged"
                break

            if step % cfg.train.log_every == 0 or step == cfg.train.steps - 1:
                mem = device_memory_mb(device)
                rec = dict(train_loss=loss_acc, train_ce=ce_acc, z_loss=z_acc,
                           lr=opt.current_lrs().get("adamw", 0.0), lr_scale=scale,
                           grad_norm=gnorm, step_time_s=dt, tokens_per_s=tps, mfu=mfu,
                           tokens_seen=tokens_seen, **mem)
                if "muon" in opt.current_lrs():
                    rec["lr_muon"] = opt.current_lrs()["muon"]
                tracker.log(step, rec)
                if not quiet and (step % (cfg.train.log_every * 10) == 0):
                    print(f"  step {step:5d}/{cfg.train.steps} loss {loss_acc:.4f} "
                          f"lr {rec['lr']:.2e} gn {gnorm:6.2f} {tps:,.0f} tok/s "
                          f"mfu {mfu:.1%}")

            if (step + 1) % cfg.train.eval_every == 0 or step == cfg.train.steps - 1:
                ev = evaluate(model, eval_batches, device, autocast_ctx, bpt, collect_stats=False)
                tev = evaluate(model, train_eval_batches, device, autocast_ctx)
                pos_loss = ev.pop("_pos_loss", None)
                tev.pop("_pos_loss", None)
                ev["train_holdout_loss"] = tev["val_loss"]
                ev["generalization_gap"] = ev["val_loss"] - tev["val_loss"]
                ev["elapsed_s"] = time.perf_counter() - t_start
                tracker.log(step, ev, flush=True)
                hist.append(dict(step=step, **{k: v for k, v in ev.items()}))
                if ev["val_loss"] < best_val:
                    best_val, best_step = ev["val_loss"], step
                    if cfg.train.save_checkpoints:
                        ckpt = dict(model=model.state_dict(), config=cfg.to_dict(), step=step,
                                    val_loss=best_val, run_id=run_id, meta=meta)
                        torch.save(ckpt, os.path.join(run_dir, "best.pt"))
                    if pos_loss:
                        json.dump(pos_loss, open(os.path.join(run_dir, "pos_loss.json"), "w"))
                if not quiet:
                    print(f"  [eval] step {step:5d} val {ev['val_loss']:.4f} "
                          f"ppl {ev['val_ppl']:.2f} acc {ev['val_acc']:.3f} "
                          f"gap {ev['generalization_gap']:+.4f}")
                if progress_cb:
                    progress_cb(step, ev)

            if tok and cfg.train.sample_every and (step + 1) % cfg.train.sample_every == 0:
                for p in sample_prompts[:2]:
                    ids = torch.tensor([tok.encode_ordinary(p)], device=device)
                    g = model.generate(ids, 60, temperature=0.8, top_p=0.9,
                                       stop_ids=(tok.eot_id,))
                    tracker.sample(step, p, tok.decode(g[0].tolist()[len(ids[0]):]),
                                   dict(temperature=0.8, top_p=0.9))
                model.train()

    except KeyboardInterrupt:
        status = "cancelled"
        tracker.event("interrupted", "warn")

    elapsed = time.perf_counter() - t_start
    if cfg.train.save_checkpoints:
        torch.save(dict(model=model.state_dict(), config=cfg.to_dict(), step=cfg.train.steps,
                        val_loss=best_val, run_id=run_id, meta=meta),
                   os.path.join(run_dir, "last.pt"))
        tracker.artifact("checkpoint", os.path.join(run_dir, "best.pt"),
                         dict(val_loss=best_val, step=best_step))

    summary = dict(
        best_val_loss=best_val, best_val_ppl=float(math.exp(min(best_val, 20))),
        best_step=best_step, elapsed_s=round(elapsed, 1),
        tokens_seen=tokens_seen, tokens_per_s=tokens_seen / max(elapsed, 1e-9),
        **budget, **opt_info,
        param_breakdown=model.param_breakdown(),
        kv_cache_mb_1x512=model.kv_cache_bytes(1, 512) / 1e6,
        peak_mem_mb=device_memory_mb(device)["peak_mb"],
    )
    if hist:
        summary |= {k: hist[-1][k] for k in
                    ("val_ppl", "val_acc", "val_acc_top5", "val_bits_per_byte",
                     "generalization_gap", "val_logit_absmax", "val_confidence")
                    if k in hist[-1]}
    tracker.finish_run(status, summary)
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w"), indent=2, default=str)
    if own_tracker:
        tracker.close()
    if not quiet:
        print(f"[train] done in {elapsed/60:.1f} min | best val {best_val:.4f} "
              f"(ppl {summary['best_val_ppl']:.2f}) @ step {best_step}")
    return dict(run_id=run_id, run_dir=run_dir, status=status, **summary)


def main() -> None:
    ap = argparse.ArgumentParser("slm.train")
    ap.add_argument("--config", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="path=value",
                    help="dotted config overrides, e.g. --set model.n_layer=8 train.lr=1e-3")
    a = ap.parse_args()

    cfg = load_config(a.config) if a.config else RunConfig()
    for k, v in [("name", a.name), ("out_dir", a.out_dir)]:
        if v is not None:
            setattr(cfg, k, v)
    for k, v in [("steps", a.steps), ("batch_size", a.batch_size), ("seq_len", a.seq_len),
                 ("lr", a.lr), ("seed", a.seed)]:
        if v is not None:
            setattr(cfg.train, k, v)
    if a.corpus:
        cfg.data.corpus = a.corpus
    if a.set:
        from .config import apply_overrides
        ov = {}
        for item in a.set:
            key, _, val = item.partition("=")
            try:
                ov[key] = json.loads(val)
            except json.JSONDecodeError:
                ov[key] = val
        cfg = apply_overrides(cfg, ov)
    if cfg.model.max_seq_len < cfg.train.seq_len:
        cfg.model.max_seq_len = cfg.train.seq_len
    train(cfg)


if __name__ == "__main__":
    main()
