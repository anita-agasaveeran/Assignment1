"""Supervised fine-tuning: turn the pretrained story model into a chatbot.

This is the smallest useful slice of the InstructGPT pipeline (Ouyang et al.
2022, arXiv:2203.02155) -- stage 1 only, no reward model and no PPO. At this
scale SFT is where essentially all of the chat behaviour comes from.

The one thing that must be right is the loss mask: gradients flow only through
assistant tokens and the <|im_end|> that terminates them. `slm/chat.py` builds
that mask; this module asserts it is non-trivial before training, because a
silently all-ones mask produces a model that fluently writes both sides of the
conversation and is easy to miss until you are already chatting with it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from .chat import DEFAULT_SYSTEM, render_prompt
from .config import RunConfig, save_config
from .model import build_model
from .optim import build_optimizer, clip_grad_norm, lr_scale
from .tokenizer import BPETokenizer
from .tracking import Tracker
from .train import git_state
from .utils import device_memory_mb, env_info, measure_peak_flops, pick_device, set_seed, sync


class ChatDataset:
    def __init__(self, prefix: str, split: str):
        self.tokens = np.load(f"{prefix}.{split}.tokens.npy")     # [N, L] uint16
        self.mask = np.load(f"{prefix}.{split}.mask.npy")         # [N, L] uint8
        assert self.tokens.shape == self.mask.shape
        self.n, self.L = self.tokens.shape

    def batch(self, bs: int, rng: np.random.Generator, device):
        ix = rng.integers(0, self.n, size=bs)
        t = self.tokens[ix].astype(np.int64)
        m = self.mask[ix].astype(np.float32)
        x = torch.from_numpy(t[:, :-1]).to(device)
        y = torch.from_numpy(t[:, 1:]).to(device)
        # a target at position i is supervised iff mask[i+1] is set
        lm = torch.from_numpy(m[:, 1:]).to(device)
        return x, y, lm

    def fixed_batches(self, bs: int, n: int, device, seed: int = 20240601):
        rng = np.random.default_rng(seed)
        return [self.batch(bs, rng, device) for _ in range(n)]


@torch.no_grad()
def eval_sft(model, batches, ac) -> dict:
    model.eval()
    tot, n = 0.0, 0.0
    correct = 0.0
    for x, y, m in batches:
        with ac:
            out = model(x, y, loss_mask=m)
        logits = out["logits"].float()
        ce = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.reshape(-1), reduction="none").view(y.shape)
        tot += float((ce * m).sum()); n += float(m.sum())
        correct += float(((logits.argmax(-1) == y).float() * m).sum())
    model.train()
    loss = tot / max(n, 1)
    return dict(val_loss=loss, val_ppl=math.exp(min(loss, 20)),
                val_acc=correct / max(n, 1), val_supervised_tokens=int(n))


def run_sft(args: argparse.Namespace) -> dict:
    device = pick_device(args.device)
    ckpt = torch.load(args.base_ckpt, map_location="cpu", weights_only=False)
    cfg = RunConfig.from_dict(ckpt["config"])
    cfg.name = args.name
    cfg.phase = "sft"
    cfg.notes = f"SFT from {args.base_ckpt}"
    cfg.train.steps = args.steps
    cfg.train.lr = args.lr
    cfg.train.batch_size = args.batch_size
    cfg.train.warmup_steps = max(10, args.steps // 20)
    cfg.train.schedule = "cosine"
    cfg.train.min_lr_ratio = 0.05
    cfg.train.seed = args.seed
    set_seed(args.seed)

    train_ds = ChatDataset(args.data, "train")
    val_ds = ChatDataset(args.data, "val")
    if train_ds.L - 1 > cfg.model.max_seq_len:
        raise SystemExit(f"chat seq len {train_ds.L} exceeds model context {cfg.model.max_seq_len}")

    frac = float(train_ds.mask.mean())
    if not 0.05 < frac < 0.95:
        raise SystemExit(f"loss mask covers {frac:.1%} of tokens - refusing to train; "
                         "expected roughly the assistant share of the conversation")
    print(f"[sft] mask sanity: {frac:.1%} of packed positions supervised")

    model = build_model(cfg.model).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"[sft] loaded base step {ckpt.get('step')} val_loss {ckpt.get('val_loss'):.4f}")

    opt, opt_info = build_optimizer(model, cfg.train)
    use_amp = device.type in ("cuda", "mps")
    ac = (torch.autocast(device_type=device.type, dtype=torch.bfloat16) if use_amp
          else torch.autocast("cpu", enabled=False))

    env = env_info(device) | dict(git=git_state(), base_ckpt=args.base_ckpt,
                                 peak_flops_measured=measure_peak_flops(device))
    tracker = Tracker(os.path.join(args.out_dir, "tracking.db"))
    run_id = tracker.start_run(cfg.name, "sft", cfg.to_dict(), env, ["sft", "chat"],
                               cfg.notes, cfg.fingerprint(),
                               parent_run=ckpt.get("run_id"))
    n_total = sum(p.numel() for p in model.parameters())
    tracker.set_params(n_total, n_total - model.num_params(non_embedding=True))
    run_dir = os.path.join(args.out_dir, run_id); os.makedirs(run_dir, exist_ok=True)
    save_config(cfg, os.path.join(run_dir, "config.yaml"))   # phase lookup depends on this

    profile_path = f"{args.data}.profile.json"
    if os.path.exists(profile_path):
        tracker.event("sft_data_profile", data=json.load(open(profile_path)))

    val_batches = val_ds.fixed_batches(args.batch_size, args.eval_iters, device)
    rng = np.random.default_rng(args.seed)
    tok = BPETokenizer.load(cfg.data.tokenizer)
    demo = ["Write a short story that uses the words: cat, hat, run.",
            "Tell me a story about a lost balloon."]

    best = float("inf"); best_step = 0
    t0 = time.perf_counter()
    tokens_per_step = args.batch_size * (train_ds.L - 1)
    model.train()

    for step in range(args.steps):
        ts = time.perf_counter()
        opt.set_lr_scale(lr_scale(step, cfg.train))
        x, y, m = train_ds.batch(args.batch_size, rng, device)
        with ac:
            out = model(x, y, loss_mask=m)
        out["loss"].backward()
        gn = clip_grad_norm(model, cfg.train.grad_clip)
        opt.step(); opt.zero_grad(set_to_none=True)
        sync(device)
        dt = time.perf_counter() - ts

        if step % args.log_every == 0:
            tracker.log(step, dict(train_loss=out["loss"].detach().item(),
                                   lr=opt.current_lrs()["adamw"], grad_norm=gn,
                                   step_time_s=dt, tokens_per_s=tokens_per_step / dt,
                                   supervised_frac=float(m.mean()),
                                   **device_memory_mb(device)))
        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            ev = eval_sft(model, val_batches, ac)
            ev["elapsed_s"] = time.perf_counter() - t0
            tracker.log(step, ev, flush=True)
            print(f"  [sft] step {step:5d} train {out['loss'].item():.4f} "
                  f"val {ev['val_loss']:.4f} ppl {ev['val_ppl']:.2f} acc {ev['val_acc']:.3f}")
            if ev["val_loss"] < best:
                best, best_step = ev["val_loss"], step
                torch.save(dict(model=model.state_dict(), config=cfg.to_dict(), step=step,
                                val_loss=best, run_id=run_id, meta=ckpt.get("meta", {})),
                           os.path.join(run_dir, "best.pt"))
        if args.sample_every and (step + 1) % args.sample_every == 0:
            for d in demo:
                p = render_prompt([dict(role="user", content=d)], DEFAULT_SYSTEM)
                ids = torch.tensor([tok.encode(p)], device=device)
                g = model.generate(ids, 110, temperature=0.8, top_p=0.9,
                                   stop_ids=(tok.special_tokens["<|im_end|>"], tok.eot_id))
                tracker.sample(step, d, tok.decode(g[0].tolist()[len(ids[0]):]),
                               dict(temperature=0.8, top_p=0.9))
            model.train()

    elapsed = time.perf_counter() - t0
    torch.save(dict(model=model.state_dict(), config=cfg.to_dict(), step=args.steps,
                    val_loss=best, run_id=run_id, meta=ckpt.get("meta", {})),
               os.path.join(run_dir, "last.pt"))
    summary = dict(best_val_loss=best, best_val_ppl=math.exp(min(best, 20)), best_step=best_step,
                   elapsed_s=round(elapsed, 1), steps=args.steps, base_ckpt=args.base_ckpt,
                   n_train_examples=train_ds.n, n_val_examples=val_ds.n,
                   supervised_fraction=round(frac, 4), n_params_total=n_total,
                   peak_mem_mb=device_memory_mb(device)["peak_mb"], **opt_info)
    tracker.artifact("checkpoint", os.path.join(run_dir, "best.pt"), dict(val_loss=best))
    tracker.finish_run("finished", summary)
    json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w"), indent=2, default=str)
    tracker.close()
    print(f"[sft] done in {elapsed/60:.1f} min | best val {best:.4f} -> {run_dir}/best.pt")
    return dict(run_id=run_id, run_dir=run_dir, **summary)


def main() -> None:
    ap = argparse.ArgumentParser("slm.sft")
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--data", default="data/processed/chat")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--name", default="chat-sft")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-iters", type=int, default=20)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--sample-every", type=int, default=500)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    run_sft(a)


if __name__ == "__main__":
    main()
