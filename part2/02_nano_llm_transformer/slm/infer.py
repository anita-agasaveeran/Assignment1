"""CRISP-DM phase 6: Deployment. Checkpoint loading + streaming generation.

`LM` is the single object the CLI, the evaluation suite and the dashboard's chat
endpoint all share, so what you measure offline is exactly what serves online.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Iterator

import torch
import torch.nn.functional as F

from .chat import DEFAULT_SYSTEM, IM_END, render_prompt
from .config import RunConfig
from .model import KVCache, SLM, build_model
from .tokenizer import BPETokenizer
from .utils import pick_device, sync


class LM:
    def __init__(self, ckpt_path: str, tokenizer_path: str | None = None,
                 device: str = "auto"):
        self.device = pick_device(device)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.cfg = RunConfig.from_dict(ckpt["config"])
        self.model: SLM = build_model(self.cfg.model)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()
        tp = tokenizer_path or self.cfg.data.tokenizer
        self.tok = BPETokenizer.load(tp)
        self.ckpt_path = ckpt_path
        self.step = ckpt.get("step")
        self.val_loss = ckpt.get("val_loss")
        self.run_id = ckpt.get("run_id")
        self.im_end = self.tok.special_tokens[IM_END]
        self.eot = self.tok.eot_id

    # ------------------------------------------------------------ streaming
    @torch.no_grad()
    def stream(self, prompt: str, max_new_tokens: int = 200, temperature: float = 0.8,
               top_k: int = 0, top_p: float = 0.9, repetition_penalty: float = 1.05,
               stop_ids: tuple[int, ...] | None = None,
               allow_special: bool = True) -> Iterator[dict]:
        """Yield {'token': str, 'id': int} per step, then a final {'done': stats}."""
        stop_ids = stop_ids if stop_ids is not None else (self.im_end, self.eot)
        ids = self.tok.encode(prompt, allowed_special=allow_special)
        max_ctx = self.cfg.model.max_seq_len
        if len(ids) >= max_ctx - 8:
            ids = ids[-(max_ctx - 8):]                       # keep the most recent context
        x = torch.tensor([ids], device=self.device)
        cache = [KVCache.alloc(self.cfg.model, 1, max_ctx, self.device,
                               next(self.model.parameters()).dtype)
                 for _ in range(self.cfg.model.n_layer)]

        t0 = time.perf_counter()
        out = self.model(x, cache=cache, start_pos=0)
        sync(self.device)
        ttft = time.perf_counter() - t0
        pos = x.size(1)
        logits = out["logits"][:, -1, :].float()
        seen = list(ids)
        emitted = 0
        buf = bytearray()
        t1 = time.perf_counter()

        while emitted < max_new_tokens and pos < max_ctx:
            if repetition_penalty != 1.0 and seen:
                uniq = torch.tensor(sorted(set(seen[-256:])), device=self.device)
                lg = logits[0, uniq]
                logits[0, uniq] = torch.where(lg > 0, lg / repetition_penalty,
                                              lg * repetition_penalty)
            if temperature <= 0:
                nxt = int(logits.argmax(-1))
            else:
                lg = logits / temperature
                if top_k:
                    kth = torch.topk(lg, min(top_k, lg.size(-1)), dim=-1).values[:, -1:]
                    lg = lg.masked_fill(lg < kth, float("-inf"))
                if 0 < top_p < 1.0:
                    srt, sidx = torch.sort(lg, descending=True, dim=-1)
                    p = F.softmax(srt, dim=-1)
                    drop = (p.cumsum(-1) - p) >= top_p
                    srt = srt.masked_fill(drop, float("-inf"))
                    lg = torch.full_like(lg, float("-inf")).scatter(-1, sidx, srt)
                nxt = int(torch.multinomial(F.softmax(lg, dim=-1), 1))

            if nxt in stop_ids:
                break
            seen.append(nxt)
            emitted += 1
            # decode incrementally; a BPE piece can split a UTF-8 character
            buf += self.tok.vocab.get(nxt, b"")
            try:
                text = buf.decode("utf-8")
                buf.clear()
            except UnicodeDecodeError:
                text = ""
            if text:
                yield dict(token=text, id=nxt)

            step_in = torch.tensor([[nxt]], device=self.device)
            out = self.model(step_in, cache=cache, start_pos=pos)
            pos += 1
            logits = out["logits"][:, -1, :].float()

        sync(self.device)
        total = time.perf_counter() - t1
        yield dict(done=True, stats=dict(
            prompt_tokens=len(ids), completion_tokens=emitted,
            ttft_ms=round(ttft * 1000, 1),
            decode_tok_per_s=round(emitted / max(total, 1e-9), 1),
            total_s=round(time.perf_counter() - t0, 3),
            kv_cache_mb=round(sum(c.bytes() for c in cache) / 1e6, 2)))

    def generate(self, prompt: str, **kw) -> str:
        return "".join(ev["token"] for ev in self.stream(prompt, **kw) if "token" in ev)

    def chat(self, messages: list[dict], system: str | None = DEFAULT_SYSTEM, **kw) -> str:
        return self.generate(render_prompt(messages, system), **kw)

    def chat_stream(self, messages: list[dict], system: str | None = DEFAULT_SYSTEM, **kw):
        return self.stream(render_prompt(messages, system), **kw)

    # ---------------------------------------------------------- diagnostics
    @torch.no_grad()
    def score(self, text: str) -> dict:
        """Token-level NLL of `text` under the model."""
        ids = self.tok.encode_ordinary(text)[: self.cfg.model.max_seq_len]
        if len(ids) < 2:
            return dict(nll=float("nan"), ppl=float("nan"), n_tokens=len(ids))
        x = torch.tensor([ids[:-1]], device=self.device)
        y = torch.tensor([ids[1:]], device=self.device)
        out = self.model(x, y)
        ce = float(out["ce"])
        import math
        return dict(nll=ce, ppl=math.exp(min(ce, 20)), n_tokens=len(ids))

    def benchmark(self, prompt: str = "Once upon a time", n_tokens: int = 128,
                  runs: int = 3) -> dict:
        res = []
        for _ in range(runs):
            for ev in self.stream(prompt, max_new_tokens=n_tokens, temperature=0.8):
                if ev.get("done"):
                    res.append(ev["stats"])
        return dict(
            runs=runs, n_tokens=n_tokens,
            ttft_ms=round(sum(r["ttft_ms"] for r in res) / len(res), 1),
            decode_tok_per_s=round(sum(r["decode_tok_per_s"] for r in res) / len(res), 1),
            kv_cache_mb=res[0]["kv_cache_mb"],
        )


def _run_phase(run_dir: str) -> str | None:
    """Phase of a run, from config.yaml, falling back to the checkpoint itself."""
    cp = os.path.join(run_dir, "config.yaml")
    if os.path.exists(cp):
        try:
            import yaml
            return (yaml.safe_load(open(cp)) or {}).get("phase")
        except Exception:
            pass
    bp = os.path.join(run_dir, "best.pt")
    try:
        return torch.load(bp, map_location="cpu", weights_only=False)["config"].get("phase")
    except Exception:
        return None


def latest_checkpoint(runs_dir: str = "runs", phase: str | None = None) -> str | None:
    """Newest best.pt, optionally restricted to runs of a given phase."""
    best = None
    for d in sorted(os.listdir(runs_dir)) if os.path.isdir(runs_dir) else []:
        p = os.path.join(runs_dir, d, "best.pt")
        if not os.path.exists(p):
            continue
        if phase and _run_phase(os.path.join(runs_dir, d)) != phase:
            continue
        if best is None or os.path.getmtime(p) > os.path.getmtime(best):
            best = p
    return best


def main() -> None:
    ap = argparse.ArgumentParser("slm.infer")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--phase", default=None, help="restrict auto-pick to sft/pretrain")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--chat", action="store_true", help="interactive chat REPL")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--repetition-penalty", type=float, default=1.05)
    a = ap.parse_args()

    ckpt = a.ckpt or latest_checkpoint(phase=a.phase) or latest_checkpoint()
    if not ckpt:
        raise SystemExit("no checkpoint found; train one first")
    lm = LM(ckpt)
    print(f"[infer] {ckpt} | step {lm.step} | val_loss {lm.val_loss:.4f} | {lm.device}")

    if a.bench:
        print(json.dumps(lm.benchmark(), indent=2)); return

    kw = dict(max_new_tokens=a.max_new_tokens, temperature=a.temperature,
              top_p=a.top_p, top_k=a.top_k, repetition_penalty=a.repetition_penalty)
    if not a.chat:
        for ev in lm.stream(a.prompt, **kw):
            if "token" in ev:
                print(ev["token"], end="", flush=True)
            else:
                print(f"\n\n[{ev['stats']['decode_tok_per_s']} tok/s, "
                      f"TTFT {ev['stats']['ttft_ms']}ms]")
        return

    print("chat mode - blank line or 'exit' to quit, '/reset' to clear history\n")
    history: list[dict] = []
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user or user == "exit":
            break
        if user == "/reset":
            history.clear(); print("(history cleared)\n"); continue
        history.append(dict(role="user", content=user))
        print("bot> ", end="", flush=True)
        reply = ""
        for ev in lm.chat_stream(history, **kw):
            if "token" in ev:
                print(ev["token"], end="", flush=True); reply += ev["token"]
            else:
                print(f"\n   [{ev['stats']['decode_tok_per_s']} tok/s]\n")
        history.append(dict(role="assistant", content=reply.strip()))
        history = history[-8:]


if __name__ == "__main__":
    main()
