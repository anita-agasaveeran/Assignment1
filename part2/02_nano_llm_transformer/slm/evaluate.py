"""CRISP-DM phase 5: Evaluation.

Held-out loss alone is not an evaluation. This module reports:

  quality      cross-entropy, perplexity, bits-per-byte, top-1/top-5 accuracy
  calibration  expected calibration error + reliability bins -- a model can be
               accurate and badly overconfident, which matters the moment you
               sample from it (arXiv:1512.00567 motivates the counter-measure)
  structure    loss vs. position in context (is long context actually used?)
               loss vs. token frequency decile (rare-token competence, which the
               aggregate number hides completely)
  generation   distinct-n, repeated-4gram rate, stop-rate, mean length
  decoding     a sweep over greedy / top-k / top-p / temperature that reproduces
               the degeneration result of Holtzman et al. 2019 (arXiv:1904.09751)
  task         instruction compliance: did the story actually contain the words
               the user asked for? (only meaningful for SFT checkpoints)

Everything lands in runs/<run_id>/eval.json and in the tracker, so the dashboard
renders measured numbers rather than re-deriving them.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from .chat import DEFAULT_SYSTEM, render_prompt
from .infer import LM, latest_checkpoint
from .tokenizer import BPETokenizer
from .tracking import Tracker
from .utils import pick_device, sync

EVAL_SEED = 20240601


# ------------------------------------------------------------ core metrics
@torch.no_grad()
def teacher_forced_metrics(lm: LM, bin_path: str, n_batches: int = 40,
                           batch_size: int = 16, seq_len: int | None = None,
                           unigram_counts: np.ndarray | None = None,
                           bytes_per_token: float = 0.0) -> dict:
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")
    T = seq_len or lm.cfg.model.max_seq_len
    rng = np.random.default_rng(EVAL_SEED)
    dev = lm.device
    ac = torch.autocast(device_type=dev.type, dtype=torch.bfloat16) if dev.type in ("mps", "cuda") \
        else torch.autocast("cpu", enabled=False)

    tot_ce = 0.0; tot_n = 0; c1 = 0; c5 = 0
    pos_loss = torch.zeros(T, device=dev)
    conf_bins = np.zeros(10); acc_bins = np.zeros(10); cnt_bins = np.zeros(10)
    freq_loss = np.zeros(10); freq_cnt = np.zeros(10)
    logZ = 0.0; logit_absmax = 0.0
    ent_sum = 0.0

    decile = None
    if unigram_counts is not None:
        order = np.argsort(-unigram_counts)                 # most frequent first
        rank = np.empty_like(order); rank[order] = np.arange(len(order))
        # decile by cumulative token mass, not by vocab index
        cum = np.cumsum(unigram_counts[order]) / max(unigram_counts.sum(), 1)
        dec_of_rank = np.minimum((cum * 10).astype(int), 9)
        decile = np.empty(len(order), dtype=np.int64)
        decile[order] = dec_of_rank

    for _ in range(n_batches):
        ix = rng.integers(0, len(data) - T - 1, size=batch_size)
        x = torch.from_numpy(np.stack([data[i:i + T] for i in ix]).astype(np.int64)).to(dev)
        y = torch.from_numpy(np.stack([data[i + 1:i + 1 + T] for i in ix]).astype(np.int64)).to(dev)
        with ac:
            out = lm.model(x)
        logits = out["logits"].float()
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1), reduction="none")
        tot_ce += float(ce.sum()); tot_n += y.numel()
        pos_loss += ce.view(y.shape).mean(0)

        probs = F.softmax(logits, -1)
        top = probs.topk(5, -1)
        pred = top.indices[..., 0]
        c1 += int((pred == y).sum())
        c5 += int((top.indices == y.unsqueeze(-1)).any(-1).sum())
        ent_sum += float(-(probs * torch.log(probs.clamp_min(1e-9))).sum(-1).sum())
        logZ += float(torch.logsumexp(logits, -1).sum())
        logit_absmax = max(logit_absmax, float(logits.abs().max()))

        conf = top.values[..., 0].reshape(-1).cpu().numpy()
        corr = (pred == y).reshape(-1).cpu().numpy().astype(float)
        b = np.minimum((conf * 10).astype(int), 9)
        np.add.at(conf_bins, b, conf); np.add.at(acc_bins, b, corr); np.add.at(cnt_bins, b, 1)

        if decile is not None:
            yd = decile[y.reshape(-1).cpu().numpy()]
            cen = ce.detach().cpu().numpy()
            np.add.at(freq_loss, yd, cen); np.add.at(freq_cnt, yd, 1)

    ce_mean = tot_ce / tot_n
    nz = cnt_bins > 0
    ece = float(np.sum(cnt_bins[nz] / cnt_bins.sum() *
                       np.abs(acc_bins[nz] / cnt_bins[nz] - conf_bins[nz] / cnt_bins[nz])))
    res = dict(
        loss=ce_mean, ppl=math.exp(min(ce_mean, 20)),
        acc_top1=c1 / tot_n, acc_top5=c5 / tot_n,
        entropy_nats=ent_sum / tot_n, logZ=logZ / tot_n, logit_absmax=logit_absmax,
        ece=ece, n_tokens_scored=tot_n,
        reliability=[dict(bin=round(i / 10 + 0.05, 2),
                          confidence=float(conf_bins[i] / cnt_bins[i]) if cnt_bins[i] else None,
                          accuracy=float(acc_bins[i] / cnt_bins[i]) if cnt_bins[i] else None,
                          count=int(cnt_bins[i])) for i in range(10)],
        loss_by_position=[round(v, 4) for v in (pos_loss / n_batches).cpu().tolist()],
    )
    if bytes_per_token > 0:
        res["bits_per_byte"] = ce_mean / math.log(2) / bytes_per_token
    if decile is not None:
        res["loss_by_freq_decile"] = [
            dict(decile=i + 1, loss=round(float(freq_loss[i] / freq_cnt[i]), 4),
                 n=int(freq_cnt[i])) for i in range(10) if freq_cnt[i] > 0]
    return res


def unigram_counts_from_bin(bin_path: str, vocab_size: int, limit: int = 20_000_000) -> np.ndarray:
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")
    sl = np.asarray(data[:limit], dtype=np.int64)
    return np.bincount(sl, minlength=vocab_size).astype(np.float64)


# ---------------------------------------------------------- generation eval
def _ngrams(toks: list, n: int) -> list[tuple]:
    return [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]


def generation_metrics(texts: list[str]) -> dict:
    words = [t.split() for t in texts]
    allw = [w for ws in words for w in ws]
    d = {}
    for n in (1, 2, 3):
        grams = [g for ws in words for g in _ngrams(ws, n)]
        d[f"distinct_{n}"] = round(len(set(grams)) / max(len(grams), 1), 4)
    rep = 0; tot = 0
    for ws in words:
        g = _ngrams(ws, 4)
        if not g:
            continue
        tot += len(g)
        rep += len(g) - len(set(g))
    return d | dict(
        repeated_4gram_rate=round(rep / max(tot, 1), 4),
        mean_words=round(float(np.mean([len(w) for w in words])) if words else 0.0, 1),
        type_token_ratio=round(len(set(allw)) / max(len(allw), 1), 4),
    )


def decoding_study(lm: LM, prompts: list[str], max_new_tokens: int = 120,
                   seed: int = 0) -> list[dict]:
    """Reproduce the degeneration result: greedy repeats, nucleus does not."""
    settings = [
        dict(name="greedy", temperature=0.0, top_k=0, top_p=1.0, repetition_penalty=1.0),
        dict(name="temp=0.5", temperature=0.5, top_k=0, top_p=1.0, repetition_penalty=1.0),
        dict(name="temp=1.0", temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0),
        dict(name="top-k=40", temperature=1.0, top_k=40, top_p=1.0, repetition_penalty=1.0),
        dict(name="top-p=0.9", temperature=1.0, top_k=0, top_p=0.9, repetition_penalty=1.0),
        dict(name="top-p=0.9 t=0.8", temperature=0.8, top_k=0, top_p=0.9, repetition_penalty=1.0),
        dict(name="top-p=0.9 t=0.8 rp=1.1", temperature=0.8, top_k=0, top_p=0.9,
             repetition_penalty=1.1),
    ]
    out = []
    for s in settings:
        torch.manual_seed(seed)
        texts, stops, tps = [], 0, []
        t0 = time.perf_counter()
        for p in prompts:
            txt = ""
            for ev in lm.stream(p, max_new_tokens=max_new_tokens,
                                **{k: v for k, v in s.items() if k != "name"}):
                if "token" in ev:
                    txt += ev["token"]
                else:
                    tps.append(ev["stats"]["decode_tok_per_s"])
                    if ev["stats"]["completion_tokens"] < max_new_tokens:
                        stops += 1
            texts.append(txt)
        gm = generation_metrics(texts)
        out.append(dict(**s, **gm, stop_rate=round(stops / len(prompts), 3),
                        tok_per_s=round(float(np.mean(tps)), 1),
                        seconds=round(time.perf_counter() - t0, 1),
                        sample=texts[0][:400]))
    return out


# --------------------------------------------------------------- task evals
WORDS_RE = re.compile(r"words[:\s]+([a-z,\s]+?)(?:\.|$)", re.I)


def instruction_compliance(lm: LM, instruct_path: str, n: int = 60, seed: int = 0,
                           max_new_tokens: int = 220) -> dict:
    """Did the model use the words the user asked for? (SFT checkpoints only)"""
    from .chat import parse_instruct_record, record_to_conversation
    rng = random.Random(seed)
    with open(instruct_path, encoding="utf-8", errors="replace") as f:
        raw = f.read(6_000_000)
    recs = []
    for block in raw.split("<|endoftext|>"):
        r = parse_instruct_record(block)
        if r and r.get("words"):
            recs.append(r)
        if len(recs) >= n * 3:
            break
    rng.shuffle(recs)
    recs = recs[:n]

    hits = tot = 0
    per_example, texts = [], []
    for r in recs:
        conv = record_to_conversation(r, rng)
        if not conv:
            continue
        prompt = render_prompt([conv["messages"][0]], DEFAULT_SYSTEM)
        torch.manual_seed(seed)
        reply = lm.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.8, top_p=0.9,
                            repetition_penalty=1.05)
        texts.append(reply)
        want = [w.strip().lower() for w in r["words"].split(",") if w.strip()]
        got = sum(1 for w in want if w in reply.lower())
        hits += got; tot += len(want)
        per_example.append(dict(words=want, found=got, n=len(want),
                                reply=reply[:300], prompt=conv["messages"][0]["content"][:200]))
    return dict(n_examples=len(per_example), word_inclusion_rate=round(hits / max(tot, 1), 4),
                fully_compliant_rate=round(
                    sum(1 for e in per_example if e["found"] == e["n"]) / max(len(per_example), 1), 4),
                **generation_metrics(texts), examples=per_example[:8])


# ------------------------------------------------------------------ driver
def full_eval(args: argparse.Namespace) -> dict:
    ckpt = args.ckpt or latest_checkpoint(args.runs_dir, args.phase) or latest_checkpoint(args.runs_dir)
    if not ckpt:
        raise SystemExit("no checkpoint found")
    lm = LM(ckpt)
    run_dir = os.path.dirname(ckpt)
    meta_path = f"{lm.cfg.data.corpus}.meta.json"
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    bpt = float(meta.get("bytes_per_token_val", 0.0))

    print(f"[eval] {ckpt}  (step {lm.step}, val_loss {lm.val_loss:.4f})")
    t0 = time.perf_counter()
    report: dict = dict(ckpt=ckpt, run_id=lm.run_id, step=lm.step,
                        config=lm.cfg.to_dict(), created_at=time.time())

    uni = None
    train_bin = f"{lm.cfg.data.corpus}.train.bin"
    if os.path.exists(train_bin):
        uni = unigram_counts_from_bin(train_bin, lm.cfg.model.vocab_size)

    val_bin = f"{lm.cfg.data.corpus}.val.bin"
    print("[eval] teacher-forced metrics")
    report["heldout"] = teacher_forced_metrics(
        lm, val_bin, n_batches=args.n_batches, batch_size=args.batch_size,
        seq_len=lm.cfg.model.max_seq_len, unigram_counts=uni, bytes_per_token=bpt)
    print(f"       loss {report['heldout']['loss']:.4f}  ppl {report['heldout']['ppl']:.2f}  "
          f"acc {report['heldout']['acc_top1']:.3f}  ECE {report['heldout']['ece']:.4f}")

    print("[eval] decoding study")
    prompts = args.prompts or ["Once upon a time", "Lily and Tom were playing",
                               "The little dog saw a", "One day, a girl named Sara",
                               "Tim was very happy because"]
    report["decoding"] = decoding_study(lm, prompts, args.gen_tokens)
    for d in report["decoding"]:
        print(f"       {d['name']:24s} rep4 {d['repeated_4gram_rate']:.3f}  "
              f"distinct2 {d['distinct_2']:.3f}  stop {d['stop_rate']:.2f}")

    print("[eval] inference benchmark")
    report["latency"] = lm.benchmark(n_tokens=args.gen_tokens, runs=3)
    report["latency"]["params_total"] = sum(p.numel() for p in lm.model.parameters())
    report["latency"]["kv_cache_mb_full_ctx"] = lm.model.kv_cache_bytes(
        1, lm.cfg.model.max_seq_len) / 1e6

    if args.instruct and os.path.exists(args.instruct):
        print("[eval] instruction compliance")
        report["instruction"] = instruction_compliance(lm, args.instruct, args.n_instruct)
        print(f"       word inclusion {report['instruction']['word_inclusion_rate']:.3f}  "
              f"fully compliant {report['instruction']['fully_compliant_rate']:.3f}")

    report["seconds"] = round(time.perf_counter() - t0, 1)
    out = os.path.join(run_dir, "eval.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)

    if lm.run_id:
        tr = Tracker(os.path.join(args.runs_dir, "tracking.db"))
        tr.run_id = lm.run_id
        flat = {f"eval_{k}": v for k, v in report["heldout"].items() if isinstance(v, (int, float))}
        flat |= {f"lat_{k}": v for k, v in report["latency"].items() if isinstance(v, (int, float))}
        if "instruction" in report:
            flat["eval_word_inclusion"] = report["instruction"]["word_inclusion_rate"]
        tr.log(lm.step or 0, flat, flush=True)
        tr.event("evaluation complete", data=dict(path=out))
        tr.artifact("eval", out, dict(loss=report["heldout"]["loss"]))
        tr.close()
    print(f"[eval] wrote {out} ({report['seconds']}s)")
    return report


def main() -> None:
    ap = argparse.ArgumentParser("slm.evaluate")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--phase", default=None)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--n-batches", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--gen-tokens", type=int, default=120)
    ap.add_argument("--prompts", nargs="*", default=None)
    ap.add_argument("--instruct", default=None,
                    help="path to instruct corpus for compliance eval (SFT models)")
    ap.add_argument("--n-instruct", type=int, default=60)
    a = ap.parse_args()
    full_eval(a)


if __name__ == "__main__":
    main()
