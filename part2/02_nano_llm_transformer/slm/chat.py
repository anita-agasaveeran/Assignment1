"""Chat templating and instruction-tuning data (CRISP-DM phase 3, chat track).

Template is ChatML, the format used by most open chat models:

    <|im_start|>system\n{system}<|im_end|>\n
    <|im_start|>user\n{user}<|im_end|>\n
    <|im_start|>assistant\n{reply}<|im_end|>\n

Two details matter and are easy to get wrong:

1. **Loss masking.** We train only on assistant tokens (plus the closing
   <|im_end|>). Training on the user's turn teaches the model to *write*
   questions, which shows up at chat time as the model interviewing itself.
   `build_sft_example` returns an explicit mask so this is auditable.
2. **Turn-boundary supervision.** The closing <|im_end|> is inside the loss, so
   the model learns to stop. Without it, generation runs until the token budget
   and the chatbot rambles into a new turn.

Instruction pairs come from TinyStories-Instruct (Eldan & Li 2023,
arXiv:2305.07759), which ships Features / Words / Summary fields alongside each
story. We invert those fields into natural user requests -- a template-based
instance of the Self-Instruct idea (arXiv:2212.10560) with no model in the loop.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time

import numpy as np

from .tokenizer import BPETokenizer

IM_START, IM_END = "<|im_start|>", "<|im_end|>"
DEFAULT_SYSTEM = "You are a friendly assistant who tells short, simple stories for children."

# The instruct corpus uses these field labels, in varying order per record.
FIELD_RE = re.compile(r"^(Features|Words|Summary|Story|Random sentence):", re.M)


def parse_instruct_record(block: str) -> dict[str, str] | None:
    parts = FIELD_RE.split(block.strip())
    if len(parts) < 3:
        return None
    rec: dict[str, str] = {}
    for i in range(1, len(parts) - 1, 2):
        rec[parts[i].strip().lower().replace(" ", "_")] = parts[i + 1].strip()
    if not rec.get("story"):
        return None
    return rec


PROMPT_TEMPLATES = [
    "Write a short story that uses the words: {words}.",
    "Can you tell me a little story with these words: {words}?",
    "Please write a story for a child using the words {words}.",
    "Tell me a story about this: {summary}",
    "Write a short story. Summary: {summary}",
    "Make up a story that includes {words}. It should be simple and easy to read.",
    "I want a story with the words {words}. The story should be about: {summary}",
]
FEATURE_HINTS = {
    "dialogue": "Include some dialogue.",
    "twist": "Give it a surprise ending.",
    "foreshadowing": "Hint at what will happen later.",
    "conflict": "Include a small problem the characters solve.",
    "moralvalue": "End with a gentle lesson.",
    "badending": "It does not need a happy ending.",
}


def record_to_conversation(rec: dict[str, str], rng: random.Random) -> dict | None:
    words = rec.get("words", "")
    summary = rec.get("summary", "")
    have_words, have_sum = bool(words), bool(summary)
    cands = [t for t in PROMPT_TEMPLATES
             if ("{words}" not in t or have_words) and ("{summary}" not in t or have_sum)]
    if not cands:
        return None
    prompt = rng.choice(cands).format(words=words, summary=summary)
    for f in re.split(r"[,\s]+", rec.get("features", "").lower()):
        hint = FEATURE_HINTS.get(f.strip())
        if hint and rng.random() < 0.7:
            prompt += " " + hint
            break
    return dict(messages=[dict(role="user", content=prompt),
                          dict(role="assistant", content=rec["story"].strip())])


# --------------------------------------------------------------- templating
def render_prompt(messages: list[dict], system: str | None = DEFAULT_SYSTEM,
                  add_generation_prompt: bool = True) -> str:
    out = ""
    if system:
        out += f"{IM_START}system\n{system}{IM_END}\n"
    for m in messages:
        out += f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n"
    if add_generation_prompt:
        out += f"{IM_START}assistant\n"
    return out


def build_sft_example(tok: BPETokenizer, messages: list[dict], system: str | None,
                      max_len: int) -> tuple[list[int], list[int]] | None:
    """Return (token_ids, loss_mask) with 1s only on assistant content + <|im_end|>."""
    ids: list[int] = []
    mask: list[int] = []
    im_start = tok.special_tokens[IM_START]
    im_end = tok.special_tokens[IM_END]
    nl = tok.encode_ordinary("\n")

    def add(chunk_ids: list[int], learn: int) -> None:
        ids.extend(chunk_ids)
        mask.extend([learn] * len(chunk_ids))

    if system:
        add([im_start], 0)
        add(tok.encode_ordinary(f"system\n{system}"), 0)
        add([im_end], 0); add(nl, 0)
    for m in messages:
        learn = int(m["role"] == "assistant")
        add([im_start], 0)
        add(tok.encode_ordinary(f"{m['role']}\n"), 0)
        add(tok.encode_ordinary(m["content"]), learn)
        add([im_end], learn)          # teach the model to stop
        add(nl, 0)
    if len(ids) > max_len or sum(mask) == 0:
        return None
    return ids, mask


# ------------------------------------------------------------- dataset build
def build_sft_dataset(args: argparse.Namespace) -> None:
    tok = BPETokenizer.load(args.tokenizer)
    rng = random.Random(args.seed)
    t0 = time.time()

    def load(path: str, limit: int) -> list[dict]:
        with open(path, encoding="utf-8", errors="replace") as f:
            raw = f.read()
        convos: list[dict] = []
        for block in raw.split("<|endoftext|>"):
            rec = parse_instruct_record(block)
            if not rec:
                continue
            c = record_to_conversation(rec, rng)
            if c:
                convos.append(c)
            if limit and len(convos) >= limit:
                break
        return convos

    train_c = load(args.input, args.max_examples)
    val_c = load(args.val, args.max_val_examples) if args.val else []
    print(f"[sft] parsed {len(train_c):,} train / {len(val_c):,} val conversations "
          f"in {time.time()-t0:.0f}s")

    stats = dict(n_train_convos=len(train_c), n_val_convos=len(val_c))

    def pack(convos: list[dict], out_path: str, tag: str) -> dict:
        toks, masks, dropped = [], [], 0
        lens, sup = [], []
        for c in convos:
            ex = build_sft_example(tok, c["messages"], DEFAULT_SYSTEM, args.max_len)
            if ex is None:
                dropped += 1
                continue
            ids, mk = ex
            pad = args.max_len - len(ids)
            lens.append(len(ids)); sup.append(sum(mk))
            toks.append(np.array(ids + [tok.special_tokens["<|pad|>"]] * pad, dtype=np.uint16))
            masks.append(np.array(mk + [0] * pad, dtype=np.uint8))
        T = np.stack(toks); M = np.stack(masks)
        np.save(out_path + ".tokens.npy", T)
        np.save(out_path + ".mask.npy", M)
        s = dict(n=len(T), dropped_too_long=dropped, max_len=args.max_len,
                 mean_len=round(float(np.mean(lens)), 1),
                 p95_len=int(np.percentile(lens, 95)),
                 mean_supervised_tokens=round(float(np.mean(sup)), 1),
                 supervised_fraction=round(float(np.sum(sup) / np.sum(lens)), 4),
                 padding_fraction=round(1 - float(np.sum(lens)) / T.size, 4))
        print(f"[sft] {tag}: {s['n']:,} examples, {s['supervised_fraction']:.1%} of tokens "
              f"supervised, {s['padding_fraction']:.1%} padding, dropped {dropped:,}")
        return s

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    stats["train"] = pack(train_c, f"{args.out}.train", "train")
    if val_c:
        stats["val"] = pack(val_c, f"{args.out}.val", "val")
    stats["examples"] = [render_prompt(c["messages"], DEFAULT_SYSTEM, False)[:1200]
                         for c in train_c[:3]]
    stats["template"] = "chatml"
    stats["system"] = DEFAULT_SYSTEM
    with open(f"{args.out}.profile.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[sft] wrote {args.out}.*.npy and profile")


def main() -> None:
    ap = argparse.ArgumentParser("slm.chat")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build-sft")
    p.add_argument("--input", default="data/raw/instruct_train.slice.txt")
    p.add_argument("--val", default="data/raw/instruct_valid.txt")
    p.add_argument("--out", default="data/processed/chat")
    p.add_argument("--tokenizer", default="data/processed/tokenizer.json")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--max-examples", type=int, default=40000)
    p.add_argument("--max-val-examples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=build_sft_dataset)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
