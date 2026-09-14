"""CRISP-DM phases 2 & 3: Data Understanding and Data Preparation.

Produces, from raw text:
  <prefix>.train.bin / .val.bin   uint16 token streams (memory-mapped at train time)
  <prefix>.meta.json              shapes, split sizes, tokenizer pointer
  <prefix>.profile.json           the full data profile the dashboard renders

The profile is not decoration. Three things here materially change the modeling
phase and are usually skipped in toy LM projects:

  1. Exact + near-duplicate removal (MinHash/LSH, Broder-style banding).
     Duplicated documents inflate held-out performance because the model has
     memorized, not generalized.
  2. Train/val leakage audit. Val documents that near-duplicate a train document
     are removed *before* any number is reported, so val loss means what it says.
  3. Compression + budget accounting. bytes/token, total tokens, and the
     Chinchilla-optimal token count for the chosen parameter count, so the
     compute-optimality gap is stated up front instead of discovered later.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter

import numpy as np

from .tokenizer import BPETokenizer

DOC_SEP = "<|endoftext|>"
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


# --------------------------------------------------------------------------- io
def read_documents(path: str, sep: str = DOC_SEP, min_chars: int = 1) -> list[str]:
    with open(path, encoding="utf-8", errors="replace") as f:
        raw = f.read()
    docs = [d.strip() for d in raw.split(sep)]
    return [d for d in docs if len(d) >= min_chars]


def normalize_for_hash(text: str) -> str:
    return _WS.sub(" ", _PUNCT.sub(" ", text.lower())).strip()


# ------------------------------------------------------------------ dedup: exact
def exact_dedup(docs: list[str]) -> tuple[list[int], dict]:
    seen: dict[str, int] = {}
    keep: list[int] = []
    dupes = 0
    for i, d in enumerate(docs):
        h = hashlib.sha1(normalize_for_hash(d).encode()).hexdigest()
        if h in seen:
            dupes += 1
            continue
        seen[h] = i
        keep.append(i)
    return keep, dict(n_in=len(docs), n_exact_dupes=dupes, n_out=len(keep))


# ------------------------------------------------- dedup: near (MinHash + LSH)
_MERSENNE = (1 << 61) - 1


def minhash_signatures(docs: list[str], n_perms: int = 64, shingle: int = 5,
                       seed: int = 0) -> np.ndarray:
    """Broder MinHash signatures over word-level k-shingles.

    Returns int64 [n_docs, n_perms]. Empty/short docs get a max-value signature
    so they never collide with anything.
    """
    rng = np.random.default_rng(seed)
    a = rng.integers(1, _MERSENNE, size=n_perms, dtype=np.int64)
    b = rng.integers(0, _MERSENNE, size=n_perms, dtype=np.int64)
    sigs = np.full((len(docs), n_perms), np.iinfo(np.int64).max, dtype=np.int64)

    for i, d in enumerate(docs):
        words = normalize_for_hash(d).split()
        if len(words) < shingle:
            if not words:
                continue
            shingles = {" ".join(words)}
        else:
            shingles = {" ".join(words[j:j + shingle]) for j in range(len(words) - shingle + 1)}
        h = np.fromiter(
            (int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big") & _MERSENNE
             for s in shingles), dtype=np.int64, count=len(shingles))
        # (a*h + b) mod p  -- vectorized over permutations
        perm = (np.multiply.outer(h, a) + b) % _MERSENNE
        sigs[i] = perm.min(axis=0)
    return sigs


class _UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.p[max(rx, ry)] = min(rx, ry)


def near_dedup(docs: list[str], sigs: np.ndarray, bands: int = 16,
               threshold: float = 0.8) -> tuple[list[int], dict]:
    """LSH-band candidate generation, then exact signature-Jaccard verification."""
    n, perms = sigs.shape
    rows = perms // bands
    uf = _UnionFind(n)
    n_candidates = 0
    for band in range(bands):
        buckets: dict[bytes, list[int]] = {}
        chunk = sigs[:, band * rows:(band + 1) * rows]
        for i in range(n):
            buckets.setdefault(chunk[i].tobytes(), []).append(i)
        for members in buckets.values():
            if len(members) < 2:
                continue
            anchor = members[0]
            for m in members[1:]:
                n_candidates += 1
                if uf.find(anchor) == uf.find(m):
                    continue
                if (sigs[anchor] == sigs[m]).mean() >= threshold:
                    uf.union(anchor, m)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(uf.find(i), []).append(i)
    keep = sorted(min(v) for v in clusters.values())
    sizes = sorted((len(v) for v in clusters.values() if len(v) > 1), reverse=True)
    return keep, dict(n_in=n, n_out=len(keep), n_removed=n - len(keep),
                      n_lsh_candidates=n_candidates, n_clusters=len(sizes),
                      largest_cluster=sizes[0] if sizes else 0,
                      bands=bands, rows_per_band=rows, threshold=threshold)


def leakage_audit(train_docs: list[str], val_docs: list[str], n_perms: int = 64,
                  shingle: int = 5, bands: int = 16, threshold: float = 0.8) -> tuple[list[int], dict]:
    """Return val indices that are safe to keep (no near-duplicate in train)."""
    t0 = time.time()
    tr = minhash_signatures(train_docs, n_perms, shingle)
    va = minhash_signatures(val_docs, n_perms, shingle)
    rows = n_perms // bands
    train_buckets: list[dict[bytes, list[int]]] = []
    for band in range(bands):
        d: dict[bytes, list[int]] = {}
        chunk = tr[:, band * rows:(band + 1) * rows]
        for i in range(len(train_docs)):
            d.setdefault(chunk[i].tobytes(), []).append(i)
        train_buckets.append(d)

    leaked: list[int] = []
    exact_train = {hashlib.sha1(normalize_for_hash(d).encode()).hexdigest() for d in train_docs}
    n_exact = 0
    for i, d in enumerate(val_docs):
        if hashlib.sha1(normalize_for_hash(d).encode()).hexdigest() in exact_train:
            leaked.append(i); n_exact += 1; continue
        hit = False
        for band in range(bands):
            key = va[i, band * rows:(band + 1) * rows].tobytes()
            for j in train_buckets[band].get(key, ()):
                if (va[i] == tr[j]).mean() >= threshold:
                    hit = True; break
            if hit:
                break
        if hit:
            leaked.append(i)
    keep = [i for i in range(len(val_docs)) if i not in set(leaked)]
    return keep, dict(n_val_in=len(val_docs), n_leaked=len(leaked), n_leaked_exact=n_exact,
                      n_leaked_near=len(leaked) - n_exact, n_val_out=len(keep),
                      leak_rate=round(len(leaked) / max(len(val_docs), 1), 5),
                      seconds=round(time.time() - t0, 1))


# ------------------------------------------------- detector positive control
def validate_dedup_detectors(docs: list[str], n_exact: int = 50, n_near: int = 30,
                             n_leak: int = 25, perms: int = 64, bands: int = 16,
                             shingle: int = 5, threshold: float = 0.8) -> dict:
    """Plant known duplicates in a clean sample and measure detector recall.

    A corpus can legitimately contain zero duplicates. Reporting "0 removed"
    without a positive control is indistinguishable from reporting "the detector
    is broken", so we always run this and publish the recall alongside the
    finding.
    """
    base = [d for d in docs[:2000] if len(d) > 200][:1200]
    if len(base) < max(n_exact, n_near, n_leak) * 2:
        return dict(ran=False, reason="not enough long documents to run the control")

    planted = list(base)
    planted += base[:n_exact]                                       # exact copies
    planted += [d[:-25] + " and then it was over." for d in base[:n_near]]   # near copies

    keep, ex = exact_dedup(planted)
    exact_recall = ex["n_exact_dupes"] / n_exact

    after_exact = [planted[i] for i in keep]
    sigs = minhash_signatures(after_exact, perms, shingle)
    _, nd = near_dedup(after_exact, sigs, bands, threshold)
    near_recall = min(nd["n_removed"] / n_near, 1.0)

    train_side = base[:600]
    val_side = base[600:900] + train_side[:n_leak]                  # planted leaks
    _, lk = leakage_audit(train_side, val_side, perms, shingle, bands, threshold)
    leak_recall = lk["n_leaked"] / n_leak
    # false positives: docs we removed that we did not plant
    near_fp = max(0, nd["n_removed"] - n_near)
    leak_fp = max(0, lk["n_leaked"] - n_leak)

    return dict(ran=True, n_control_docs=len(base),
                exact_planted=n_exact, exact_detected=ex["n_exact_dupes"],
                exact_recall=round(exact_recall, 3),
                near_planted=n_near, near_detected=nd["n_removed"],
                near_recall=round(near_recall, 3), near_false_positives=near_fp,
                leak_planted=n_leak, leak_detected=lk["n_leaked"],
                leak_recall=round(leak_recall, 3), leak_false_positives=leak_fp)


# ------------------------------------------------------------------- profiling
def _hist(values: np.ndarray, bins: int = 40, clip_pct: float = 99.5) -> dict:
    if len(values) == 0:
        return dict(edges=[], counts=[])
    hi = float(np.percentile(values, clip_pct))
    lo = float(values.min())
    if hi <= lo:
        hi = lo + 1
    counts, edges = np.histogram(np.clip(values, lo, hi), bins=bins, range=(lo, hi))
    return dict(edges=[round(float(e), 2) for e in edges],
                counts=[int(c) for c in counts])


def _pcts(v: np.ndarray) -> dict:
    if len(v) == 0:
        return {}
    q = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    return {f"p{p}": round(float(np.percentile(v, p)), 2) for p in q} | {
        "mean": round(float(v.mean()), 2), "std": round(float(v.std()), 2)}


def profile_corpus(docs: list[str], tok: BPETokenizer, name: str,
                   token_lens: np.ndarray | None = None,
                   sample_for_tokens: int = 20000) -> dict:
    """Descriptive statistics a data scientist would ask for before modeling."""
    char_lens = np.array([len(d) for d in docs], dtype=np.int64)
    word_lens = np.array([d.count(" ") + 1 for d in docs], dtype=np.int64)

    if token_lens is None:
        idx = np.linspace(0, len(docs) - 1, min(sample_for_tokens, len(docs))).astype(int)
        token_lens = np.array([len(tok.encode_ordinary(docs[i])) for i in idx], dtype=np.int64)

    # token frequency over a sample -> Zipf curve + vocabulary coverage
    sample_idx = np.linspace(0, len(docs) - 1, min(5000, len(docs))).astype(int)
    tf: Counter[int] = Counter()
    for i in sample_idx:
        tf.update(tok.encode_ordinary(docs[i]))
    total_tok = sum(tf.values())
    ranked = tf.most_common()
    pieces = tok.token_pieces()
    top = [dict(rank=r + 1, id=int(t), piece=pieces[t] if t < len(pieces) else "?",
                count=int(c), freq=round(c / total_tok, 6))
           for r, (t, c) in enumerate(ranked[:40])]
    zipf = [dict(rank=r + 1, count=int(c)) for r, (_, c) in
            enumerate(ranked[:2000]) if (r + 1) in _log_ranks(len(ranked))]

    chars = Counter()
    for d in docs[:5000]:
        chars.update(d)
    cls = dict(letters=0, digits=0, space=0, punct=0, other=0)
    for ch, c in chars.items():
        if ch.isalpha(): cls["letters"] += c
        elif ch.isdigit(): cls["digits"] += c
        elif ch.isspace(): cls["space"] += c
        elif ch.isprintable(): cls["punct"] += c
        else: cls["other"] += c
    tot_ch = max(sum(cls.values()), 1)

    total_bytes = int(sum(len(d.encode()) for d in docs))
    return dict(
        name=name, n_docs=len(docs), total_chars=int(char_lens.sum()), total_bytes=total_bytes,
        char_len=_pcts(char_lens), word_len=_pcts(word_lens), token_len=_pcts(token_lens),
        char_len_hist=_hist(char_lens), token_len_hist=_hist(token_lens),
        top_tokens=top, zipf=zipf,
        vocab_used=len(tf), vocab_size=tok.n_vocab,
        vocab_coverage=round(len(tf) / tok.n_vocab, 4),
        type_token_ratio=round(len(tf) / max(total_tok, 1), 5),
        char_classes={k: round(v / tot_ch, 4) for k, v in cls.items()},
        est_tokens=int(token_lens.mean() * len(docs)) if len(token_lens) else 0,
    )


def _log_ranks(n: int) -> set[int]:
    return {int(x) for x in np.unique(np.logspace(0, np.log10(max(n, 2)), 120).astype(int))}


# ------------------------------------------------------------------ tokenizing
def tokenize_to_bin(docs: list[str], tok: BPETokenizer, out_path: str,
                    add_eot: bool = True, verbose: bool = True) -> dict:
    t0 = time.time()
    eot = tok.eot_id
    buf: list[np.ndarray] = []
    n = 0
    for i, d in enumerate(docs):
        ids = tok.encode_ordinary(d)
        if add_eot:
            ids.append(eot)
        buf.append(np.array(ids, dtype=np.uint16))
        n += len(ids)
        if verbose and (i + 1) % 25000 == 0:
            print(f"  tokenized {i+1:,}/{len(docs):,} docs -> {n/1e6:.1f}M tokens "
                  f"({time.time()-t0:.0f}s)")
    arr = np.concatenate(buf) if buf else np.zeros(0, dtype=np.uint16)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    arr.tofile(out_path)
    return dict(path=out_path, n_tokens=int(arr.size), dtype="uint16",
                seconds=round(time.time() - t0, 1))


# ----------------------------------------------------------------------- CLI
def prepare_pretrain(args: argparse.Namespace) -> None:
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    report: dict = dict(created_at=time.time(), inputs=dict(train=args.input, val=args.val),
                        params={k: v for k, v in vars(args).items() if not callable(v)})

    print(f"[data] reading {args.input}")
    train_docs = read_documents(args.input, min_chars=args.min_doc_chars)
    val_docs = read_documents(args.val, min_chars=args.min_doc_chars) if args.val else []
    if args.max_docs:
        train_docs = train_docs[:args.max_docs]
    if args.max_val_docs:
        val_docs = val_docs[:args.max_val_docs]
    report["raw"] = dict(train_docs=len(train_docs), val_docs=len(val_docs))
    print(f"[data] raw: {len(train_docs):,} train docs, {len(val_docs):,} val docs")

    # ---- positive control for the dedup detectors (see docstring above)
    if args.dedup_selftest:
        ctl = validate_dedup_detectors(train_docs, perms=args.minhash_perms,
                                       bands=args.minhash_bands, shingle=args.shingle_size,
                                       threshold=args.near_dup_threshold)
        report["dedup_control"] = ctl
        if ctl.get("ran"):
            print(f"[data] dedup positive control: exact {ctl['exact_detected']}/{ctl['exact_planted']}, "
                  f"near {ctl['near_detected']}/{ctl['near_planted']}, "
                  f"leak {ctl['leak_detected']}/{ctl['leak_planted']} "
                  f"(false positives: near={ctl['near_false_positives']}, leak={ctl['leak_false_positives']})")

    # ---- exact dedup
    keep, ex = exact_dedup(train_docs)
    train_docs = [train_docs[i] for i in keep]
    report["exact_dedup_train"] = ex
    print(f"[data] exact dedup: removed {ex['n_exact_dupes']:,} -> {len(train_docs):,}")

    # ---- near dedup
    if args.near_dedup:
        t0 = time.time()
        sigs = minhash_signatures(train_docs, args.minhash_perms, args.shingle_size)
        keep, nd = near_dedup(train_docs, sigs, args.minhash_bands, args.near_dup_threshold)
        nd["seconds"] = round(time.time() - t0, 1)
        train_docs = [train_docs[i] for i in keep]
        report["near_dedup_train"] = nd
        print(f"[data] near dedup (MinHash{args.minhash_perms}/LSH{args.minhash_bands}): "
              f"removed {nd['n_removed']:,} in {nd['seconds']}s -> {len(train_docs):,}")

    # ---- leakage audit
    if val_docs and args.leakage_audit:
        keep, lk = leakage_audit(train_docs, val_docs, args.minhash_perms,
                                 args.shingle_size, args.minhash_bands, args.near_dup_threshold)
        val_docs = [val_docs[i] for i in keep]
        report["leakage_audit"] = lk
        print(f"[data] leakage: dropped {lk['n_leaked']:,}/{lk['n_val_in']:,} val docs "
              f"({lk['leak_rate']:.3%}) in {lk['seconds']}s")

    # ---- tokenizer
    if args.tokenizer and os.path.exists(args.tokenizer) and not args.retrain_tokenizer:
        print(f"[data] loading tokenizer {args.tokenizer}")
        tok = BPETokenizer.load(args.tokenizer)
    else:
        print(f"[data] training BPE vocab={args.vocab_size}")
        budget = args.tokenizer_train_chars
        acc, total = [], 0
        for d in train_docs:
            acc.append(d); total += len(d)
            if total >= budget:
                break
        tok = BPETokenizer().train(["\n".join(acc)], vocab_size=args.vocab_size)
        tok.save(args.tokenizer)
    report["tokenizer"] = dict(path=args.tokenizer, **tok.train_stats) | {"merge_log": None}
    report["tokenizer"]["merge_log_head"] = tok.train_stats.get("merge_log", [])[:60]

    # ---- profile (before writing bins)
    print("[data] profiling")
    report["profile_train"] = profile_corpus(train_docs, tok, "train")
    if val_docs:
        report["profile_val"] = profile_corpus(val_docs, tok, "val")

    # ---- tokenize
    print("[data] tokenizing train")
    report["bin_train"] = tokenize_to_bin(train_docs, tok, f"{args.out}.train.bin")
    if val_docs:
        print("[data] tokenizing val")
        report["bin_val"] = tokenize_to_bin(val_docs, tok, f"{args.out}.val.bin")

    n_train = report["bin_train"]["n_tokens"]
    n_val = report.get("bin_val", {}).get("n_tokens", 0)
    tr_bytes = report["profile_train"]["total_bytes"]
    va_bytes = report.get("profile_val", {}).get("total_bytes", 0)
    meta = dict(prefix=args.out, tokenizer=args.tokenizer, vocab_size=tok.n_vocab,
                train_tokens=n_train, val_tokens=n_val, dtype="uint16",
                eot_id=tok.eot_id, specials=tok.special_tokens,
                train_bytes=tr_bytes, val_bytes=va_bytes,
                # bits-per-byte needs this: it is the scale-free loss unit that
                # stays comparable across different tokenizers (arXiv:2010.14701)
                bytes_per_token_val=round(va_bytes / max(n_val, 1), 4),
                bytes_per_token_train=round(tr_bytes / max(n_train, 1), 4))
    with open(f"{args.out}.meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    report["meta"] = meta
    with open(f"{args.out}.profile.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[data] wrote {n_train/1e6:.2f}M train / {n_val/1e6:.2f}M val tokens")
    print(f"[data] profile -> {args.out}.profile.json")


def main() -> None:
    ap = argparse.ArgumentParser("slm.data")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare-pretrain")
    p.add_argument("--input", default="data/raw/stories_train.slice.txt")
    p.add_argument("--val", default="data/raw/stories_valid.txt")
    p.add_argument("--out", default="data/processed/stories")
    p.add_argument("--tokenizer", default="data/processed/tokenizer.json")
    p.add_argument("--vocab-size", type=int, default=8192)
    p.add_argument("--retrain-tokenizer", action="store_true")
    p.add_argument("--tokenizer-train-chars", type=int, default=30_000_000)
    p.add_argument("--min-doc-chars", type=int, default=64)
    p.add_argument("--max-docs", type=int, default=0)
    p.add_argument("--max-val-docs", type=int, default=5000)
    p.add_argument("--near-dedup", action="store_true", default=True)
    p.add_argument("--no-near-dedup", dest="near_dedup", action="store_false")
    p.add_argument("--leakage-audit", action="store_true", default=True)
    p.add_argument("--near-dup-threshold", type=float, default=0.8)
    p.add_argument("--minhash-perms", type=int, default=64)
    p.add_argument("--minhash-bands", type=int, default=16)
    p.add_argument("--shingle-size", type=int, default=5)
    p.add_argument("--dedup-selftest", action="store_true", default=True)
    p.add_argument("--no-dedup-selftest", dest="dedup_selftest", action="store_false")
    p.set_defaults(func=prepare_pretrain)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
