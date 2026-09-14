"""Byte-level BPE trained from scratch (no tokenizers/sentencepiece dependency).

Follows Sennrich et al. 2015 (arXiv:1508.07909) with the byte-level base
alphabet popularized by GPT-2: the base vocabulary is the 256 byte values, so
the tokenizer is lossless for *any* input and can never emit <unk>.

Training operates on a pre-tokenized word-frequency table rather than the raw
stream. Merges are never allowed to cross a pre-token boundary, which keeps
whitespace/punctuation structure stable and makes training O(merges * affected
words) instead of O(merges * corpus).
"""
from __future__ import annotations

import json
import os
import re
import time
from collections import Counter, defaultdict
from typing import Iterable

# GPT-2-style pre-tokenizer, expressed in the stdlib `re` dialect (no external
# `regex` dependency). Contractions, letter runs, digit runs, punctuation runs,
# and whitespace are each their own pre-token.
SPLIT_PAT = re.compile(
    r"'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+"
)

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|pad|>"]


class BPETokenizer:
    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.special_tokens: dict[str, int] = {}
        self.pattern = SPLIT_PAT.pattern
        self._cache: dict[str, list[int]] = {}
        self._special_re: re.Pattern | None = None
        self.train_stats: dict = {}

    # ------------------------------------------------------------ properties
    @property
    def n_vocab(self) -> int:
        return len(self.vocab) + len(self.special_tokens)

    @property
    def eot_id(self) -> int:
        return self.special_tokens["<|endoftext|>"]

    # -------------------------------------------------------------- training
    def train(self, text_iter: Iterable[str], vocab_size: int,
              specials: list[str] | None = None, verbose: bool = True) -> "BPETokenizer":
        specials = specials or SPECIALS
        n_merges = vocab_size - 256 - len(specials)
        if n_merges < 0:
            raise ValueError(f"vocab_size={vocab_size} too small for 256 bytes + {len(specials)} specials")

        t0 = time.time()
        word_freq: Counter[str] = Counter()
        n_chars = 0
        for chunk in text_iter:
            n_chars += len(chunk)
            word_freq.update(SPLIT_PAT.findall(chunk))
        if verbose:
            print(f"[bpe] pre-tokenized {n_chars/1e6:.1f}M chars -> "
                  f"{len(word_freq):,} unique pre-tokens in {time.time()-t0:.1f}s")

        # Each unique pre-token becomes a mutable list of symbol ids.
        words: list[list[int]] = []
        freqs: list[int] = []
        for w, f in word_freq.items():
            words.append(list(w.encode("utf-8")))
            freqs.append(f)

        # Inverted index pair -> word indices, so a merge only touches the
        # words that actually contain the pair.
        pair_counts: Counter[tuple[int, int]] = Counter()
        pair_where: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
        for i, sym in enumerate(words):
            f = freqs[i]
            for p in zip(sym, sym[1:]):
                pair_counts[p] += f
                pair_where[p].add(i)

        merges: dict[tuple[int, int], int] = {}
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        merge_log: list[dict] = []
        t1 = time.time()

        for m in range(n_merges):
            if not pair_counts:
                if verbose:
                    print(f"[bpe] corpus exhausted after {m} merges")
                break
            pair, cnt = max(pair_counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
            if cnt <= 0:
                break
            new_id = 256 + m
            merges[pair] = new_id
            vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]

            for wi in list(pair_where[pair]):
                sym = words[wi]
                f = freqs[wi]
                # remove this word's old pair contributions
                for p in zip(sym, sym[1:]):
                    pair_counts[p] -= f
                    if pair_counts[p] <= 0:
                        pair_counts.pop(p, None)
                    pair_where[p].discard(wi)
                # apply every occurrence of `pair` left-to-right
                out: list[int] = []
                j = 0
                while j < len(sym):
                    if j < len(sym) - 1 and sym[j] == pair[0] and sym[j + 1] == pair[1]:
                        out.append(new_id)
                        j += 2
                    else:
                        out.append(sym[j])
                        j += 1
                words[wi] = out
                for p in zip(out, out[1:]):
                    pair_counts[p] += f
                    pair_where[p].add(wi)
            pair_where.pop(pair, None)
            pair_counts.pop(pair, None)

            merge_log.append(dict(rank=m, new_id=new_id, count=cnt,
                                  piece=vocab[new_id].decode("utf-8", errors="replace")))
            if verbose and (m + 1) % 1000 == 0:
                print(f"[bpe] merge {m+1}/{n_merges}  '{merge_log[-1]['piece']}'  count={cnt}"
                      f"  ({time.time()-t1:.0f}s)")

        self.merges, self.vocab = merges, vocab
        self.special_tokens = {s: len(vocab) + i for i, s in enumerate(specials)}
        self._build_special_re()
        self._cache.clear()

        total_bytes = sum(len(w.encode()) * f for w, f in word_freq.items())
        total_tokens = sum(len(words[i]) * freqs[i] for i in range(len(words)))
        self.train_stats = dict(
            n_chars=n_chars, unique_pretokens=len(word_freq), n_merges=len(merges),
            vocab_size=self.n_vocab, train_seconds=round(time.time() - t0, 1),
            corpus_bytes=total_bytes, corpus_tokens=total_tokens,
            bytes_per_token=round(total_bytes / max(total_tokens, 1), 3),
            merge_log=merge_log[:200],
        )
        if verbose:
            print(f"[bpe] done: vocab={self.n_vocab} "
                  f"compression={self.train_stats['bytes_per_token']:.2f} bytes/token "
                  f"in {self.train_stats['train_seconds']}s")
        return self

    # -------------------------------------------------------------- encoding
    def _build_special_re(self) -> None:
        if self.special_tokens:
            self._special_re = re.compile("(" + "|".join(
                re.escape(s) for s in sorted(self.special_tokens, key=len, reverse=True)) + ")")

    def _encode_word(self, word: str) -> list[int]:
        hit = self._cache.get(word)
        if hit is not None:
            return hit
        ids = list(word.encode("utf-8"))
        if len(ids) > 1:
            while True:
                # merge the pair with the lowest merge rank present
                best, best_rank = None, None
                for p in zip(ids, ids[1:]):
                    r = self.merges.get(p)
                    if r is not None and (best_rank is None or r < best_rank):
                        best, best_rank = p, r
                if best is None:
                    break
                out, j = [], 0
                while j < len(ids):
                    if j < len(ids) - 1 and ids[j] == best[0] and ids[j + 1] == best[1]:
                        out.append(self.merges[best]); j += 2
                    else:
                        out.append(ids[j]); j += 1
                ids = out
        if len(self._cache) < 500_000:
            self._cache[word] = ids
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        out: list[int] = []
        for w in SPLIT_PAT.findall(text):
            out.extend(self._encode_word(w))
        return out

    def encode(self, text: str, allowed_special: bool = True) -> list[int]:
        """Encode text; when `allowed_special`, special-token literals map to their id."""
        if not allowed_special or not self._special_re:
            return self.encode_ordinary(text)
        out: list[int] = []
        for part in self._special_re.split(text):
            if not part:
                continue
            sid = self.special_tokens.get(part)
            out.append(sid) if sid is not None else out.extend(self.encode_ordinary(part))
        return out

    def decode(self, ids: Iterable[int]) -> str:
        inv = {v: k for k, v in self.special_tokens.items()}
        buf = bytearray()
        chunks: list[str] = []
        for i in ids:
            i = int(i)
            if i in inv:
                if buf:
                    chunks.append(buf.decode("utf-8", errors="replace")); buf.clear()
                chunks.append(inv[i])
            else:
                buf += self.vocab.get(i, b"")
        if buf:
            chunks.append(buf.decode("utf-8", errors="replace"))
        return "".join(chunks)

    # ------------------------------------------------------------ persistence
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(dict(
                pattern=self.pattern,
                merges=[[a, b, i] for (a, b), i in self.merges.items()],
                special_tokens=self.special_tokens,
                train_stats=self.train_stats,
            ), f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path) as f:
            d = json.load(f)
        t = cls()
        t.pattern = d["pattern"]
        t.merges = {(a, b): i for a, b, i in d["merges"]}
        t.special_tokens = {k: int(v) for k, v in d["special_tokens"].items()}
        t.train_stats = d.get("train_stats", {})
        for (a, b), i in sorted(t.merges.items(), key=lambda kv: kv[1]):
            t.vocab[i] = t.vocab[a] + t.vocab[b]
        t._build_special_re()
        return t

    def token_pieces(self) -> list[str]:
        pieces = [self.vocab[i].decode("utf-8", errors="replace") for i in sorted(self.vocab)]
        pieces += [s for s, _ in sorted(self.special_tokens.items(), key=lambda kv: kv[1])]
        return pieces
