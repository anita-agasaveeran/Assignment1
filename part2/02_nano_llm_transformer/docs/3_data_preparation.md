# Phase 3 — Data Preparation

Command: `python -m slm.data prepare-pretrain` → `data/processed/stories.*`

## Selection

Documents are split on the `<|endoftext|>` delimiter and dropped below 64
characters. The 120 MB training slice is taken as a byte range from the head of
the upstream file and truncated at the last complete document boundary, so no
document is torn.

The validation split comes from the **upstream validation file**, not from a
random split of training data. This matters: TinyStories documents are generated
from prompt templates, so a random split would place near-identical documents on
both sides. Using the publisher's split, then auditing it, is strictly safer.

## Cleaning

Ordered, because order changes the result — exact duplicates must go before
MinHash, or duplicate clusters distort the LSH bucket distribution:

1. **Exact dedup** — SHA-1 of case-folded, punctuation-stripped, whitespace-collapsed text.
2. **Near dedup** — MinHash signatures (64 permutations over 5-word shingles),
   banded LSH (16 bands × 4 rows) for candidate generation, exact
   signature-Jaccard verification at ≥ 0.8, union-find clustering, keep lowest index.
3. **Leakage audit** — validation documents matching any training document
   (exact or near) are removed from validation *before* any metric is computed.

Banding parameters are the standard Broder trade-off: with 16 bands × 4 rows the
probability of a pair becoming a candidate is `1-(1-J⁴)¹⁶`, which is ≈ 0.98 at
J = 0.8 and ≈ 0.07 at J = 0.4 — high recall at the threshold, cheap at low
similarity. Both are verified empirically by the positive control (100% recall,
0 false positives).

## Construction — tokenizer

Byte-level BPE, **trained from scratch** (`slm/tokenizer.py`), no `tokenizers` or
`sentencepiece` dependency.

- Base alphabet is the 256 byte values, so the tokenizer is lossless on any
  input and can never emit `<unk>`.
- GPT-2-style regex pre-tokenization; merges never cross a pre-token boundary,
  which preserves whitespace/punctuation structure and makes training
  `O(merges × affected words)` instead of `O(merges × corpus)`.
- Training uses an inverted index from pair → containing words, so each merge
  touches only the words that contain it. 8,192 merges over 30M characters
  trains in **~8 seconds**.
- Specials: `<|endoftext|>`, `<|im_start|>`, `<|im_end|>`, `<|pad|>`.
- **Fitted on the training split only** — fitting on validation text would leak
  distributional information into the compression rate.

Result: **4.08 bytes/token**. This is the denominator for bits-per-byte, the
tokenizer-independent loss unit (Henighan et al. 2020, arXiv:2010.14701).

## Formatting — pretraining

Documents are encoded, an `<|endoftext|>` appended to each, and the stream
concatenated into a flat `uint16` array written to `.train.bin` / `.val.bin`.
Training memory-maps these and samples random fixed-length windows, so the
sampler is O(1) and RAM never holds the corpus.

`uint16` is safe because vocabulary ≤ 65,536; it halves I/O against `int32`.

## Formatting — instruction data

Command: `python -m slm.chat build-sft` → `data/processed/chat.*`

TinyStories-Instruct records carry `Features` / `Words` / `Summary` / `Story`.
These invert into user requests via templates — a no-model-in-the-loop instance
of the Self-Instruct idea (arXiv:2212.10560):

```
Words: quit, oak, gloomy
Summary: Sara and Ben were playing in the park…
→ user: "I want a story with the words quit, oak, gloomy. The story should be
   about: Sara and Ben were playing in the park… Include some dialogue."
→ assistant: <the story>
```

Rendered into **ChatML** and packed to 512 tokens with an explicit **loss mask**.

**The loss mask is the part that matters.** Gradients flow only through assistant
content and the closing `<|im_end|>`:

- Training on the *user* turn teaches the model to write questions, which at chat
  time shows up as the model interviewing itself.
- Including the closing `<|im_end|>` in the loss is what teaches it to **stop**.
  Omit it and generation runs to the token budget every time.

`slm/sft.py` refuses to start if the mask covers <5% or >95% of positions, since
a silently all-ones mask is easy to ship and hard to notice.

Resulting statistics: **38,349** train / **1,982** validation conversations,
**72.5%** of packed tokens supervised, **48.4%** padding, 1,651 dropped for
exceeding the context window.

The 48% padding is real waste and the obvious next optimization (sequence packing
or length-bucketed batching); it is left in place because it keeps the mask
logic auditable, which at this scale is the better trade.
