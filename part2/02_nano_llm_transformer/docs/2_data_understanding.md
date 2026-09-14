# Phase 2 — Data Understanding

## Source

**TinyStories V2 (GPT-4 generated)** — Eldan & Li (2023), *TinyStories: How Small
Can Language Models Be and Still Speak Coherent English?*, arXiv:2305.07759.
Retrieved from the `roneneldan/TinyStories` and `roneneldan/TinyStoriesInstruct`
repositories on Hugging Face.

| Split | File | Used |
|---|---|---|
| Pretrain train | `TinyStoriesV2-GPT4-train.txt` | first 120 MB (HTTP range request) |
| Pretrain val | `TinyStoriesV2-GPT4-valid.txt` | full 21 MB, capped to 4,000 documents |
| Instruction train | `TinyStories-Instruct-train.txt` | first 60 MB |
| Instruction val | `TinyStories-Instruct-valid.txt` | full 26 MB |

### Why this corpus

The central constraint is that a ~9M-parameter model has almost no capacity for
world knowledge. TinyStories exists precisely for this regime: it is synthetic
text restricted to the vocabulary and situations of a 3–4 year old, which lets a
very small model produce genuinely fluent, coherent English instead of the
plausible-looking word salad that a small model trained on web text produces.

The instruction variant is what makes a *chatbot* possible rather than just a
text continuer: each record carries `Features`, `Words`, and `Summary` fields
alongside the story, which invert cleanly into natural user requests.

### Provenance caveats

Being model-generated, this corpus carries the stylistic and distributional
biases of its generator, and its "coherence" is partly an artifact of a narrow
generator distribution. Results here do **not** transfer to natural web text.

## Description

- **153,506** training documents / **4,000** held-out documents after preparation
- **30.29M** training tokens, **782.7K** validation tokens at vocabulary 8,192
- Median document ≈ **722 characters / 175 tokens**; p95 ≈ 1,463 characters
- **4.08 bytes per token** compression
- Character composition is overwhelmingly lowercase letters and spaces; almost no
  digits, and a very small punctuation set

## Exploration findings

**Vocabulary saturation.** Pre-tokenizing 30M characters yields only **14,545
unique pre-tokens**. That is a startlingly small type count and it is the single
most important property of this corpus. It means an 8,192-entry BPE vocabulary is
near saturation: by merge ~6,000 the merge counts fall below 40 occurrences, so
the tail of the vocabulary is memorizing individual rare words rather than
learning reusable subword structure. The merge-saturation curve on the Data tab
shows this directly, and it is the empirical argument for a *smaller* vocabulary
at this scale.

**Zipfian structure.** The rank–frequency curve is close to a straight line on
log–log axes, as expected for natural-language-like text, confirming the
generator did not produce a degenerate distribution.

## Quality verification

Three checks ran before any modeling:

1. **Exact duplicates** (SHA-1 over punctuation-stripped, case-folded text): **0**
2. **Near duplicates** (MinHash, 64 permutations, 16 LSH bands × 4 rows, 5-word
   shingles, Jaccard ≥ 0.8): **0**
3. **Train↔validation leakage** (same machinery, cross-split): **0 of 4,000**

### The zero problem, and the positive control

All three checks returned zero, which is exactly what a *silently broken*
detector also returns. So the pipeline always runs a **positive control** first:
50 exact copies, 30 truncated near-copies, and 25 planted cross-split leaks are
inserted into a clean sample, and recall is measured.

| Detector | Planted | Detected | Recall | False positives |
|---|---|---|---|---|
| Exact (SHA-1) | 50 | 50 | 100% | 0 |
| Near (MinHash/LSH) | 30 | 30 | 100% | 0 |
| Train↔val leakage | 25 | 25 | 100% | 0 |

With the detectors demonstrated at 100% recall and zero false positives, the zero
finding is credible: **this corpus is genuinely clean**, because it was
deduplicated upstream by its authors. Held-out numbers in this project can be
taken at face value.
