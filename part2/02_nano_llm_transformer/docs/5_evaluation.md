# Phase 5 — Evaluation

Command: `python -m slm.evaluate` → `runs/<run_id>/eval.json`

Held-out loss alone is not an evaluation. It is one scalar that hides where the
loss comes from, whether the model's confidence means anything, and whether it
uses its context at all.

## Metrics and what each is for

### Quality
- **Cross-entropy / perplexity** — the optimization target.
- **Bits per byte** — `CE / ln2 / bytes_per_token`. The tokenizer-independent
  unit; perplexity is not comparable across vocabularies, bits-per-byte is
  (arXiv:2010.14701).
- **Top-1 / top-5 accuracy** — next-token accuracy. Diverges from perplexity when
  the model is right but unconfident.

### Calibration
- **ECE + reliability diagram** — predictions binned by confidence; the gap
  between mean confidence and observed accuracy per bin. A model can be accurate
  and badly overconfident, and this matters the moment you *sample* from it:
  sampling temperature interacts directly with how honest the distribution is.
  Bars below the diagonal mean overconfidence.

### Structure — where the aggregate number hides things
- **Loss by position in context** — average loss at each of the 512 positions. A
  falling curve means earlier context is genuinely being used; a flat curve means
  the model is effectively ignoring it, which no perplexity number would reveal.
- **Loss by token-frequency decile** — targets bucketed by corpus frequency mass.
  Aggregate perplexity is dominated by the handful of tokens carrying most of the
  probability mass; this separates "good at `the`" from "good at rare words."

### Generation
- **distinct-1/2/3**, **repeated 4-gram rate**, **stop rate**, mean length.

### Decoding study
A direct reproduction of Holtzman et al. 2019 (arXiv:1904.09751): identical
prompts and seed, varying only the decoding rule across greedy, temperature,
top-k, top-p, and top-p with repetition penalty. The expected result is that
maximization decoding collapses into repetition while nucleus sampling does not.
This also fixes the serving defaults empirically rather than by folklore.

### Task metric — instruction compliance
For SFT checkpoints: the user asks for a story containing specific words; the
harness checks whether those words actually appear. Reported as **word inclusion
rate** and **fully compliant rate**. This is a real, automatically-checkable
task metric rather than a proxy — the closest thing to "did it do what was
asked" available at this scale.

## Interpreting results at this scale

Three honest caveats belong on every number produced here:

1. **The model is under-trained relative to its size.** The token budget is well
   under the Chinchilla-optimal ≈20 tokens/parameter, so quality is data-limited,
   not capacity-limited. Loss would keep falling with more tokens at fixed size.
   The exact ratio is shown on the Training tab and stated before each run.
2. **The domain is extremely narrow.** Low perplexity here reflects a ~1,500-word
   vocabulary of simple children's stories. It says nothing about general ability.
3. **AutoResearch conclusions are scale-local.** A technique rejected at 3M
   parameters over 300 steps is not refuted — several of these techniques
   (QK-norm, z-loss, logit soft-capping) exist to prevent instabilities that only
   appear at much larger scale, and there is no instability at this scale for
   them to prevent. The dashboard reports "did not clear the bar here," which is
   a different claim from "does not work."

## Process review

What this evaluation still lacks, stated rather than glossed:

- No comparison against an external baseline model at matched parameter count.
- No human evaluation of story quality; distinct-n and compliance are proxies.
- Instruction compliance checks word *presence*, not correct usage or coherence.
- A single held-out set from one generator; no distribution-shift evaluation.

---

## Measured results (champion, 16.9M params, 65.5M tokens)

| Metric | Pretrained | After SFT |
|---|---|---|
| Held-out CE / perplexity | 1.486 / 4.42 | 1.751 / 5.76 (story val) |
| Bits per byte | 0.526 | — |
| Top-1 / top-5 accuracy | 61.9 % / 87.3 % | 58.2 % / 84.5 % |
| ECE | 0.032 | 0.048 |
| Loss at position 0 → 511 | 3.67 → 1.66 | — |
| Loss, most vs least frequent decile | 0.45 vs 3.50 nats | — |
| Greedy repeated-4-gram rate | 0.5 % | 9.7 % (nucleus: 1.1 %) |
| Word-inclusion compliance | 8.3 % | 43.9 % |
| Fully compliant replies | 0 % | 11.7 % |
| Decode / TTFT | 441 tok/s / 1.9 ms | — |

**Against the success criteria in phase 1**

- Perplexity 4.42 against a vocabulary of 8,192. Measured baselines on the same
  validation tokens: **unigram 341.5 (5.83 nats), bigram 36.4 (3.60 nats)**. The
  model's 1.49 nats is 2.1 nats below the bigram model — the structure it has
  learned is long-range, not just local. Compute budget stated: 0.31×
  Chinchilla-optimal. ✅
- Every accepted architectural change exceeded 1σ of measured seed noise; five
  did, twelve did not, and two were significant-but-unaccepted at the round
  limit. ✅ (with the budget-limited caveat)
- Zero leakage, demonstrated with a positive control. ✅

**What did not reproduce as expected**

The decoding study was designed to reproduce Holtzman et al. (2019). On the
pretrained model greedy decoding barely repeated itself at all. The effect
appears after SFT (9.7 % vs 1.1 %) but is an order of magnitude milder than on
open-domain text. This is a property of the corpus, not a refutation of the
paper: TinyStories has ~1,500 word types and short, formulaic documents, so
the argmax path rarely enters the high-probability loops that cause
degeneration on web text. It is recorded here because a negative result that
is explained is more useful than a positive one that is assumed.
