# Phase 1 — Business Understanding

## Objective

Build a language model and chatbot **from first principles**, using current
architectural primitives, that trains end-to-end on a single laptop GPU — and
instrument the whole thing so that every design decision is traceable to
evidence.

The deliverable is not the model. A 9M-parameter story model has no standalone
value. The deliverable is a **reproducible pipeline plus the evidence trail**:
which techniques were tried, which paper proposed each one, what each was
measured to do at this scale, and whether that measurement was distinguishable
from noise.

## Why this framing

The usual failure mode of a "build a small LLM" project is cargo-culting: copy a
reference implementation containing RMSNorm, RoPE, SwiGLU and GQA, observe that
it trains, and conclude the choices were validated. Nothing was validated. The
model would also have trained without them.

So the project is organized around an inversion: start from a deliberately dated
2019-era baseline and make every modern primitive **earn its place** through a
measured, significance-tested A/B. That turns the architecture section from a
list of assertions into a set of results.

## Business questions

| # | Question | Where answered |
|---|---|---|
| Q1 | Can a useful chatbot be trained end-to-end on one laptop GPU, within hours? | Training + Model card |
| Q2 | Which published techniques actually help at this scale — and which do not? | AutoResearch |
| Q3 | Is held-out performance trustworthy, or inflated by duplicated/leaked data? | Data |
| Q4 | Is the model calibrated, and does it actually use its context? | Evaluation |
| Q5 | What does it cost to serve, and what are its failure modes? | Model card |

## Success criteria

**Analytic**
- Held-out perplexity substantially below a unigram baseline, with the compute
  budget stated relative to the Chinchilla-optimal point rather than hidden.
- Every accepted architectural change supported by an effect exceeding 1σ of
  measured seed-to-seed noise. Effects inside the noise band are reported as
  *null results*, not as wins.
- Zero train/validation leakage, demonstrated with a **positive control** —
  detectors validated on planted duplicates, because a detector that reports
  zero is otherwise indistinguishable from a broken one.

**Engineering**
- Full pipeline runs from raw text to served chatbot with documented commands.
- Every run's config, environment, git state and metrics recorded in a single
  queryable database; any dashboard number reproducible from it.
- Inference and evaluation share one code path, so offline numbers describe the
  thing actually being served.

**Explicitly out of scope:** competitive benchmark scores, multilingual support,
factual question answering, RLHF/preference optimization, safety tuning,
multi-GPU or distributed training.

## Constraints

| Constraint | Value | Consequence |
|---|---|---|
| Compute | 1× Apple M4 Pro, 20-core GPU, 24 GB unified | Model ≲ 20M params; bf16; no CUDA-only kernels |
| Backend | PyTorch MPS | No FlashAttention CUDA kernel; use `scaled_dot_product_attention` |
| Wall-clock | Hours, not days | Architecture search must run on a low-fidelity proxy |
| Dependencies | Offline-capable | Tokenizer written from scratch; dashboard has no CDN assets |

## Risks

| Risk | Mitigation |
|---|---|
| Noise mistaken for improvement | Measure seed variance first; require >1σ to accept |
| Proxy-scale results not transferring | Retrain champion at full scale and re-validate across seeds |
| Held-out set contaminated | MinHash/LSH leakage audit before any number is reported |
| Greedy search missing interactions | Stated as a known limitation; accepted order published |
| Overclaiming from one corpus | Domain limits stated explicitly in the model card |

## Process

CRISP-DM, run as an explicit loop rather than a waterfall: AutoResearch is
literally a Modeling → Evaluation cycle executed dozens of times, with each
iteration's outcome feeding the next configuration.

> Wirth, R. & Hipp, J. (2000). *CRISP-DM: Towards a Standard Process Model for
> Data Mining.*
