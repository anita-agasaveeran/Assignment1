# CRISP-LM — a small language model, built to be argued with

A language model and chatbot built from scratch with current architectural
primitives, trained end-to-end on a laptop GPU (Apple M4 Pro, PyTorch/MPS),
organized as a full CRISP-DM cycle — and instrumented so that **every design
choice traces to a measurement, and every measurement traces to a paper.**

The model is not the deliverable. A 9M-parameter story model has no standalone
value. The deliverable is the pipeline plus the evidence trail: which techniques
were tried, who proposed each, what each measurably did at this scale, and
whether that was distinguishable from noise.

---

## The central idea

The usual way to "build a small LLM" is to copy a reference implementation
containing RMSNorm, RoPE, SwiGLU and GQA, watch it train, and call the choices
validated. Nothing was validated — it would have trained without them.

So this project inverts it. **AutoResearch** starts from a deliberately dated
2019-era baseline — LayerNorm, learned absolute positions, GELU feed-forward,
full multi-head attention, untied embeddings, dropout 0.1, AdamW + cosine — and
makes every modern primitive earn its place through a paired,
significance-tested A/B against a *measured* seed-noise floor.

### What the search found (study `hillclimb-v1`, 74 minutes, 86 proxy training runs: 72 candidate trials, 8 noise-floor/incumbent references, 6 final validation runs)

| | val loss | |
|---|---|---|
| 2019 baseline (mean of 3 seeds) | **3.219** | |
| Champion (mean of 3 fresh seeds) | **2.472** | −23.2 %, Cohen's *d* = 1.82 |

Noise floor σ = 0.038 nats across baseline seeds; a move had to beat the
incumbent by more than 1σ to be accepted.

**Accepted, in order** (each Δ is against the incumbent at that round):

| Round | Technique | Δ loss | z | Paper |
|---|---|---|---|---|
| 1 | Muon optimizer | −0.333 | −8.9 | Liu et al. 2025, arXiv:2502.16982 |
| 2 | Disable dropout | −0.169 | −4.5 | Srivastava 2014 / Hoffmann 2022 |
| 3 | Rotary position embeddings | −0.097 | −2.6 | Su et al. 2021, arXiv:2104.09864 |
| 4 | QK-normalization | −0.056 | −1.5 | Henry 2020 / Dehghani 2023 |
| 5 | 2× peak learning rate | −0.062 | −1.7 | Yang et al. 2022, arXiv:2203.03466 |

**Three results a copy-the-recipe approach would never have produced:**

- **RMSNorm measured Δ = −0.0004.** That is *exactly* the paper's claim — "matches
  LayerNorm quality" — the benefit is speed, not loss, and the search correctly
  reported a null on loss.
- **QK-norm hurt by +0.33 in round 1 and helped by −0.06 in round 4.** It only
  pays off once Muon and RoPE are in place. Greedy search can miss interactions
  like this; here it happened to catch one because the technique stayed in the
  pool.
- **2× learning rate was a disaster under AdamW (val 3.72, z = +14) and a win
  under Muon (z = −1.7).** The optimizer changed what learning rate was safe — an
  interaction the search discovered on its own.

**Rejected at this scale:** GQA (+0.02, within noise — its benefit is KV-cache
size, which is measured separately), z-loss and logit soft-capping (both ≈0 —
they prevent large-scale instabilities that do not exist at 3M parameters),
label smoothing, longer warmup, WSD schedule, residual-init scaling, weight
decay, RoPE base.

**Budget-limited, not converged:** tied embeddings (−0.045) and SwiGLU (−0.044)
still cleared the significance bar in the final round. The study stopped at its
5-round limit; both would most likely have been accepted next. The champion is a
lower bound on the search space.

Full per-round tables, every trial's claim-vs-measurement, and the noise band are
on the **AutoResearch** dashboard tab. Method and limitations:
[docs/autoresearch.md](docs/autoresearch.md).

---

## Results — the trained model

Champion recipe scaled to **6 layers × d384, 16.9M parameters** (10.6M
non-embedding), trained 4,000 steps × 32 × 512 = **65.5M tokens in 47 min** on
the M4 Pro at ~23K tok/s (≈34 % MFU against a measured 7.2 TFLOP/s roofline).
That is **0.31× the Chinchilla-optimal token budget** — the model is
data-limited and val loss was still falling at the last step.

| | Pretrained | After SFT (chat) |
|---|---|---|
| Held-out cross-entropy | **1.486** nats | 1.751 nats* |
| &nbsp;&nbsp;↳ baselines on same tokens: unigram / bigram | 5.833 / 3.595 nats | |
| Perplexity | **4.42** | 5.76* |
| Bits per byte | **0.526** | — |
| Next-token accuracy (top-1 / top-5) | 61.9 % / 87.3 % | 58.2 % / 84.5 % |
| Calibration error (ECE) | **0.032** | 0.048 |
| Instruction compliance (requested words used) | 8.3 % | **43.9 %** |
| Fully compliant replies | 0 % | 11.7 % |
| Decode speed / TTFT (batch 1, KV cache) | 441 tok/s / 1.9 ms | ~150–440 tok/s |
| KV cache at full 512 context | 4.7 MB | 4.7 MB |

\* measured on the *story* validation set, on which the chat-tuned model is
expectedly a little worse — it has been pulled toward the ChatML distribution.

**What the structural metrics show**

- **Context is used.** Loss at position 0 is 3.67 nats; by the end of the
  512-token window it is 1.66. The model genuinely conditions on what came before.
- **Rare tokens are where the loss lives.** The most-frequent decile of targets
  costs 0.45 nats; the rarest decile costs 3.50 — an 8× gap that aggregate
  perplexity hides entirely.
- **Well calibrated.** ECE 0.032 on next-token prediction; the reliability
  diagram sits close to the diagonal, slightly under-confident in the middle bins.
- **Degeneration is weak here.** Greedy decoding produced 0.5 % repeated
  4-grams on the base model and 9.7 % after SFT (vs 1.1 % with nucleus
  sampling). The Holtzman et al. effect reproduces in *direction* on the chat
  model but not in the dramatic magnitude the paper reports on web text —
  TinyStories is low-entropy and short, which is the honest explanation.
- **SFT did its job.** Word-inclusion compliance rose from 8 % to 44 %; the
  model stops at `<|im_end|>` reliably instead of running to the token budget.

A sample, from the dashboard chat (temperature 0.8, top-p 0.9):

> **User:** Write a short story that uses the words: cat, hat, run.
> **Model:** Once upon a time, there was a cat named Kitty. Kitty loved to run
> and play. One day, Kitty was running very fast and saw a big hat on the
> ground. She thought it would be fun to wear the hat so she could run faster…

---

## Quickstart

```bash
make setup      # venv + dependencies (torch, numpy, pyyaml, fastapi, uvicorn)
make download   # TinyStories — a 120MB byte-range slice, not the full 1.9GB
make data       # profile, dedup, leakage audit, train tokenizer, tokenize
make sft-data   # build the ChatML instruction dataset
```

Then:

```bash
make autoresearch   # hill climb over the paper registry   (~75 min)
make champion       # scale the discovered recipe to full size
make pretrain       # train the full model                  (~55 min)
make sft            # instruction-tune it into a chatbot    (~10 min)
make evaluate       # full evaluation report                (~5 min)
make dashboard      # dashboard + chat at http://127.0.0.1:8099
make chat           # terminal chat REPL
make test           # 33 tests
```

`make all` runs the whole pipeline in order.

---

## What is in here

Everything below is implemented in this repository — no `transformers`, no
`tokenizers`, no chart library, no experiment-tracking service. The project runs
fully offline.

| Module | What it does |
|---|---|
| `slm/model.py` | Decoder-only transformer; every primitive is a config flag (RMSNorm/LayerNorm, RoPE/learned, SwiGLU/GELU, GQA, QK-norm, tied embeddings, z-loss, soft-capping). KV-cache generation with top-k/top-p/repetition penalty. |
| `slm/optim.py` | AdamW with correct 2-D-only weight decay; **Muon** (Newton–Schulz orthogonalized momentum) on hidden weights; cosine / WSD / linear schedules. |
| `slm/tokenizer.py` | Byte-level BPE trained from scratch — lossless on any input, 8K vocab in ~8s via an inverted pair index. |
| `slm/data.py` | Profiling, exact + MinHash/LSH near-dedup, train↔val **leakage audit**, and a **positive control** that plants duplicates and measures recall before trusting a zero. |
| `slm/train.py` | Training loop with fixed eval batches (identical across runs), MFU against a *measured* roofline, Chinchilla accounting up front, full provenance before step 0. |
| `slm/chat.py`, `slm/sft.py` | ChatML templating with an explicit assistant-only loss mask; supervised fine-tuning. |
| `slm/autoresearch.py` | Greedy hill climbing: noise floor first, paired trials, proxy scale, champion re-validated on fresh seeds. |
| `slm/evaluate.py` | Perplexity, bits/byte, ECE + reliability, loss by context position, loss by token-frequency decile, a decoding study reproducing Holtzman 2019, instruction compliance. |
| `slm/registry.py` | 60 papers, metadata **fetched from the arXiv API** (not recalled), each mapped to the intervention that tests it. |
| `slm/tracking.py` | SQLite run store — the single source of truth the dashboard reads. |
| `app/` | FastAPI dashboard + streaming chat; hand-written SVG charts; colorblind-safe palette; light + dark. |

---

## CRISP-DM mapping

| Phase | Where | Document |
|---|---|---|
| 1 Business understanding | framing, success criteria, risks | [docs/1](docs/1_business_understanding.md) |
| 2 Data understanding | `slm/data.py` · Data tab | [docs/2](docs/2_data_understanding.md) |
| 3 Data preparation | dedup, tokenizer, ChatML packing | [docs/3](docs/3_data_preparation.md) |
| 4 Modeling | `model.py`, `train.py`, `sft.py`, `autoresearch.py` | [docs/4](docs/4_modeling.md) |
| 5 Evaluation | `evaluate.py` · Evaluation tab | [docs/5](docs/5_evaluation.md) |
| 6 Deployment | `app/server.py`, `slm/infer.py` · Chat tab | [docs/6](docs/6_deployment.md) |

CRISP-DM is run as a loop: AutoResearch is a Modeling → Evaluation cycle
executed 86 times.

---

## Data

TinyStories V2 (Eldan & Li 2023, arXiv:2305.07759): 153,506 training documents,
30.3M tokens; 4,000 held-out documents from the publisher's own validation
split. **Zero** exact duplicates, near-duplicates, or train↔val leaks — a
finding that is credible only because the detectors were first shown to hit
100 % recall with 0 false positives on planted duplicates. Details:
[docs/2](docs/2_data_understanding.md).

## Reproducibility

- Every run records config, environment, git commit + dirty flag, and a measured
  FLOP roofline **before step 0**, into `runs/tracking.db`.
- Evaluation batches come from an RNG seeded independently of the run seed, so
  every run is scored on identical held-out windows.
- Configs are hashed; cosmetic fields are excluded from the fingerprint.
- The dashboard only reads the DB, over a read-only connection.

## Honest limitations

- **Narrow domain.** Trained on synthetic children's stories. Outside that it
  produces fluent nonsense, and it has no factual grounding.
- **Under-trained for its size** — below the Chinchilla-optimal token budget; the
  ratio is printed before every run and shown on the Training tab.
- **AutoResearch conclusions are scale-local.** Stability techniques cannot show
  a benefit where there is no instability. "Did not clear the bar here" ≠ "does
  not work."
- **Greedy search misses interactions**, and the accepted order is a search-path
  artifact. The search was also budget-limited (see above).
- **No safety tuning.** SFT only. The server binds localhost with no auth.

## Layout

```
slm/        model, tokenizer, data, training, SFT, evaluation, AutoResearch, registry
app/        FastAPI dashboard + chat (server.py, static/)
docs/       CRISP-DM phase documentation + AutoResearch method
configs/    run configurations (baseline_2019, base, proxy, champion)
scripts/    data download, champion scale-up, pipeline chain
tests/      33 tests — causal mask, KV-cache equivalence, loss mask, dedup recall
runs/       tracking.db, checkpoints, per-run artifacts
```
