# Phase 4 — Modeling

## Architecture

Decoder-only transformer (`slm/model.py`). Every primitive is a **config flag**,
not a hard-coded choice, because AutoResearch must ablate each independently.

| Component | Default | Source |
|---|---|---|
| Normalization | RMSNorm, pre-norm | arXiv:1910.07467, arXiv:2002.04745 |
| Position | Rotary (RoPE), θ=10,000 | arXiv:2104.09864 |
| Feed-forward | SwiGLU, d_ff = 8/3·d rounded to 64 | arXiv:2002.05202 |
| Attention | Grouped-query, 6 heads / 2 KV | arXiv:2305.13245 |
| Attention kernel | `scaled_dot_product_attention` | arXiv:2205.14135 |
| QK normalization | RMSNorm on q,k per head | arXiv:2010.04245, arXiv:2302.05442 |
| Embeddings | Tied input/output | arXiv:1608.05859 |
| Init | N(0, 0.02); residual projections ÷ √(2L) | GPT-2 |
| Objective | Cross-entropy (+ optional z-loss, logit soft-cap) | arXiv:2204.02311, arXiv:2408.00118 |

Implementation notes worth knowing:

- **RMSNorm computes in fp32** and casts back. Under bf16 autocast the reduction
  otherwise loses enough precision to matter.
- **GQA** expands KV heads with `repeat_interleave` rather than SDPA's
  `enable_gqa`, for MPS backend compatibility.
- **Attention soft-capping forces the manual attention path** (you cannot cap
  logits inside a fused kernel), so it costs both memory and speed. That cost is
  visible in the AutoResearch throughput column, which is the point.
- `d_ff = 8/3·d` for SwiGLU keeps parameter count matched to a 4·d GELU block, so
  the SwiGLU-vs-GELU comparison is parameter-matched rather than confounded.

## Optimization

AdamW (arXiv:1711.05101) with the standard group split: **weight decay on 2-D
matrices only**, never on norms, biases, or embeddings.

**Muon** (arXiv:2502.16982; Jordan et al. 2024) is available as a candidate. It
replaces the momentum direction with its nearest orthogonal matrix via a
five-step Newton–Schulz iteration — matmuls only, so it runs on MPS with no
custom kernels. It is applied to 2-D hidden weights only; embeddings, LM head,
norms and biases stay on AdamW, as the papers prescribe.

Schedules: cosine (arXiv:1608.03983), Warmup-Stable-Decay (arXiv:2404.06395),
linear, constant — all with linear warmup. Gradient clipping at 1.0
(arXiv:1211.5063). bf16 autocast (arXiv:1710.03740).

## Test design

Three deliberate choices make comparisons trustworthy:

1. **Fixed evaluation batches.** Drawn from an RNG seeded with a constant that is
   independent of the run seed, so every run — every trial, every seed — is
   scored on *identical* held-out windows. Without this, a 0.01 loss difference is
   indistinguishable from batch luck.
2. **Paired trials.** Within an AutoResearch round, the incumbent is re-measured
   at that round's seed and every candidate uses the same seed. The variance being
   tested against is therefore seed variance, not batch variance.
3. **Full provenance.** Config, environment, measured FLOP roofline and git state
   are written before step 0, so any dashboard row is re-runnable from the DB.

### On MFU

Model FLOPs Utilization is reported against an **empirically measured** dense
matmul roofline (~7.5 TFLOP/s bf16 on this machine), not a vendor peak. Apple
does not publish a number that any real kernel achieves, and measuring gives a
stricter, more interpretable denominator. Steady-state MFU is ~25%; the first
few steps read much lower purely from warm-up, which is why throughput is
measured after warm-up.

FLOPs per token uses the Kaplan/PaLM convention: `6N + 12·L·T·d`, where N counts
non-embedding matmul parameters plus the output head.

## AutoResearch

See `docs/autoresearch.md` for the search procedure, acceptance rule, and the
statistical reasoning behind the noise floor.
