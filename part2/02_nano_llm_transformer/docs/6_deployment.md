# Phase 6 — Deployment

## What ships

One FastAPI process (`app/server.py`) serving both the analyst dashboard and the
chat endpoint:

```bash
python -m uvicorn app.server:app --port 8099
```

The **same `LM` class** (`slm/infer.py`) backs the CLI, the offline evaluation
suite, and the web endpoint. This is deliberate: if evaluation used a different
generation path from serving, the offline numbers would describe something other
than what users get.

## Inference

- **KV cache** preallocated to the full context; decode is one token per forward
  pass. Greedy decoding with the cache is verified equal to greedy without it —
  a cache bug is otherwise silent and produces subtly worse text.
- Sampling: temperature, top-k, top-p (arXiv:1904.09751), repetition penalty
  over a 256-token window.
- **Incremental UTF-8 decoding.** A BPE token can end mid-character, so bytes are
  buffered until they decode; without this, streaming emits replacement
  characters on multi-byte sequences.
- Streaming over SSE, reporting time-to-first-token, decode tokens/s and KV cache
  size per response.
- Context overflow truncates from the left, keeping the most recent turns.

## Dashboard

Eight views mapped to the CRISP-DM phases. Reads `runs/tracking.db` (SQLite, WAL
mode) over a **read-only** connection, so opening the dashboard can never
interfere with a training run writing to the same file — the dashboard is usable
*while* training.

Charts are hand-written inline SVG with no CDN or chart library, so the whole
thing works offline. The palette is a validated colorblind-safe set, capped at
three categorical series per chart, with light and dark modes defined separately
rather than auto-inverted.

## Monitoring

Recorded per run: loss, learning rate, gradient norm, step time, tokens/s, MFU,
allocated and peak device memory, generalization gap, logit magnitude, softmax
log-Z, and periodic generation samples.

Signals that indicate real trouble:

| Signal | Meaning |
|---|---|
| Gradient-norm spike | Precedes divergence; lower LR or lengthen warmup |
| Rising `val_logit_absmax` | Logit drift; z-loss or soft-capping is the counter |
| Generalization gap widening | Over-fitting; more data or re-enable regularization |
| tokens/s falling mid-run | Thermal throttling or memory pressure, not a model change |
| Non-finite loss | Run marked `diverged` and stopped; recorded, not crashed |

## Maintenance and known operational limits

- **Checkpoints are `torch.save` pickles.** Loading uses
  `weights_only=False` because the config travels with the weights, so only load
  checkpoints you produced. Converting to safetensors is the correct hardening
  step and is not done here.
- **The server binds localhost and has no authentication.** It is a local
  analysis tool. Do not expose it to a network without putting auth in front.
- **Single-process, single-model.** Checkpoint swaps take a lock; concurrent
  chat requests serialize on the GPU. Batched serving and PagedAttention-style
  KV management (arXiv:2309.06180) are the scaling path, not implemented here.
- AutoResearch proxy trials write no checkpoints by design; only full runs do.

## Deployment risks

The model is a **9M-parameter story generator with no factual grounding, no
safety tuning, and no refusal behaviour**. It should never be presented as a
general assistant. The model card states the domain limits explicitly, and the
chat view says plainly when it is serving a raw pretrained checkpoint rather than
an instruction-tuned one — those behave very differently and users would
otherwise read continuation as a bug.
