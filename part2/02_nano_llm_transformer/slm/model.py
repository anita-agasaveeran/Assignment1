"""Decoder-only transformer with switchable modern primitives.

Default configuration is the post-LLaMA recipe:
  pre-norm RMSNorm         Zhang & Sennrich 2019, arXiv:1910.07467
  rotary position emb      Su et al. 2021,        arXiv:2104.09864
  SwiGLU feed-forward      Shazeer 2020,          arXiv:2002.05202
  grouped-query attention  Ainslie et al. 2023,   arXiv:2305.13245
  QK-normalization         Henry et al. 2020,     arXiv:2010.04245
  tied embeddings          Press & Wolf 2016,     arXiv:1608.05859
  softmax z-loss           Chowdhery et al. 2022, arXiv:2204.02311
  tanh logit soft-cap      Gemma Team 2024,       arXiv:2408.00118

Every one of those is a config flag rather than a hard-coded choice, because
AutoResearch needs to ablate each independently. Attention runs through
scaled_dot_product_attention so it uses the fused/flash kernel where the backend
provides one (Dao et al. 2022, arXiv:2205.14135).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


# ------------------------------------------------------------------ norms
class RMSNorm(nn.Module):
    """x / rms(x) * g. No mean subtraction, no bias (arXiv:1910.07467)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()                                   # normalize in fp32 for stability
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def make_norm(cfg: ModelConfig, dim: int) -> nn.Module:
    if cfg.norm == "rmsnorm":
        return RMSNorm(dim, cfg.norm_eps)
    return nn.LayerNorm(dim, eps=cfg.norm_eps, bias=cfg.bias)


# ------------------------------------------------------------------- rope
def build_rope_cache(head_dim: int, max_seq: int, theta: float,
                     device=None, dtype=torch.float32):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq, device=device).float()
    freqs = torch.outer(t, inv)                          # [T, hd/2]
    return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, H, T, hd]. Rotates (x1, x2) pairs by the position angle."""
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# -------------------------------------------------------------- kv caching
@dataclass
class KVCache:
    k: torch.Tensor              # [B, n_kv_head, max_seq, head_dim]
    v: torch.Tensor
    length: int = 0

    @classmethod
    def alloc(cls, cfg: ModelConfig, batch: int, max_seq: int, device, dtype) -> "KVCache":
        shape = (batch, cfg.n_kv_head, max_seq, cfg.head_dim)
        return cls(torch.zeros(shape, device=device, dtype=dtype),
                   torch.zeros(shape, device=device, dtype=dtype), 0)

    def bytes(self) -> int:
        return 2 * self.k.numel() * self.k.element_size()


# -------------------------------------------------------------- attention
class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.n_head, self.n_kv_head, self.hd = cfg.n_head, cfg.n_kv_head, cfg.head_dim
        self.n_rep = cfg.n_head // cfg.n_kv_head
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_head * cfg.head_dim, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_head * cfg.head_dim, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_head * cfg.head_dim, bias=cfg.bias)
        self.o_proj = nn.Linear(cfg.n_head * cfg.head_dim, cfg.d_model, bias=cfg.bias)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.norm_eps) if cfg.qk_norm else None
        self.k_norm = RMSNorm(cfg.head_dim, cfg.norm_eps) if cfg.qk_norm else None
        self.drop_p = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.last_logit_max: float = 0.0        # diagnostic, see evaluate.py

    def forward(self, x, cos=None, sin=None, cache: KVCache | None = None,
                start_pos: int = 0, collect_stats: bool = False):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.hd).transpose(1, 2)

        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        if cos is not None:
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        if cache is not None:
            cache.k[:B, :, start_pos:start_pos + T] = k
            cache.v[:B, :, start_pos:start_pos + T] = v
            cache.length = start_pos + T
            k = cache.k[:B, :, :cache.length]
            v = cache.v[:B, :, :cache.length]

        if self.n_rep > 1:                                # GQA: broadcast kv groups
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        if self.cfg.attn_soft_cap > 0 or collect_stats:
            # manual path: needed to soft-cap or inspect attention logits
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
            if collect_stats:
                self.last_logit_max = float(scores.detach().abs().max())
            cap = self.cfg.attn_soft_cap
            if cap > 0:
                scores = cap * torch.tanh(scores / cap)
            Tq, Tk = scores.shape[-2], scores.shape[-1]
            mask = torch.ones(Tq, Tk, dtype=torch.bool, device=x.device).tril(Tk - Tq)
            scores = scores.masked_fill(~mask, float("-inf"))
            att = F.softmax(scores.float(), dim=-1).to(q.dtype)
            if self.drop_p and self.training:
                att = F.dropout(att, self.drop_p)
            y = att @ v
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=(cache is None or T > 1),
                dropout_p=self.drop_p if self.training else 0.0)

        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.hd)
        return self.resid_drop(self.o_proj(y))


# ------------------------------------------------------------------- mlp
class SwiGLU(nn.Module):
    """FFN(x) = W_down( SiLU(W_gate x) * W_up x )  -- arXiv:2002.05202."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class GeluMLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.down(F.gelu(self.up(x), approximate="tanh")))


# ----------------------------------------------------------------- block
class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.norm1 = make_norm(cfg, cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = make_norm(cfg, cfg.d_model)
        self.mlp = SwiGLU(cfg) if cfg.ffn == "swiglu" else GeluMLP(cfg)

    def forward(self, x, cos=None, sin=None, cache=None, start_pos=0, collect_stats=False):
        if self.cfg.norm_placement == "pre":
            x = x + self.attn(self.norm1(x), cos, sin, cache, start_pos, collect_stats)
            x = x + self.mlp(self.norm2(x))
        else:                                   # post-norm (original arXiv:1706.03762)
            x = self.norm1(x + self.attn(x, cos, sin, cache, start_pos, collect_stats))
            x = self.norm2(x + self.mlp(x))
        return x


# ----------------------------------------------------------------- model
class SLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model) if cfg.pos == "learned" else None
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm_f = make_norm(cfg, cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        if cfg.pos == "rope":
            cos, sin = build_rope_cache(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        if cfg.scale_resid_init:
            # GPT-2 trick: shrink residual-branch outputs so activation variance
            # does not grow with depth.
            std = cfg.init_std / math.sqrt(2 * cfg.n_layer)
            for blk in self.blocks:
                nn.init.normal_(blk.attn.o_proj.weight, mean=0.0, std=std)
                nn.init.normal_(blk.mlp.down.weight, mean=0.0, std=std)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)

    # ------------------------------------------------------------- forward
    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                cache: list[KVCache] | None = None, start_pos: int = 0,
                loss_mask: torch.Tensor | None = None, collect_stats: bool = False,
                label_smoothing: float = 0.0):
        B, T = idx.shape
        cfg = self.cfg
        if start_pos + T > cfg.max_seq_len:
            raise ValueError(f"sequence {start_pos + T} exceeds max_seq_len {cfg.max_seq_len}")

        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            pos = torch.arange(start_pos, start_pos + T, device=idx.device)
            x = x + self.pos_emb(pos)[None]
        x = self.drop(x)

        cos = sin = None
        if cfg.pos == "rope":
            cos = self.rope_cos[start_pos:start_pos + T]
            sin = self.rope_sin[start_pos:start_pos + T]

        for i, blk in enumerate(self.blocks):
            x = blk(x, cos, sin, cache[i] if cache else None, start_pos, collect_stats)
        x = self.norm_f(x)

        logits = self.lm_head(x)
        if cfg.logit_soft_cap > 0:
            logits = cfg.logit_soft_cap * torch.tanh(logits / cfg.logit_soft_cap)

        out: dict[str, torch.Tensor] = {"logits": logits}
        if targets is None:
            return out

        fl = logits.float()
        flat_logits = fl.view(-1, fl.size(-1))
        flat_targets = targets.reshape(-1)
        ce = F.cross_entropy(flat_logits, flat_targets, reduction="none",
                             ignore_index=-100, label_smoothing=label_smoothing)
        if loss_mask is not None:
            m = loss_mask.reshape(-1).float()
            denom = m.sum().clamp(min=1.0)
            ce_mean = (ce * m).sum() / denom
        else:
            valid = (flat_targets != -100).float()
            ce_mean = (ce * valid).sum() / valid.sum().clamp(min=1.0)

        loss = ce_mean
        out["ce"] = ce_mean
        if cfg.z_loss > 0:
            # keeps log Z near 0 so the softmax normalizer cannot drift (PaLM)
            lse = torch.logsumexp(flat_logits, dim=-1)
            zl = cfg.z_loss * (lse ** 2).mean()
            loss = loss + zl
            out["z"] = zl
        out["loss"] = loss
        return out

    # --------------------------------------------------------- introspection
    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
            if self.pos_emb is not None:
                n -= self.pos_emb.weight.numel()
            if not self.cfg.tie_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    def param_breakdown(self) -> dict[str, int]:
        g: dict[str, int] = {}
        for name, p in self.named_parameters():
            if name.startswith("tok_emb"): k = "token_embedding"
            elif name.startswith("pos_emb"): k = "position_embedding"
            elif name.startswith("lm_head"): k = "lm_head"
            elif ".attn.q_proj" in name or ".attn.k_proj" in name or ".attn.v_proj" in name: k = "attn_qkv"
            elif ".attn.o_proj" in name: k = "attn_out"
            elif ".attn." in name: k = "attn_qk_norm"
            elif ".mlp." in name: k = "mlp"
            elif "norm" in name: k = "norms"
            else: k = "other"
            g[k] = g.get(k, 0) + p.numel()
        if self.cfg.tie_embeddings:
            g.pop("lm_head", None)          # shared storage, do not double count
        return dict(sorted(g.items(), key=lambda kv: -kv[1]))

    def flops_per_token(self, seq_len: int | None = None) -> float:
        """Forward+backward FLOPs per token (Kaplan/PaLM convention: 6N + attention)."""
        T = seq_len or self.cfg.max_seq_len
        cfg = self.cfg
        # non-embedding weights + the output head (a real matmul whether or not
        # its storage is shared with the input embedding)
        n_matmul = self.num_params(non_embedding=True) + cfg.d_model * cfg.vocab_size
        return 6 * n_matmul + 12 * cfg.n_layer * T * cfg.d_model

    def kv_cache_bytes(self, batch: int, seq_len: int, dtype_bytes: int = 2) -> int:
        cfg = self.cfg
        return 2 * batch * cfg.n_layer * cfg.n_kv_head * seq_len * cfg.head_dim * dtype_bytes

    # -------------------------------------------------------------- generate
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8,
                 top_k: int = 0, top_p: float = 0.9, repetition_penalty: float = 1.0,
                 stop_ids: tuple[int, ...] = (), use_cache: bool = True,
                 min_new_tokens: int = 0):
        """Sampling with a KV cache. top-p is Holtzman et al. 2019 (arXiv:1904.09751)."""
        self.eval()
        B = idx.size(0)
        device = idx.device
        cache = None
        if use_cache:
            dtype = next(self.parameters()).dtype
            cache = [KVCache.alloc(self.cfg, B, self.cfg.max_seq_len, device, dtype)
                     for _ in range(self.cfg.n_layer)]

        generated = idx
        pos = 0
        cur = idx
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for step in range(max_new_tokens):
            if generated.size(1) >= self.cfg.max_seq_len:
                break
            out = self(cur, cache=cache, start_pos=pos)
            pos += cur.size(1)
            logits = out["logits"][:, -1, :].float()

            if repetition_penalty != 1.0:
                for b in range(B):
                    uniq = torch.unique(generated[b])
                    lg = logits[b, uniq]
                    logits[b, uniq] = torch.where(lg > 0, lg / repetition_penalty,
                                                  lg * repetition_penalty)
            if step < min_new_tokens:
                for s in stop_ids:
                    logits[:, s] = float("-inf")

            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                if 0 < top_p < 1.0:
                    srt, sidx = torch.sort(logits, descending=True, dim=-1)
                    probs = F.softmax(srt, dim=-1).cumsum(-1)
                    drop = probs - F.softmax(srt, dim=-1) >= top_p
                    srt = srt.masked_fill(drop, float("-inf"))
                    logits = torch.full_like(logits, float("-inf")).scatter(-1, sidx, srt)
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)

            if stop_ids:
                hit = torch.zeros_like(finished)
                for s in stop_ids:
                    hit |= nxt.squeeze(-1) == s
                finished |= hit
            generated = torch.cat([generated, nxt], dim=1)
            cur = nxt if use_cache else generated
            if use_cache is False:
                pos = 0
            if finished.all():
                break
        return generated


def build_model(cfg: ModelConfig) -> SLM:
    return SLM(cfg)
