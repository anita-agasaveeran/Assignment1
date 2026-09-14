"""Optimizers and learning-rate schedules.

AdamW is the default (Loshchilov & Hutter, arXiv:1711.05101), with the standard
parameter-group split: weight decay applies to 2-D weight matrices only, never
to norms, biases, or embeddings -- decaying a normalization gain or an embedding
row is a different (and usually harmful) intervention than decaying a matrix.

Muon (Jordan et al. 2024; Liu et al. 2025, arXiv:2502.16982) is offered as an
AutoResearch candidate. It replaces the momentum direction with its nearest
orthogonal matrix, computed by a Newton-Schulz iteration that needs only matmuls
-- which is why it runs fine on an MPS backend with no custom kernels. Muon is
applied only to 2-D hidden weights; embeddings, the LM head, norms and biases
stay on AdamW, exactly as the papers prescribe.
"""
from __future__ import annotations

import math
from typing import Any, Iterable

import torch
from torch import Tensor

from .config import TrainConfig


# --------------------------------------------------------------------- Muon
@torch.no_grad()
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Approximate the orthogonal polar factor of G using a quintic iteration.

    The coefficients are tuned so the iteration converges fast in the region the
    spectrum lands in after normalization; it is deliberately *not* exact --
    Muon only needs the update direction, not a precise polar decomposition.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16() if G.device.type != "cpu" else G.float()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Momentum-Orthogonalized-by-Newton-Schulz, for 2-D hidden weights only."""

    def __init__(self, params: Iterable[Tensor], lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5, weight_decay: float = 0.0):
        super().__init__(list(params), dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                            ns_steps=ns_steps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, mom = group["lr"], group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim != 2:
                    raise ValueError("Muon received a non-2D parameter; route it to AdamW")
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(g)
                buf = st["momentum_buffer"]
                buf.mul_(mom).add_(g)
                d = g.add(buf, alpha=mom) if group["nesterov"] else buf
                u = zeropower_via_newtonschulz5(d, group["ns_steps"])
                # Scale so the update RMS is comparable across differently shaped
                # matrices (an orthogonal matrix has unit singular values).
                scale = math.sqrt(max(1.0, p.size(-2) / p.size(-1)))
                if group["weight_decay"]:
                    p.mul_(1 - lr * group["weight_decay"])
                p.add_(u, alpha=-lr * scale)
        return loss


# ------------------------------------------------------------------ bundle
class OptimizerBundle:
    """One handle over possibly-two optimizers, with a shared LR multiplier."""

    def __init__(self, opts: list[torch.optim.Optimizer], base_lrs: list[list[float]],
                 names: list[str]):
        self.opts, self.base_lrs, self.names = opts, base_lrs, names

    def zero_grad(self, set_to_none: bool = True) -> None:
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for o in self.opts:
            o.step()

    def set_lr_scale(self, scale: float) -> None:
        for o, bases in zip(self.opts, self.base_lrs):
            for g, base in zip(o.param_groups, bases):
                g["lr"] = base * scale

    def current_lrs(self) -> dict[str, float]:
        return {n: o.param_groups[0]["lr"] for n, o in zip(self.names, self.opts)}

    def state_dict(self) -> list[dict]:
        return [o.state_dict() for o in self.opts]

    def load_state_dict(self, sds: list[dict]) -> None:
        for o, sd in zip(self.opts, sds):
            o.load_state_dict(sd)


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig,
                    verbose: bool = True) -> tuple[OptimizerBundle, dict[str, Any]]:
    decay, no_decay, muon_params = [], [], []
    names_muon: list[str] = []

    hidden_2d = set()
    if cfg.optimizer == "muon":
        for name, p in model.named_parameters():
            if p.ndim == 2 and not name.startswith(("tok_emb", "pos_emb", "lm_head")):
                hidden_2d.add(name)

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name in hidden_2d:
            muon_params.append(p); names_muon.append(name)
        elif p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)

    groups = [dict(params=decay, weight_decay=cfg.weight_decay),
              dict(params=no_decay, weight_decay=0.0)]
    groups = [g for g in groups if g["params"]]
    adamw = torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps)
    opts, bases, names = [adamw], [[cfg.lr] * len(adamw.param_groups)], ["adamw"]

    if muon_params:
        m = Muon(muon_params, lr=cfg.muon_lr, momentum=0.95, nesterov=True,
                 weight_decay=cfg.weight_decay)
        opts.append(m); bases.append([cfg.muon_lr]); names.append("muon")

    info = dict(
        optimizer=cfg.optimizer,
        n_decay=sum(p.numel() for p in decay), n_no_decay=sum(p.numel() for p in no_decay),
        n_muon=sum(p.numel() for p in muon_params), n_muon_tensors=len(muon_params),
        muon_param_names=names_muon[:12],
    )
    if verbose:
        print(f"[optim] {cfg.optimizer}: decay={info['n_decay']/1e6:.2f}M "
              f"no_decay={info['n_no_decay']/1e3:.1f}K muon={info['n_muon']/1e6:.2f}M "
              f"({info['n_muon_tensors']} tensors)")
    return OptimizerBundle(opts, bases, names), info


# --------------------------------------------------------------- schedules
def lr_scale(step: int, cfg: TrainConfig) -> float:
    """Multiplier on the peak LR at `step` (0-indexed)."""
    warm, total = cfg.warmup_steps, cfg.steps
    floor = cfg.min_lr_ratio
    if warm > 0 and step < warm:
        return (step + 1) / warm
    if cfg.schedule == "constant":
        return 1.0
    prog = (step - warm) / max(1, total - warm)
    prog = min(max(prog, 0.0), 1.0)
    if cfg.schedule == "cosine":
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * prog))
    if cfg.schedule == "linear":
        return floor + (1 - floor) * (1 - prog)
    if cfg.schedule == "wsd":
        # Warmup-Stable-Decay: hold peak, then decay only over the final fraction
        d = cfg.wsd_decay_frac
        if prog < 1 - d:
            return 1.0
        local = (prog - (1 - d)) / max(d, 1e-9)
        return floor + (1 - floor) * (1 - local)
    raise ValueError(f"unknown schedule {cfg.schedule!r}")


def clip_grad_norm(model: torch.nn.Module, max_norm: float) -> float:
    """Pascanu et al. 2012 (arXiv:1211.5063). Returns the pre-clip norm."""
    if max_norm <= 0:
        total = torch.sqrt(sum((p.grad.detach() ** 2).sum()
                               for p in model.parameters() if p.grad is not None))
        return float(total)
    return float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm))
