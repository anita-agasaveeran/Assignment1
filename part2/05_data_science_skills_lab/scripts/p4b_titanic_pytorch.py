"""CRISP-DM Phase 4 - Modeling (Track A: neural net + diagnosis).

Skills demonstrated:
  * pytorch-training-loop -- the canonical loop: train()/eval() modes, zero_grad,
                             no_grad validation, grad clipping, best-checkpointing,
                             .item() logging, deterministic seeding
  * ml-debugging          -- overfit-one-batch test, LR sweep to NaN, and a real
                             leakage hunt (inject a leak, watch it get caught)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from torch.utils.data import DataLoader, TensorDataset

from common import FIG, MOD, PALETTE, PROC, SEED, banner, seed_everything, style_plots, write_json

TARGET = "Survived"
NUM = ["Age", "SibSp", "Parch", "Fare", "FarePerPerson", "FareLog", "FamilySize",
       "AgeBand", "TicketPrefix_te", "Pclass"]
CAT = ["Sex", "Embarked", "Title", "Deck"]
BIN = ["Age_was_missing", "CabinKnown", "IsAlone"]
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, d_in: int, hidden: int = 64, p_drop: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_prep():
    return ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), NUM),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("o", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT),
        ("bin", "passthrough", BIN),
    ])


def to_tensors(prep, X_tr, y_tr, X_va, y_va):
    A = prep.fit_transform(X_tr).astype("float32")
    B = prep.transform(X_va).astype("float32")
    return (torch.tensor(A), torch.tensor(y_tr.to_numpy(), dtype=torch.float32),
            torch.tensor(B), torch.tensor(y_va.to_numpy(), dtype=torch.float32))


# --------------------------------------------------------------- ml-debugging #1
def overfit_one_batch(d_in: int, xb: torch.Tensor, yb: torch.Tensor) -> float:
    """The fastest sanity check in ML: can the wiring memorise a single batch?"""
    banner("ml-debugging", "Phase 4 - Modeling", "first move - can it overfit ONE batch?")
    torch.manual_seed(SEED)
    m = MLP(d_in, p_drop=0.0).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    crit = nn.BCEWithLogitsLoss()
    m.train()
    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
    first = last = None
    for i in range(300):
        opt.zero_grad(set_to_none=True)
        loss = crit(m(xb), yb)
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
        last = loss.item()
    print(f"  batch of {len(yb)}: loss {first:.4f} -> {last:.6f} over 300 steps")
    if last < 0.05:
        print("  -> PASS. Model/loss/label wiring is correct; any failure to learn the full")
        print("     dataset is an optimisation or regularisation problem, not a plumbing bug.")
    else:
        print("  -> FAIL. Stop and check model/loss/label wiring before touching hyperparameters.")
    return last


# --------------------------------------------------------------- ml-debugging #2
def lr_sweep(d_in, Xtr, ytr) -> list[dict]:
    """'Loss = NaN' is almost always LR. Sweep it on a log scale and show the cliff."""
    banner("ml-debugging", "Phase 4 - Modeling", "symptom 'loss = NaN' -> sweep LR on a log scale")
    rows = []
    crit = nn.BCEWithLogitsLoss()
    for lr in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e3, 1e5]:
        torch.manual_seed(SEED)
        m = MLP(d_in).to(DEVICE)
        opt = torch.optim.SGD(m.parameters(), lr=lr)
        m.train()
        xb, yb = Xtr[:256].to(DEVICE), ytr[:256].to(DEVICE)
        losses = []
        for _ in range(60):
            opt.zero_grad(set_to_none=True)
            loss = crit(m(xb), yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())          # .item() -- never accumulate the tensor
        final = losses[-1]
        if not np.isfinite(final):
            state = "NaN"
        elif final > losses[0]:
            state = "DIVERGED (loss above its start)"
        elif abs(losses[0] - final) < 0.02:
            state = "flat (LR too low)"
        else:
            state = "learning"
        rows.append({"lr": lr, "final_loss": final, "state": state})
        print(f"  lr={lr:<9} final loss {final:>14.4f}   {state}")
    ok = [r["lr"] for r in rows if r["state"] == "learning"]
    bad = [r["lr"] for r in rows if r["state"].startswith(("NaN", "DIVERGED"))]
    print(f"  -> usable band: {ok}")
    print(f"  -> diverges at: {bad}  <- this is the cliff behind almost every 'loss = NaN' report")
    return rows


# --------------------------------------------------------------- ml-debugging #3
def leakage_hunt(train: pd.DataFrame) -> dict:
    """'Metric implausibly high' -> hunt the leak. We inject one, then catch it."""
    banner("ml-debugging", "Phase 4 - Modeling",
           "symptom 'metric is implausibly high' -> run the leakage checklist")
    X, y = train.drop(columns=[TARGET]), train[TARGET]
    clean_cv = cross_val_score(
        Pipeline([("p", make_prep()), ("c", HistGradientBoostingClassifier(random_state=SEED))]),
        X, y, cv=5, scoring="roc_auc").mean()

    # A post-outcome column: 'was a survivor manifest entry filed?' -- known only AFTER the outcome.
    rng = np.random.default_rng(SEED)
    leaked = X.copy()
    leaked["manifest_filed"] = np.where(rng.random(len(y)) < 0.93, y, 1 - y)
    num2 = NUM + ["manifest_filed"]
    prep2 = ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num2),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("o", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT),
        ("bin", "passthrough", BIN),
    ])
    leaked_cv = cross_val_score(
        Pipeline([("p", prep2), ("c", HistGradientBoostingClassifier(random_state=SEED))]),
        leaked, y, cv=5, scoring="roc_auc").mean()

    print(f"  clean feature set   ROC-AUC {clean_cv:.4f}")
    print(f"  with 'manifest_filed' ROC-AUC {leaked_cv:.4f}   <- implausible jump of "
          f"{leaked_cv - clean_cv:+.4f}")
    print("\n  running the skill's 5-point leakage checklist:")
    corr = leaked.select_dtypes("number").corrwith(y).abs().sort_values(ascending=False)
    top = corr.index[0]
    print(f"    1. |corr| ~ 1.0 with target?      -> '{top}' at {corr.iloc[0]:.3f}  ** FLAGGED **")
    print(f"    2. preprocessing fit before split? -> no, it is inside the Pipeline")
    print(f"    3. one entity spanning train/test? -> no, one row per passenger")
    print(f"    4. time series shuffled?           -> N/A, not temporal")
    print(f"    5. post-outcome column used?       -> YES: 'manifest_filed' is recorded after")
    print(f"       the outcome is known. It cannot exist at prediction time.")
    print(f"  -> root cause found. Drop the column; the honest number is {clean_cv:.4f}.")
    return {"clean_roc_auc": float(clean_cv), "leaked_roc_auc": float(leaked_cv),
            "inflation": float(leaked_cv - clean_cv), "flagged_column": str(top),
            "flagged_corr": float(corr.iloc[0])}


# ---------------------------------------------------------- pytorch-training-loop
def train_loop(d_in, Xtr, ytr, Xva, yva, epochs: int = 60) -> dict:
    banner("pytorch-training-loop", "Phase 4 - Modeling",
           f"the canonical loop, device={DEVICE}")
    torch.manual_seed(SEED)
    model = MLP(d_in).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.BCEWithLogitsLoss()
    # AMP only helps on CUDA; enabled=False keeps the identical code path on CPU/MPS.
    use_amp = DEVICE == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    train_dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True,
                          generator=torch.Generator().manual_seed(SEED))
    val_dl = DataLoader(TensorDataset(Xva, yva), batch_size=256, shuffle=False)

    best_val, best_epoch, hist = float("inf"), -1, []
    ckpt = MOD / "titanic_mlp_best.pt"

    for epoch in range(epochs):
        # ---- TRAIN ----
        model.train()                                    # rule 1: dropout/BN ON
        tr_loss, n = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)              # rule 2: every step
            with torch.amp.autocast(device_type=DEVICE if use_amp else "cpu", enabled=use_amp):
                loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # rule 5
            scaler.step(opt)
            scaler.update()
            tr_loss += loss.item() * xb.size(0)          # rule 4: .item(), not the tensor
            n += xb.size(0)
        tr_loss /= n
        sched.step()

        # ---- VALIDATE ----
        model.eval()                                     # rule 1: dropout/BN OFF
        va_loss, m, probs = 0.0, 0, []
        with torch.no_grad():                            # rule 3: no graph, no memory growth
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                va_loss += crit(out, yb).item() * xb.size(0)
                m += xb.size(0)
                probs.append(torch.sigmoid(out).cpu().numpy())
        va_loss /= m
        pr = average_precision_score(yva.numpy(), np.concatenate(probs))
        hist.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, "val_pr_auc": pr})

        # ---- CHECKPOINT BEST ----
        if va_loss < best_val:
            best_val, best_epoch = va_loss, epoch
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                        "epoch": epoch, "val_loss": va_loss, "d_in": d_in}, ckpt)
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:>3}  train {tr_loss:.4f}  val {va_loss:.4f}  "
                  f"val PR-AUC {pr:.4f}{'   <- best' if epoch == best_epoch else ''}")

    print(f"\n  best epoch {best_epoch} (val loss {best_val:.4f}) checkpointed to {ckpt.name}")

    # Demonstrate why rule 1 matters: score the SAME weights with dropout/BN left on.
    state = torch.load(ckpt, weights_only=False)
    model.load_state_dict(state["model"])
    Xva_d = Xva.to(DEVICE)
    model.eval()
    with torch.no_grad():
        p_eval = torch.sigmoid(model(Xva_d)).cpu().numpy()
    model.train()                                        # the bug: forgot .eval()
    with torch.no_grad():
        p_train_mode = torch.sigmoid(model(Xva_d)).cpu().numpy()
    pr_eval = average_precision_score(yva.numpy(), p_eval)
    pr_bug = average_precision_score(yva.numpy(), p_train_mode)
    print(f"\n  same weights, scored correctly with model.eval() : PR-AUC {pr_eval:.4f}")
    print(f"  same weights, scored with model.train() (the bug) : PR-AUC {pr_bug:.4f}")
    print(f"  -> forgetting .eval() costs {pr_eval - pr_bug:+.4f} PR-AUC silently. No error is raised.")

    return {"history": hist, "best_epoch": best_epoch, "best_val_loss": best_val,
            "val_pr_auc_eval_mode": float(pr_eval), "val_pr_auc_train_mode_bug": float(pr_bug),
            "device": DEVICE}


def main() -> None:
    seed_everything()
    print(f"  torch {torch.__version__}, device {DEVICE}, deterministic seeding applied")
    train = pd.read_csv(PROC / "titanic_train.csv")
    test = pd.read_csv(PROC / "titanic_test.csv")

    X, y = train.drop(columns=[TARGET]), train[TARGET]
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    prep = make_prep()
    Xtr, ytr, Xva, yva = to_tensors(prep, X_tr, y_tr, X_va, y_va)
    d_in = Xtr.shape[1]
    print(f"  encoded feature width: {d_in}")

    one_batch = overfit_one_batch(d_in, Xtr[:32], ytr[:32])
    sweep = lr_sweep(d_in, Xtr, ytr)
    hist = train_loop(d_in, Xtr, ytr, Xva, yva)
    leak = leakage_hunt(train)

    # honest comparison against the Phase 4a gradient-boosted pipeline
    banner("model-evaluation", "Phase 4 - Modeling", "does the neural net earn its complexity?")
    Xte = torch.tensor(prep.transform(test.drop(columns=[TARGET])).astype("float32"))
    yte = test[TARGET].to_numpy()
    model = MLP(d_in).to(DEVICE)
    model.load_state_dict(torch.load(MOD / "titanic_mlp_best.pt", weights_only=False)["model"])
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(Xte.to(DEVICE))).cpu().numpy()
    mlp_pr, mlp_roc = average_precision_score(yte, p), roc_auc_score(yte, p)
    import json

    hgb = json.loads((Path(__file__).parents[1] / "outputs/tables/p4a_titanic_model_results.json").read_text())
    print(f"  MLP  holdout PR-AUC {mlp_pr:.4f}  ROC-AUC {mlp_roc:.4f}")
    print(f"  HGB  holdout PR-AUC {hgb['holdout']['pr_auc']:.4f}  "
          f"ROC-AUC {hgb['holdout']['roc_auc']:.4f}")
    better = "MLP" if mlp_pr > hgb["holdout"]["pr_auc"] else "HGB"
    print(f"  -> {better} wins on PR-AUC. On 712 training rows of tabular data a gradient-boosted")
    print(f"     tree is the right default; the MLP is here to demonstrate the training loop,")
    print(f"     not because deep learning suits this dataset.")

    write_json({"one_batch_final_loss": one_batch, "lr_sweep": sweep,
                "training": hist, "leakage_hunt": leak,
                "mlp_holdout": {"pr_auc": float(mlp_pr), "roc_auc": float(mlp_roc)},
                "hgb_holdout": hgb["holdout"], "winner": better},
               "p4b_pytorch_debug_results.json")

    import matplotlib.pyplot as plt

    style_plots()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    h = pd.DataFrame(hist["history"])
    ax = axes[0]
    ax.plot(h["epoch"], h["train_loss"], label="train", color=PALETTE[0])
    ax.plot(h["epoch"], h["val_loss"], label="val", color=PALETTE[1])
    ax.axvline(hist["best_epoch"], ls="--", c="grey", lw=1)
    ax.text(hist["best_epoch"] + 1, ax.get_ylim()[1] * 0.95, "checkpoint", fontsize=8, color="grey")
    ax.set_title("Curves stay together - no overfit")
    ax.set_xlabel("epoch"); ax.set_ylabel("BCE loss"); ax.legend(frameon=False)

    ax = axes[1]
    lrs = [r["lr"] for r in sweep]
    fl = [r["final_loss"] if np.isfinite(r["final_loss"]) else np.nan for r in sweep]
    ax.plot(lrs, fl, "o-", color=PALETTE[2])
    ax.set_xscale("log"); ax.set_title("LR sweep finds the usable band")
    ax.set_xlabel("learning rate"); ax.set_ylabel("loss after 60 steps")

    ax = axes[2]
    ax.bar(["clean", "with leaked\ncolumn"], [leak["clean_roc_auc"], leak["leaked_roc_auc"]],
           color=[PALETTE[0], PALETTE[1]])
    ax.set_ylim(0.5, 1.0); ax.set_ylabel("CV ROC-AUC")
    ax.set_title(f"Leakage inflates ROC-AUC by {leak['inflation']:+.3f}")
    for i, v in enumerate([leak["clean_roc_auc"], leak["leaked_roc_auc"]]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")

    fig.suptitle("Titanic - PyTorch training loop and ML debugging", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "p4b_pytorch_debug.png")
    plt.close(fig)
    print("  -> wrote p4b_pytorch_debug.png")


if __name__ == "__main__":
    main()
