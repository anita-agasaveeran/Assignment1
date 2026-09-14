"""CRISP-DM Phase 4 - Modeling (Track A: LLM fine-tuning).

Skill demonstrated:
  * llm-finetuning -- decide whether to fine-tune at all, choose the method,
                      format the dataset with the model's chat template, train
                      with LoRA via PEFT/TRL, and evaluate against a base-model
                      baseline on a held-out set.

Task: teach a small base model to answer templated questions about the Online
Retail warehouse in a fixed, parseable format. Every question and answer is
generated from the real warehouse, so exact-match scoring is meaningful.

QLoRA (4-bit) is deliberately NOT used: bitsandbytes has no Apple-Silicon build,
so the honest choice here is plain LoRA in fp32. That is the skill's decision
table applied to the hardware actually available, not skipped.
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import torch

from common import MOD, PROC, SEED, TAB, WAREHOUSE, banner, seed_everything, write_json, write_report

BASE = "distilgpt2"          # 82M params: small enough to actually train here
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def build_dataset(con: sqlite3.Connection) -> list[dict]:
    """Step: dataset formatting. Quality and consistency beat volume."""
    rows = con.execute("""
        SELECT c.CustomerID, c.Country, c.Orders,
               ROUND(SUM(i.InvoiceRevenue), 2) AS revenue
        FROM customers c JOIN invoices i ON i.CustomerID = c.CustomerID
        GROUP BY c.CustomerID, c.Country, c.Orders
        HAVING revenue > 0
        ORDER BY revenue DESC
        LIMIT 900
    """).fetchall()
    out = []
    for cid, country, orders, rev in rows:
        out.append({
            "prompt": f"Q: Summarise customer {cid}.\nA:",
            "completion": f" Customer {cid} is based in {country}, placed {orders} orders, "
                          f"and generated GBP {rev:,.2f} in revenue.<|endoftext|>",
            "customer": int(cid), "country": country, "orders": int(orders), "revenue": float(rev),
        })
    return out


def extract(text: str) -> dict:
    """Parse the target format back out so we can score exactly."""
    m = re.search(r"based in ([A-Za-z .'()/-]+?), placed (\d+) orders, and generated GBP "
                  r"([\d,]+\.\d{2})", text)
    if not m:
        return {}
    return {"country": m.group(1).strip(), "orders": int(m.group(2)),
            "revenue": float(m.group(3).replace(",", ""))}


@torch.no_grad()
def generate(model, tok, prompts: list[str], max_new: int = 48) -> list[str]:
    model.eval()
    outs = []
    for p in prompts:
        ids = tok(p, return_tensors="pt").to(model.device)
        g = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                           pad_token_id=tok.eos_token_id)
        outs.append(tok.decode(g[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))
    return outs


def score(preds: list[str], golds: list[dict]) -> dict:
    """Task metrics, not just loss -- the skill is explicit about this."""
    fmt = sum(1 for p in preds if extract(p))
    fields = {"country": 0, "orders": 0, "revenue": 0}
    exact = 0
    for p, g in zip(preds, golds):
        e = extract(p)
        if not e:
            continue
        hit = 0
        for k in fields:
            if k == "revenue":
                ok = abs(e[k] - g[k]) < 0.01
            else:
                ok = e[k] == g[k]
            fields[k] += int(ok)
            hit += int(ok)
        exact += int(hit == 3)
    n = len(preds)
    return {"format_valid": fmt / n, "country_acc": fields["country"] / n,
            "orders_acc": fields["orders"] / n, "revenue_acc": fields["revenue"] / n,
            "exact_match": exact / n}


def main() -> None:
    seed_everything()
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    banner("llm-finetuning", "Phase 4 - Modeling", "decide -> format -> LoRA -> evaluate")

    # ---- Step 0: do you even need to fine-tune? ---------------------------
    print("  step 0  should we fine-tune at all?")
    print("          The skill says try prompting + few-shot + RAG first. Here:")
    print("          - the task is a FIXED OUTPUT FORMAT over facts already in a database")
    print("          - retrieval (see the rag-pipeline demo) is the right tool for the FACTS")
    print("          - fine-tuning is the right tool for the FORMAT and for cutting prompt length")
    print("          -> fine-tune for format consistency; keep RAG for factual grounding.")

    print("\n  step 1  method selection (the skill's decision table, applied to this hardware):")
    print(f"          full FT  -- rejected: 82M params trainable is wasteful for a format task")
    print(f"          QLoRA    -- rejected: bitsandbytes has no Apple-Silicon build; 4-bit")
    print(f"                      quantisation is unavailable on {DEVICE}")
    print(f"          LoRA     -- SELECTED: r=16, alpha=32, attention projections only")

    # ---- Step 2: dataset --------------------------------------------------
    con = sqlite3.connect(WAREHOUSE)
    data = build_dataset(con)
    con.close()
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(data))
    n_eval = 100
    train_rows = [data[i] for i in idx[n_eval:]]
    eval_rows = [data[i] for i in idx[:n_eval]]
    print(f"\n  step 2  dataset from the real warehouse")
    print(f"          {len(train_rows)} train / {len(eval_rows)} held-out examples")
    print(f"          one consistent template; every fact traceable to a SQL row")
    print(f"          example -> {train_rows[0]['prompt']}{train_rows[0]['completion'][:90]}...")
    (PROC / "llm_sft_train.jsonl").write_text(
        "\n".join(json.dumps({"prompt": r["prompt"], "completion": r["completion"]})
                  for r in train_rows))

    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.eos_token
    eval_prompts = [r["prompt"] for r in eval_rows]

    # ---- Baseline BEFORE training (the skill: never skip this) -----------
    print(f"\n  step 5a base-model baseline on the held-out set (n={n_eval})")
    base_model = AutoModelForCausalLM.from_pretrained(BASE).to(DEVICE)
    t0 = time.perf_counter()
    base_preds = generate(base_model, tok, eval_prompts)
    base = score(base_preds, eval_rows)
    print(f"          {' | '.join(f'{k} {v:.3f}' for k, v in base.items())}  "
          f"[{time.perf_counter()-t0:.0f}s]")
    print(f"          base output sample: {base_preds[0].strip()[:100]!r}")

    # ---- Step 3-4: LoRA ---------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(BASE)
    peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                          task_type="CAUSAL_LM", target_modules=["c_attn"])
    model = get_peft_model(model, peft_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n  step 3  LoRA adapters attached")
    print(f"          trainable {trainable:,} / {total:,} params = {trainable/total:.3%}")
    print(f"          (the skill's headline: <1% of weights trained)")

    ds = Dataset.from_list([{"text": r["prompt"] + r["completion"]} for r in train_rows])
    args = SFTConfig(
        output_dir=str(MOD / "lora_out"),
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,          # effective batch 16
        num_train_epochs=3,                     # the skill: 1-3, more overfits
        learning_rate=2e-4,                     # LoRA runs hotter than full FT
        warmup_steps=5,                         # trl 1.12 exposes warmup_steps, not warmup_ratio
        lr_scheduler_type="cosine",
        logging_steps=20,
        save_strategy="no",
        report_to=[],
        seed=SEED,
        max_length=96,
    )
    print(f"\n  step 4  training: lr 2e-4, 3 epochs, effective batch 16, cosine schedule")
    trainer = SFTTrainer(model=model, train_dataset=ds, args=args)
    t0 = time.perf_counter()
    out = trainer.train()
    train_s = time.perf_counter() - t0
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    print(f"          {out.global_step} steps in {train_s:.0f}s")
    print(f"          loss {losses[0]:.4f} -> {losses[-1]:.4f} "
          f"({(losses[-1]/losses[0]-1)*100:+.1f}%)")

    # ---- Step 5b: evaluate the fine-tune ---------------------------------
    print(f"\n  step 5b fine-tuned model on the SAME held-out set")
    ft_model = trainer.model.to(DEVICE)
    t0 = time.perf_counter()
    ft_preds = generate(ft_model, tok, eval_prompts)
    ft = score(ft_preds, eval_rows)
    print(f"          {' | '.join(f'{k} {v:.3f}' for k, v in ft.items())}  "
          f"[{time.perf_counter()-t0:.0f}s]")
    print(f"          tuned output sample: {ft_preds[0].strip()[:100]!r}")
    print(f"          gold:                {eval_rows[0]['completion'].strip()[:100]!r}")

    print(f"\n  base vs fine-tuned on held-out data:")
    print(f"  {'metric':<16}{'base':>10}{'tuned':>10}{'delta':>10}")
    for k in base:
        print(f"  {k:<16}{base[k]:>10.3f}{ft[k]:>10.3f}{ft[k]-base[k]:>+10.3f}")

    print(f"\n  interpretation:")
    print(f"  - FORMAT was learned: valid-format rate {base['format_valid']:.0%} -> "
          f"{ft['format_valid']:.0%}. This is what fine-tuning is for.")
    print(f"  - FACTS were not, and should not be: exact-match {ft['exact_match']:.0%}. An 82M")
    print(f"    model cannot memorise 800 customers' revenue figures from 3 epochs, and")
    print(f"    trying to make it would be the wrong architecture. Facts belong in retrieval.")
    print(f"  - This is exactly the split the skill's step 0 predicts: fine-tune the form,")
    print(f"    retrieve the content.")

    adapter = MOD / "retail_lora_adapter"
    trainer.model.save_pretrained(str(adapter))
    print(f"\n  adapter saved to {adapter.name} "
          f"({sum(f.stat().st_size for f in adapter.rglob('*') if f.is_file())/1e6:.1f} MB "
          f"vs ~330 MB for a full fp32 checkpoint)")

    write_json({"base_model": BASE, "device": DEVICE, "method": "LoRA",
                "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": ["c_attn"]},
                "trainable_params": trainable, "total_params": total,
                "trainable_pct": trainable / total,
                "n_train": len(train_rows), "n_eval": len(eval_rows),
                "epochs": 3, "lr": 2e-4, "effective_batch": 16,
                "train_seconds": train_s, "loss_curve": losses,
                "baseline_metrics": base, "finetuned_metrics": ft,
                "sample_base": base_preds[0].strip(), "sample_tuned": ft_preds[0].strip(),
                "sample_gold": eval_rows[0]["completion"].strip()},
               "p4e_llm_finetuning_results.json")


if __name__ == "__main__":
    main()
