"""Run the entire CRISP-DM pipeline in order.

  python scripts/run_all.py            # every phase
  python scripts/run_all.py --fast     # skip the two model-downloading phases

Implements the `reproducible-ml` skill's hand-off: one command recreates every
number in every report from the raw data.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
PHASES = [
    ("Phase 0  fetch + verify raw data", "download_data.py", False),
    ("Phase 1  Business Understanding", "p1_business_understanding.py", False),
    ("Phase 2  (setup) build warehouse", "p0_build_warehouse.py", False),
    ("Phase 2  Data Understanding - Titanic", "p2a_titanic_eda.py", False),
    ("Phase 2  Data Understanding - Retail profile", "p2b_retail_profile.py", False),
    ("Phase 2  Data Understanding - warehouse skills", "p2c_warehouse_skills.py", False),
    ("Phase 3  Data Preparation - Titanic", "p3a_titanic_prep.py", False),
    ("Phase 4  Modeling - Titanic pipeline", "p4a_titanic_model.py", False),
    ("Phase 4  Modeling - PyTorch + debugging", "p4b_titanic_pytorch.py", False),
    ("Phase 4  Modeling - Retail analytics", "p4c_retail_analytics.py", False),
    ("Phase 4  Modeling - RAG pipeline", "p4d_rag_pipeline.py", True),
    ("Phase 4  Modeling - LoRA fine-tuning", "p4e_llm_finetuning.py", True),
    ("Phase 6  Deployment - model serving", "p6_model_serving.py", False),
    ("Phase 5  Evaluation", "p5_evaluation.py", False),
    ("Phase 6  Deployment - documentation", "p6b_deployment_docs.py", False),
]


def main() -> None:
    fast = "--fast" in sys.argv
    results, t_all = [], time.perf_counter()
    for label, script, needs_download in PHASES:
        if fast and needs_download:
            print(f"\n{'#'*78}\n# SKIP (--fast): {label}\n{'#'*78}")
            results.append((label, "skipped", 0.0))
            continue
        print(f"\n{'#'*78}\n# {label}  ->  {script}\n{'#'*78}")
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, str(HERE / script)])
        dt = time.perf_counter() - t0
        results.append((label, "ok" if r.returncode == 0 else f"FAILED ({r.returncode})", dt))
        if r.returncode != 0:
            print(f"\n!! {script} exited {r.returncode}; stopping.")
            break

    print(f"\n{'='*78}\nPIPELINE SUMMARY\n{'='*78}")
    for label, status, dt in results:
        print(f"  {status:<12} {dt:>7.1f}s  {label}")
    print(f"  {'':<12} {time.perf_counter()-t_all:>7.1f}s  TOTAL")


if __name__ == "__main__":
    main()
