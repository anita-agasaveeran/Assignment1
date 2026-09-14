"""CRISP-DM Phase 2 - Data Understanding (Track B: Online Retail).

Skills demonstrated by *executing the scripts each skill ships*:
  * programmatic-eda    -- data_overview, null_profiler, outlier_detector,
                           distribution_summary, correlation_explorer
  * data-quality-audit  -- null_counter, duplicate_finder, referential_integrity,
                           value_range_validator, freshness_check
  * data-catalog-entry  -- catalog_extractor against the SQLite warehouse
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import PROC, REP, RETAIL, ROOT, WAREHOUSE, banner

SKILLS = Path.home() / ".claude" / "skills"
PY = sys.executable
LOG: list[str] = []


def run(skill: str, script: str, *args: str, label: str = "") -> str:
    """Invoke a skill's own bundled script and capture its output verbatim."""
    path = SKILLS / skill / "scripts" / script
    cmd = [PY, str(path), *args]
    print(f"\n--- {skill}/scripts/{script} {label}")
    print(f"    $ python {path.name} {' '.join(args)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    print("\n".join("    " + ln for ln in out.strip().splitlines()[:40]))
    if len(out.strip().splitlines()) > 40:
        print(f"    ... ({len(out.strip().splitlines()) - 40} more lines, full text in the log)")
    LOG.append(f"\n{'='*78}\n$ python {skill}/scripts/{script} {' '.join(args)}\n{'='*78}\n{out}")
    if r.returncode != 0:
        print(f"    [exit {r.returncode}]")
    return out


def main() -> None:
    # A 40k-row stratified slice keeps the bundled scripts fast; full-file checks
    # that matter (nulls, duplicates, ranges) run on all 541,909 rows.
    import pandas as pd

    sample = PROC / "retail_sample.csv"
    if not sample.exists():
        pd.read_csv(RETAIL).sample(40_000, random_state=42).to_csv(sample, index=False)

    # ---------------- programmatic-eda: its 7-step process ------------------
    banner("programmatic-eda", "Phase 2 - Data Understanding",
           "running the skill's own scripts against Online Retail (541,909 rows)")
    run("programmatic-eda", "data_overview.py", "--input", str(RETAIL), "--sample", "3",
        label="[step 1 load & overview]")
    run("programmatic-eda", "null_profiler.py", "--input", str(RETAIL),
        "--warn-pct", "5", "--fail-pct", "30",
        "--output", str(ROOT / "outputs/tables/retail_null_profile.csv"),
        label="[step 2 null profile]")
    run("programmatic-eda", "outlier_detector.py", "--input", str(sample), "--method", "both",
        "--output", str(ROOT / "outputs/tables/retail_outliers.csv"),
        label="[step 3 outliers]")
    run("programmatic-eda", "distribution_summary.py", "--input", str(sample), "--bins", "8",
        "--output", str(ROOT / "outputs/tables/retail_distributions.csv"),
        label="[step 4 distributions]")
    run("programmatic-eda", "correlation_explorer.py", "--input", str(sample), "--threshold", "0.8",
        "--output", str(ROOT / "outputs/tables/retail_correlations.csv"),
        label="[step 5 correlations]")

    # ---------------- data-quality-audit: its 7-step process ----------------
    banner("data-quality-audit", "Phase 2 - Data Understanding",
           "quality scorecard against business rules")
    thresholds = json.dumps({"CustomerID": 0.0, "Description": 0.0, "InvoiceNo": 0.0})
    run("data-quality-audit", "null_counter.py", "--input", str(RETAIL),
        "--thresholds", thresholds, label="[step 1 completeness]")
    run("data-quality-audit", "duplicate_finder.py", "--input", str(RETAIL),
        "--key", "InvoiceNo,StockCode,Quantity,InvoiceDate", label="[step 2 duplicates]")

    # Referential integrity needs parent/child extracts from the warehouse
    import sqlite3

    con = sqlite3.connect(WAREHOUSE)
    pd.read_sql("SELECT InvoiceNo, StockCode, Quantity FROM invoice_lines LIMIT 200000", con) \
      .to_csv(PROC / "ri_child_lines.csv", index=False)
    pd.read_sql("SELECT StockCode FROM products", con).to_csv(PROC / "ri_parent_products.csv", index=False)
    con.close()
    run("data-quality-audit", "referential_integrity.py",
        "--child", str(PROC / "ri_child_lines.csv"), "--parent", str(PROC / "ri_parent_products.csv"),
        "--child-key", "StockCode", "--parent-key", "StockCode", label="[step 3 referential integrity]")

    rules = json.dumps({
        "Quantity": {"min": 1, "max": 10000},
        "UnitPrice": {"min": 0.01, "max": 5000},
    })
    run("data-quality-audit", "value_range_validator.py", "--input", str(RETAIL), "--rules", rules,
        label="[step 4 business-rule ranges]")
    run("data-quality-audit", "freshness_check.py", "--input", str(RETAIL),
        "--timestamp-col", "InvoiceDate", "--max-lag-hours", "48", label="[step 5 freshness]")

    # ---------------- data-catalog-entry ------------------------------------
    banner("data-catalog-entry", "Phase 2 - Data Understanding",
           "technical metadata extraction from the warehouse")
    for table in ["invoice_lines", "customers"]:
        run("data-catalog-entry", "catalog_extractor.py",
            "--conn", f"sqlite:///{WAREHOUSE}", "--table", table,
            "--output", str(ROOT / f"outputs/reports/catalog_{table}.md"),
            label=f"[{table}]")

    (REP / "p2b_skill_script_console_log.txt").write_text(
        "Console output from every bundled skill script executed in CRISP-DM Phase 2.\n"
        + "".join(LOG)
    )
    print(f"\n  -> full console log: outputs/reports/p2b_skill_script_console_log.txt")


if __name__ == "__main__":
    main()
