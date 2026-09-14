"""Fetch the two raw datasets and verify them against the hashes logged in every run.

  python scripts/download_data.py

data/raw/ is git-ignored (222 MB with derived files), so this is step 0 of a
clean-clone reproduction. Both sources are public mirrors of the Kaggle files:
  * Titanic       -- kaggle.com/c/titanic            (datasciencedojo mirror)
  * Online Retail -- kaggle.com/datasets/carrie1/ecommerce-data  (= UCI 352)

Implements the `reproducible-ml` skill's Pillar 3: data is an immutable input
keyed by its SHA-256, and the pipeline refuses to run on a file that differs.
"""
from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import RAW, RETAIL, TITANIC, sha256

TITANIC_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
RETAIL_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"

# The SHA-256 of each file as used in every logged run (see outputs/tables/mlruns.jsonl).
EXPECTED = {
    "titanic.csv": "4a437fde05fe5264",
    "online_retail.csv": "c35975c40d0e1e10",
}


def fetch(url: str, timeout: int = 300) -> bytes:
    print(f"  GET {url}")
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)

    if not TITANIC.exists():
        TITANIC.write_bytes(fetch(TITANIC_URL))
    print(f"  titanic.csv        {TITANIC.stat().st_size:>12,} bytes")

    if not RETAIL.exists():
        xlsx = RAW / "online_retail.xlsx"
        if not xlsx.exists():
            z = zipfile.ZipFile(io.BytesIO(fetch(RETAIL_URL)))
            name = next(n for n in z.namelist() if n.lower().endswith(".xlsx"))
            xlsx.write_bytes(z.read(name))
        import pandas as pd

        print("  converting xlsx -> csv (541,909 rows; ~1 min)")
        pd.read_excel(xlsx, engine="openpyxl").to_csv(RETAIL, index=False)
    print(f"  online_retail.csv  {RETAIL.stat().st_size:>12,} bytes")

    ok = True
    for p in (TITANIC, RETAIL):
        h = sha256(p)[:16]
        match = h == EXPECTED[p.name]
        ok &= match
        print(f"  {p.name:<20} sha256 {h}  {'OK' if match else 'MISMATCH (expected ' + EXPECTED[p.name] + ')'}")
    if not ok:
        raise SystemExit("  !! a raw file differs from the one every logged run used; results will not reproduce")
    print("  raw data verified against the logged hashes")


if __name__ == "__main__":
    main()
