#!/usr/bin/env python

# Project: Labels vs. Scale: Breaking Even on Banking77 with Encoders and Decoders

"""Download Banking77 at a pinned Hub revision and save local copies.

Writes to <project root>/data/raw/: train.parquet, test.parquet, label_names.json
With --preview, also writes readable CSVs to <project root>/data/preview/ 
Re-running does nothing unless --force is passed.

Usage:
    python download_data.py            # from scripts/
    python scripts/download_data.py    # from the project root
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset


REVISION = "0b62e79500b84f0fff5a36a4404028f48fbc8621"  # pinned commit; cite in write-up
BASE = f"https://huggingface.co/datasets/PolyAI/banking77/resolve/{REVISION}/data"
DATA_FILES = {
    "train": f"{BASE}/train-00000-of-00001.parquet",
    "test": f"{BASE}/test-00000-of-00001.parquet",
}
 
# scripts/ -> project root -> data/raw  
RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
PREVIEW = RAW.parent / "preview"
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download and overwrite")
    ap.add_argument("--preview", action="store_true", help="also write readable CSVs")
    args = ap.parse_args()
 
    train_p, test_p, names_p = RAW / "train.parquet", RAW / "test.parquet", RAW / "label_names.json"
 
    if all(p.exists() for p in (train_p, test_p, names_p)) and not args.force:
        print("Raw files already present (use --force to re-download).")
    else:
        ds = load_dataset("parquet", data_files=DATA_FILES)
        RAW.mkdir(parents=True, exist_ok=True)
        ds["train"].to_parquet(str(train_p))
        ds["test"].to_parquet(str(test_p))
        names = ds["train"].features["label"].names
        names_p.write_text(json.dumps(names, indent=2))
        print(f"Saved train={len(ds['train'])}, test={len(ds['test'])}, labels={len(names)} -> {RAW}")
 
    if args.preview:
        names = json.loads(names_p.read_text())
        PREVIEW.mkdir(parents=True, exist_ok=True)
        for split in ("train", "test"):
            df = pd.read_parquet(RAW / f"{split}.parquet")
            df["label_name"] = df["label"].map(lambda i: names[i])
            df.to_csv(PREVIEW / f"{split}_preview.csv", index=False)
        print(f"Wrote preview CSVs -> {PREVIEW}")
 
 
if __name__ == "__main__":
    main()
 