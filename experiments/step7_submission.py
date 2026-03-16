"""
Step 7 — Create Submission
Reads test predictions and produces submission.csv
"""
import polars as pl
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\ujjaw\Downloads\AML_Mule_Project")
DATA = ROOT / "data" / "archive"
MODELS = ROOT / "models"

test_accounts = pl.read_parquet(DATA / "test_accounts.parquet")
preds = np.load(MODELS / "test_preds.npy")

print(f"test_accounts: {test_accounts.shape[0]}")
print(f"predictions:   {preds.shape[0]}")
assert test_accounts.shape[0] == preds.shape[0], "Mismatch in row counts!"

submission = pd.DataFrame({
    "account_id": test_accounts["account_id"].to_list(),
    "is_mule": preds,
})

out_path = ROOT / "submission.csv"
submission.to_csv(out_path, index=False)

print(f"\n✓ submission.csv saved ({submission.shape})")
print(f"  is_mule stats: mean={preds.mean():.4f}, min={preds.min():.4f}, max={preds.max():.4f}")
print(submission.head(10))
