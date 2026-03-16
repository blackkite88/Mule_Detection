"""
Step 9v10 — Final Robust Submission (Ghost Mule Avoidance + F1 Prior + Lifecycle IoU)

This is the definitive competition submission script designed to defeat the Noisy Label trap.
It combines:
1. Robust Classification: Reverts to V10 predictions (Ghost Mule down-weighted).
2. F1 Optimization: Applied 2.79% prior-based threshold.
3. IoU Optimization: Applied Full-Lifecycle temporal bounding.

Inputs:  models/test_preds_v10.npy
         models/test_ids_order.parquet (or test_ids_order_v4.parquet)
         data/archive/transactions/batch-{1-4}/*.parquet
Outputs: submission.csv
"""
import polars as pl
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\ujjaw\Downloads\AML_Mule_Project")
DATA = ROOT / "data" / "archive"
MODELS = ROOT / "models"
MULE_PRIOR = 0.027915  # Exactly the training is_mule mean

# 1. Load Robust V10 Predictions
# ─────────────────────────────────────────────────────────────────────────────
test_ids = pl.read_parquet(MODELS / "test_ids_order.parquet")
preds = np.load(MODELS / "test_preds_v10.npy")

print(f"Test accounts: {len(test_ids)}")
print(f"Preds loaded:  {len(preds)}")

# 2. Dynamic Thresholding (F1 Preservation)
# ─────────────────────────────────────────────────────────────────────────────
n_mules_to_predict = int(len(preds) * MULE_PRIOR)
threshold = np.percentile(preds, 100 * (1 - MULE_PRIOR))

print(f"\nApplying Robust 2.79% Prior...")
print(f"Target mule count: {n_mules_to_predict}")
print(f"Threshold for V10 weights: {threshold:.6f}")

test_df = test_ids.with_columns(pl.Series("pred", preds))
predicted_mules = test_df.filter(pl.col("pred") >= threshold)
mule_ids = predicted_mules["account_id"].to_list()

# 3. Lifecycle Bounding (IoU Preservation)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nFinalizing lifecycle windows for {len(mule_ids)} predicted mules...")

mule_series = pl.Series("account_id", mule_ids)
boundaries = []

for batch in ["batch-1", "batch-2", "batch-3", "batch-4"]:
    print(f"  Scanning {batch}...")
    batch_b = (
        pl.scan_parquet(DATA / "transactions" / batch / "*.parquet")
        .filter(pl.col("account_id").is_in(mule_series))
        .with_columns(pl.col("transaction_timestamp").str.to_datetime().alias("ts"))
        .group_by("account_id")
        .agg([
            pl.col("ts").min().alias("min_ts"),
            pl.col("ts").max().alias("max_ts"),
        ])
        .collect()
    )
    boundaries.append(batch_b)

# Aggregate across batches
agg_boundaries = (
    pl.concat(boundaries)
    .group_by("account_id")
    .agg([
        pl.col("min_ts").min().alias("suspicious_start"),
        pl.col("max_ts").max().alias("suspicious_end")
    ])
)

# 4. Final Formatting
# ─────────────────────────────────────────────────────────────────────────────
submission = (
    test_df
    .join(agg_boundaries, on="account_id", how="left")
    .with_columns([
        pl.col("pred").alias("is_mule"),
        pl.col("suspicious_start").dt.strftime("%Y-%m-%d %H:%M:%S").fill_null(""),
        pl.col("suspicious_end").dt.strftime("%Y-%m-%d %H:%M:%S").fill_null(""),
    ])
    .select(["account_id", "is_mule", "suspicious_start", "suspicious_end"])
    .to_pandas()
)

submission.to_csv(ROOT / "submission.csv", index=False)

print(f"\n✓ submission.csv (Robust V10) saved ({submission.shape})")
print(f"  Confirming mules flagged: {(submission['is_mule'] >= threshold).sum()}")
print(f"  Confirming windows added: {(submission['suspicious_start'] != '').sum()}")
