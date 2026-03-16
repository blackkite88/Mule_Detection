"""
Step 9v7 — Final Triple Submission (Ensemble + F1 Prior + Lifecycle IoU)

This script implements:
1. Triple Ensemble: Uses rank-blended predictions from LGB, XGB, and CatBoost.
2. Improved F1: Thresholding based on the true 2.79% mule prior.
3. Improved IoU: Full-lifecycle temporal windows (First to Last txn).

Inputs:  models/test_preds_v7.npy (LGB+XGB+CAT Rank Blend)
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

# 1. Load V7 Ensemble Predictions
# ─────────────────────────────────────────────────────────────────────────────
test_ids = pl.read_parquet(MODELS / "test_ids_order.parquet")
preds = np.load(MODELS / "test_preds_v7.npy")

print(f"Test accounts: {len(test_ids)}")
print(f"Preds loaded:  {len(preds)}")

# 2. Dynamic Thresholding (F1 Optimization)
# ─────────────────────────────────────────────────────────────────────────────
# We find the threshold for the 2.79% prior
n_mules_to_predict = int(len(preds) * MULE_PRIOR)
threshold = np.percentile(preds, 100 * (1 - MULE_PRIOR))

print(f"\nTarget mule rate: {MULE_PRIOR*100:.4f}%")
print(f"Target mule count: {n_mules_to_predict}")
print(f"Optimized Threshold: {threshold:.6f}")

test_df = test_ids.with_columns(pl.Series("pred", preds))
predicted_mules = test_df.filter(pl.col("pred") >= threshold)
mule_ids = predicted_mules["account_id"].to_list()

# 3. Lifecycle Bounding (IoU Optimization)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nScanning lifecycle boundaries for {len(mule_ids)} predicted mules...")

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

print(f"\n✓ submission.csv saved ({submission.shape})")
print(f"  Count with windows: {(submission['suspicious_start'] != '').sum()}")
print(f"  Mule Rate in Sub:   {submission['is_mule'].apply(lambda x: 1 if x >= threshold else 0).mean():.4f}")

# Sanity check top preds
print("\nTop predictions (Ensemble Ranks):")
print(submission.sort_values("is_mule", ascending=False).head(10))
