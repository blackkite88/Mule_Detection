"""
Step 9v6 — Optimal Submission (V5 Models + F1 Prior + Lifecycle IoU)

This script implements:
1. Improved F1: Thresholding based on the true 2.79% mule prior.
2. Improved IoU: Full-lifecycle temporal windows (First to Last txn).
3. Best Model: Uses the V5 SHAP-based blended predictions.

Inputs:  models/test_preds_v4.npy (V5 training output)
         models/test_ids_order.parquet
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

# 1. Load V5 Predictions
# ─────────────────────────────────────────────────────────────────────────────
test_ids = pl.read_parquet(MODELS / "test_ids_order.parquet")
# Note: step6v5 saved the predictions as 'test_preds_v4.npy' 
preds = np.load(MODELS / "test_preds_v4.npy")

print(f"Test accounts: {len(test_ids)}")
print(f"Preds loaded:  {len(preds)}")

# 2. Dynamic Thresholding (F1 Optimization)
# ─────────────────────────────────────────────────────────────────────────────
# We pick the threshold that results in Exactly 2.79% predicted mules.
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

# We calculate first and last txn timestamp for each predicted mule
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
# Match back to the full list of test accounts
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
print("\nTop predictions:")
print(submission.sort_values("is_mule", ascending=False).head(10))
