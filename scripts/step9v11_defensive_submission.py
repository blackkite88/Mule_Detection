"""
Step 9v11 — Defensive Submission (Trap Immune V11)

This script generates the final trap-immune submission.
1. Robust Signal: V11 predictions (No Target Encoding + Ghost Denoising).
2. F1 Prior: 2.79% anchor.
3. Temporal: Lifecycle Bounding.
"""
import polars as pl
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\ujjaw\Downloads\AML_Mule_Project")
DATA = ROOT / "data" / "archive"
MODELS = ROOT / "models"
MULE_PRIOR = 0.027915

# 1. Load V11 Defensive Predictions
# ─────────────────────────────────────────────────────────────────────────────
test_ids = pl.read_parquet(MODELS / "test_ids_order.parquet")
preds = np.load(MODELS / "test_preds_v11.npy")

# 2. Thresholding
# ─────────────────────────────────────────────────────────────────────────────
n_mules = int(len(preds) * MULE_PRIOR)
threshold = np.percentile(preds, 100 * (1 - MULE_PRIOR))

print(f"Applying V11 Defensive Prior...")
print(f"Threshold: {threshold:.6f}")

test_df = test_ids.with_columns(pl.Series("pred", preds))
mule_ids = test_df.filter(pl.col("pred") >= threshold)["account_id"].to_list()

# 3. Lifecycle Bounding
# ─────────────────────────────────────────────────────────────────────────────
print(f"Bounding {len(mule_ids)} mules...")
mule_series = pl.Series("account_id", mule_ids)
parts = []
for batch in ["batch-1", "batch-2", "batch-3", "batch-4"]:
    print(f"  {batch}...")
    b = (
        pl.scan_parquet(DATA / "transactions" / batch / "*.parquet")
        .filter(pl.col("account_id").is_in(mule_series))
        .with_columns(pl.col("transaction_timestamp").str.to_datetime().alias("ts"))
        .group_by("account_id")
        .agg([
            pl.col("ts").min().alias("start"),
            pl.col("ts").max().alias("end"),
        ])
        .collect()
    )
    parts.append(b)

agg = (
    pl.concat(parts)
    .group_by("account_id")
    .agg([
        pl.col("start").min().alias("suspicious_start"),
        pl.col("end").max().alias("suspicious_end")
    ])
)

# 4. Final Submission
# ─────────────────────────────────────────────────────────────────────────────
submission = (
    test_df
    .join(agg, on="account_id", how="left")
    .with_columns([
        pl.col("pred").alias("is_mule"),
        pl.col("suspicious_start").dt.strftime("%Y-%m-%d %H:%M:%S").fill_null(""),
        pl.col("suspicious_end").dt.strftime("%Y-%m-%d %H:%M:%S").fill_null(""),
    ])
    .select(["account_id", "is_mule", "suspicious_start", "suspicious_end"])
    .to_pandas()
)

submission.to_csv(ROOT / "submission.csv", index=False)
print(f"\n✓ submission.csv (Defensive V11) saved.")
