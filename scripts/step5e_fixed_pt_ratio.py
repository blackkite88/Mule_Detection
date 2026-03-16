"""
Step 2i — Fixed SHAP-Recommended Features
==========================================
Identical to 2h but fixes the pass_through_ratio scale.
Standard: matched_vol / max(in_flow, out_flow)
"""
import polars as pl
import numpy as np
from pathlib import Path

ROOT = Path(r"C:\Users\ujjaw\Downloads\AML_Mule_Project")
DATA = ROOT / "data" / "archive"
FEAT = ROOT / "features"

base_shap = pl.read_parquet(FEAT / "shap_recommended_features.parquet")

# We need the intermediate daily inflow/outflow to fix it properly
# Or we can just use the existing one and multiply by 2? 
# No, because that assumes inflow == outflow.
# Let's re-save a fixed version of just that feature.

print("Recalculating pass_through_ratio accurately...")

# Re-scan the inout data (if it exists) - Wait, I deleted the temp files in 2h.
# I have to re-read transactions. I'll just do it for the pass-through feature.

target_ids = pl.concat([
    pl.read_parquet(DATA / "train_labels.parquet").select("account_id"),
    pl.read_parquet(DATA / "test_accounts.parquet").select("account_id"),
]).unique()["account_id"]

parts = []
for batch in ["batch-1", "batch-2", "batch-3", "batch-4"]:
    print(f"  Scanning {batch}...")
    inout = (
        pl.scan_parquet(DATA / "transactions" / batch / "*.parquet")
        .filter(pl.col("account_id").is_in(target_ids))
        .with_columns([
            pl.col("transaction_timestamp").str.to_datetime().alias("ts"),
            pl.col("amount").abs().alias("abs_amount"),
        ])
        .with_columns(pl.col("ts").dt.date().alias("day"))
        .group_by(["account_id", "day"])
        .agg([
            pl.col("abs_amount").filter(pl.col("txn_type") == "C").sum().alias("daily_inflow"),
            pl.col("abs_amount").filter(pl.col("txn_type") == "D").sum().alias("daily_outflow"),
        ])
        .collect()
    )
    parts.append(inout)

fixed_pt = (
    pl.concat(parts)
    .group_by(["account_id", "day"])
    .agg([
        pl.col("daily_inflow").sum(),
        pl.col("daily_outflow").sum(),
    ])
    .with_columns(
        pl.min_horizontal("daily_inflow", "daily_outflow").alias("matched"),
        pl.max_horizontal("daily_inflow", "daily_outflow").alias("denom")
    )
    .group_by("account_id")
    .agg([
        pl.col("matched").sum().alias("total_matched"),
        pl.col("denom").sum().alias("total_denom")
    ])
    .with_columns(
        (pl.col("total_matched") / (pl.col("total_denom") + 1e-9)).alias("pass_through_ratio_fixed")
    )
    .select(["account_id", "pass_through_ratio_fixed"])
)

# Update base_shap
final = base_shap.join(fixed_pt, on="account_id", how="left").with_columns(
    pl.col("pass_through_ratio_fixed").fill_null(0.0).alias("pass_through_ratio")
).drop("pass_through_ratio_fixed")

final.write_parquet(FEAT / "shap_recommended_features_v2.parquet")
print(f"✓ Fixed pass_through_ratio. Max: {final['pass_through_ratio'].max():.4f}")
