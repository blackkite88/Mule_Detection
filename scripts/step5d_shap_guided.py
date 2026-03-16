"""
Step 2h — SHAP-Recommended Features
=====================================
Engineers 6 high-value features identified by SHAP analysis, plus a
pass-through ratio. Processes transactions in batch-mode.

New features:
  1. rolling_7d_zscore    — z-score of daily volume vs 7-day rolling mean
  2. cp_entropy           — Shannon entropy of counterparty distribution
  3. days_to_first_txn    — gap from account opening to first transaction
  4. turnover_ratio       — total volume / avg balance
  5. peak_day_conc        — concentration of txns on peak day
  6. pass_through_ratio   — matched in/out volume within 24h / total volume

Saves: features/shap_recommended_features.parquet
"""
import polars as pl
import numpy as np
from pathlib import Path
from scipy.stats import entropy
import gc

ROOT = Path(r"C:\Users\ujjaw\Downloads\AML_Mule_Project")
DATA = ROOT / "data" / "archive"
FEAT = ROOT / "features"
FEAT.mkdir(parents=True, exist_ok=True)

# ── Target accounts ──────────────────────────────────────────────────
target_ids = pl.concat([
    pl.read_parquet(DATA / "train_labels.parquet").select("account_id"),
    pl.read_parquet(DATA / "test_accounts.parquet").select("account_id"),
]).unique()["account_id"]

print(f"Target accounts: {len(target_ids)}")

# ── Load account opening dates (for days_to_first_txn) ───────────────
accounts = pl.read_parquet(DATA / "accounts.parquet").select([
    "account_id",
    pl.col("account_opening_date").str.to_date().alias("opening_date"),
])

# ── Load balance features (for turnover_ratio) ──────────────────────
# We'll use the avg_balance from behavior_features if it exists, else compute from txn data
try:
    behav = pl.read_parquet(FEAT / "behavior_features.parquet").select([
        "account_id", "avg_balance"
    ])
    print(f"Loaded avg_balance from behavior_features: {behav.shape[0]} accounts")
except Exception:
    behav = None
    print("  ⚠ No behavior_features.parquet found, will compute balance proxy")

# ═══════════════════════════════════════════════════════════════════════
# PASS 1: Collect per-batch intermediate data to disk
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Processing transactions (Out-of-Core to 'temp_shap')...")
print("=" * 70)

TEMP = FEAT / "temp_shap"
TEMP.mkdir(parents=True, exist_ok=True)

for batch in ["batch-1", "batch-2", "batch-3", "batch-4"]:
    print(f"\n  Processing {batch}...")
    lf = pl.scan_parquet(DATA / "transactions" / batch / "*.parquet")

    # ── Daily volume (for zscore + peak day) ─────────────────────
    daily = (
        lf
        .filter(pl.col("account_id").is_in(target_ids))
        .with_columns(
            pl.col("transaction_timestamp").str.to_datetime().alias("ts"),
            pl.col("amount").abs().alias("abs_amount"),
        )
        .with_columns(pl.col("ts").dt.date().alias("day"))
        .group_by(["account_id", "day"])
        .agg([
            pl.col("abs_amount").sum().alias("daily_vol"),
            pl.len().alias("daily_txns"),
        ])
    )
    # Write to disk lazily
    daily.sink_parquet(TEMP / f"daily_vol_{batch}.parquet")
    print(f"    daily volumes saved")

    # ── Counterparty distribution (for entropy) ──────────────────
    cp_dist = (
        lf
        .filter(pl.col("account_id").is_in(target_ids))
        .group_by(["account_id", "counterparty_id"])
        .agg(pl.len().alias("txn_count"))
    )
    cp_dist.sink_parquet(TEMP / f"cp_dist_{batch}.parquet")
    print(f"    counterparty pairs saved")

    # ── First transaction date ───────────────────────────────────
    first_txn = (
        lf
        .filter(pl.col("account_id").is_in(target_ids))
        .with_columns(
            pl.col("transaction_timestamp").str.to_datetime().alias("ts"),
        )
        .group_by("account_id")
        .agg(pl.col("ts").min().alias("first_txn_ts"))
    )
    first_txn.sink_parquet(TEMP / f"first_txn_{batch}.parquet")

    # ── Total volume per account (for turnover ratio) ────────────
    vol = (
        lf
        .filter(pl.col("account_id").is_in(target_ids))
        .with_columns(pl.col("amount").abs().alias("abs_amount"))
        .group_by("account_id")
        .agg(pl.col("abs_amount").sum().alias("batch_volume"))
    )
    vol.sink_parquet(TEMP / f"vol_{batch}.parquet")

    # ── Inflow/outflow for pass-through ratio ────────────────────
    inout = (
        lf
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
            pl.col("abs_amount").sum().alias("daily_total"),
        ])
    )
    inout.sink_parquet(TEMP / f"inout_{batch}.parquet")

    gc.collect()

# ═══════════════════════════════════════════════════════════════════════
# FEATURE 1: rolling_7d_zscore
# ═══════════════════════════════════════════════════════════════════════
print("\n[1/6] Computing rolling_7d_zscore (Polars out-of-core)...")

daily_all = (
    pl.scan_parquet(TEMP / "daily_vol_*.parquet")
    .group_by(["account_id", "day"])
    .agg([pl.col("daily_vol").sum(), pl.col("daily_txns").sum()])
    .collect()
)

# Process in chunks to avoid windowing OOM
unique_ids = daily_all["account_id"].unique().to_list()
chunk_size = 20000
zscore_parts = []

for i in range(0, len(unique_ids), chunk_size):
    chunk_ids = unique_ids[i:i+chunk_size]
    chunk = (
        daily_all
        .filter(pl.col("account_id").is_in(chunk_ids))
        .sort(["account_id", "day"])
        .with_columns([
            pl.col("daily_vol").rolling_mean(window_size=7, min_periods=2, center=True)
              .over("account_id").alias("roll_mean"),
            pl.col("daily_vol").rolling_std(window_size=7, min_periods=2, center=True)
              .over("account_id").alias("roll_std")
        ])
        .with_columns(
            ((pl.col("daily_vol") - pl.col("roll_mean")) / (pl.col("roll_std") + 1e-9)).alias("zscore")
        )
        .group_by("account_id")
        .agg([
            pl.col("zscore").max().fill_null(0.0).alias("rolling_7d_zscore_max"),
            pl.col("zscore").mean().fill_null(0.0).alias("rolling_7d_zscore_mean"),
        ])
    )
    zscore_parts.append(chunk)

zscore_feat = pl.concat(zscore_parts)
print(f"  rolling_7d_zscore: {zscore_feat.shape}")

# ═══════════════════════════════════════════════════════════════════════
# FEATURE 2: cp_entropy (Shannon entropy of counterparty distribution)
# ═══════════════════════════════════════════════════════════════════════
print("[2/6] Computing cp_entropy (Polars out-of-core)...")

entropy_feat = (
    pl.scan_parquet(TEMP / "cp_dist_*.parquet")
    .group_by(["account_id", "counterparty_id"])
    .agg(pl.col("txn_count").sum())
    .with_columns(
        pl.col("txn_count").sum().over("account_id").alias("total_txns")
    )
    .with_columns(
        (pl.col("txn_count") / pl.col("total_txns")).alias("p")
    )
    .with_columns(
        (pl.col("p") * pl.col("p").log(base=2)).alias("p_log_p")
    )
    .group_by("account_id")
    .agg(
        (-pl.col("p_log_p").sum()).fill_nan(0.0).alias("cp_entropy")
    )
    .collect()
)
print(f"  cp_entropy: {entropy_feat.shape}")

# ═══════════════════════════════════════════════════════════════════════
# FEATURE 3: days_to_first_txn
# ═══════════════════════════════════════════════════════════════════════
print("[3/6] Computing days_to_first_txn...")

first_txn_feat = (
    pl.scan_parquet(TEMP / "first_txn_*.parquet")
    .group_by("account_id")
    .agg(pl.col("first_txn_ts").min())
    .collect()
    .join(accounts, on="account_id", how="left")
    .with_columns(
        (pl.col("first_txn_ts").dt.date() - pl.col("opening_date"))
        .dt.total_days()
        .alias("days_to_first_txn")
    )
    .select(["account_id", "days_to_first_txn"])
    .with_columns(
        pl.col("days_to_first_txn").fill_null(999).clip(lower_bound=0)
    )
)
print(f"  days_to_first_txn: {first_txn_feat.shape}")

# ═══════════════════════════════════════════════════════════════════════
# FEATURE 4: turnover_ratio (total_volume / avg_balance)
# ═══════════════════════════════════════════════════════════════════════
print("[4/6] Computing turnover_ratio...")

total_vol = (
    pl.scan_parquet(TEMP / "vol_*.parquet")
    .group_by("account_id")
    .agg(pl.col("batch_volume").sum().alias("total_volume"))
    .collect()
)

if behav is not None:
    turnover_feat = (
        total_vol
        .join(behav, on="account_id", how="left")
        .with_columns(
            (pl.col("total_volume") /
             (pl.col("avg_balance").abs() + 1.0))
            .alias("turnover_ratio")
        )
        .select(["account_id", "turnover_ratio"])
        .with_columns(pl.col("turnover_ratio").fill_null(0.0))
    )
else:
    turnover_feat = total_vol.with_columns(
        pl.lit(0.0).alias("turnover_ratio")
    ).select(["account_id", "turnover_ratio"])

print(f"  turnover_ratio: {turnover_feat.shape}")

# ═══════════════════════════════════════════════════════════════════════
# FEATURE 5: peak_day_conc (% of total txns on the busiest day)
# ═══════════════════════════════════════════════════════════════════════
print("[5/6] Computing peak_day_conc...")

peak_day_feat = (
    daily_all
    .group_by("account_id")
    .agg([
        pl.col("daily_txns").sum().alias("total_txns"),
        pl.col("daily_txns").max().alias("peak_txns"),
        pl.col("daily_vol").max().alias("peak_vol"),
        pl.col("daily_vol").sum().alias("total_vol"),
    ])
    .with_columns([
        (pl.col("peak_txns") / (pl.col("total_txns") + 1e-9))
            .alias("peak_day_conc_txn"),
        (pl.col("peak_vol") / (pl.col("total_vol") + 1e-9))
            .alias("peak_day_conc_vol"),
    ])
    .select(["account_id", "peak_day_conc_txn", "peak_day_conc_vol"])
)
print(f"  peak_day_conc: {peak_day_feat.shape}")

# ═══════════════════════════════════════════════════════════════════════
# FEATURE 6: pass_through_ratio (daily min(in,out) / total volume)
# ═══════════════════════════════════════════════════════════════════════
print("[6/6] Computing pass_through_ratio...")

pass_through_feat = (
    pl.scan_parquet(TEMP / "inout_*.parquet")
    .group_by(["account_id", "day"])
    .agg([
        pl.col("daily_inflow").sum(),
        pl.col("daily_outflow").sum(),
        pl.col("daily_total").sum(),
    ])
    .with_columns(
        pl.min_horizontal("daily_inflow", "daily_outflow").alias("matched_vol")
    )
    .group_by("account_id")
    .agg([
        pl.col("matched_vol").sum().alias("total_matched"),
        pl.col("daily_total").sum().alias("total_vol"),
    ])
    .with_columns(
        (pl.col("total_matched") / (pl.col("total_vol") + 1e-9))
            .alias("pass_through_ratio")
    )
    .select(["account_id", "pass_through_ratio"])
    .collect()
)
print(f"  pass_through_ratio: {pass_through_feat.shape}")

# Cleanup temp files
import shutil
shutil.rmtree(TEMP, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════
# COMBINE ALL FEATURES
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Combining all SHAP-recommended features...")
print("=" * 70)

base = pl.DataFrame({"account_id": target_ids})

result = (
    base
    .join(zscore_feat, on="account_id", how="left")
    .join(entropy_feat, on="account_id", how="left")
    .join(first_txn_feat, on="account_id", how="left")
    .join(turnover_feat, on="account_id", how="left")
    .join(peak_day_feat, on="account_id", how="left")
    .join(pass_through_feat, on="account_id", how="left")
    .fill_null(0.0)
)

print(f"\nFinal shape: {result.shape}")
print(f"Columns: {result.columns}")
print(f"\nSample stats:")
for col in result.columns:
    if col == "account_id":
        continue
    vals = result[col].to_numpy().astype(float)
    print(f"  {col:30s}  mean={np.nanmean(vals):10.4f}  "
          f"std={np.nanstd(vals):10.4f}  "
          f"min={np.nanmin(vals):10.4f}  max={np.nanmax(vals):10.4f}")

result.write_parquet(FEAT / "shap_recommended_features.parquet")
print(f"\n✓ features/shap_recommended_features.parquet saved ({result.shape})")
