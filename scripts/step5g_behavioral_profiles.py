"""
Step 3b — Mule Behavior Pattern Features
Detects specific AML patterns:
  1. Dormant activation (inactivity → sudden bursts)
  2. Rapid pass-through (credit quickly followed by debit)
  3. Structuring (amounts near thresholds)
  4. Monthly velocity / time-window features
  5. Counterparty concentration
"""
import polars as pl
import gc
from pathlib import Path

ROOT = Path(r"C:\Users\ujjaw\Downloads\AML_Mule_Project")
DATA = ROOT / "data" / "archive"
FEAT = ROOT / "features"
FEAT.mkdir(parents=True, exist_ok=True)

target_ids = pl.concat([
    pl.read_parquet(DATA / "train_labels.parquet").select("account_id"),
    pl.read_parquet(DATA / "test_accounts.parquet").select("account_id"),
]).unique()["account_id"]

# ═══════════════════════════════════════════════════════════════════════
# PASS 1 — Monthly velocity features + dormancy detection
# ═══════════════════════════════════════════════════════════════════════
monthly_partials = []

for batch in ["batch-1", "batch-2", "batch-3", "batch-4"]:
    print(f"[Monthly] Processing {batch} ...")

    df = (
        pl.scan_parquet(DATA / "transactions" / batch / "*.parquet")
        .filter(pl.col("account_id").is_in(target_ids))
        .select(["account_id", "transaction_timestamp", "amount"])
        .with_columns([
            pl.col("transaction_timestamp").str.to_datetime().alias("ts"),
            pl.col("amount").abs().alias("abs_amount"),
        ])
        .with_columns([
            pl.col("ts").dt.strftime("%Y-%m").alias("year_month"),
        ])
        .group_by(["account_id", "year_month"])
        .agg([
            pl.len().alias("monthly_txns"),
            pl.col("abs_amount").sum().alias("monthly_volume"),
            pl.col("abs_amount").max().alias("monthly_max"),
        ])
        .collect()
    )
    monthly_partials.append(df)
    print(f"  {batch}: {df.shape[0]} account-months")

monthly_all = (
    pl.concat(monthly_partials)
    .group_by(["account_id", "year_month"])
    .agg([
        pl.col("monthly_txns").sum(),
        pl.col("monthly_volume").sum(),
        pl.col("monthly_max").max(),
    ])
)
del monthly_partials
gc.collect()

# Compute per-account monthly stats
monthly_features = (
    monthly_all
    .group_by("account_id")
    .agg([
        # Active months
        pl.len().alias("n_active_months"),
        # Monthly volume stats
        pl.col("monthly_volume").mean().alias("monthly_vol_mean"),
        pl.col("monthly_volume").std().alias("monthly_vol_std"),
        pl.col("monthly_volume").max().alias("monthly_vol_max"),
        pl.col("monthly_volume").min().alias("monthly_vol_min"),
        # Monthly txn count stats
        pl.col("monthly_txns").mean().alias("monthly_txns_mean"),
        pl.col("monthly_txns").std().alias("monthly_txns_std"),
        pl.col("monthly_txns").max().alias("monthly_txns_max"),
        # Max single txn in peak month
        pl.col("monthly_max").max().alias("peak_monthly_max_txn"),
    ])
)

# Velocity spike ratio: max month / mean month
monthly_features = monthly_features.with_columns([
    (pl.col("monthly_vol_max") / (pl.col("monthly_vol_mean") + 1e-9)).alias("vol_spike_ratio"),
    (pl.col("monthly_txns_max").cast(pl.Float64) / (pl.col("monthly_txns_mean") + 1e-9)).alias("txn_spike_ratio"),
    (pl.col("monthly_vol_std") / (pl.col("monthly_vol_mean") + 1e-9)).alias("monthly_vol_cv"),
])

# Dormancy: count months with zero txns (total 60 months from Jul 2020 to Jun 2025)
monthly_features = monthly_features.with_columns(
    (60 - pl.col("n_active_months")).alias("n_dormant_months"),
)

print(f"Monthly features: {monthly_features.shape}")
del monthly_all
gc.collect()

# ═══════════════════════════════════════════════════════════════════════
# PASS 2 — Rapid pass-through proxy
# Instead of full sort+shift (OOM), use per-account hourly buckets:
# count hours where both credits and debits occur = pass-through signal
# ═══════════════════════════════════════════════════════════════════════
passthrough_partials = []

for batch in ["batch-1", "batch-2", "batch-3", "batch-4"]:
    print(f"[Passthrough] Processing {batch} ...")

    agg = (
        pl.scan_parquet(DATA / "transactions" / batch / "*.parquet")
        .filter(pl.col("account_id").is_in(target_ids))
        .with_columns([
            pl.col("transaction_timestamp").str.to_datetime().alias("ts"),
            pl.col("amount").abs().alias("abs_amount"),
        ])
        .with_columns(
            pl.col("ts").dt.truncate("1h").alias("hour_bucket")
        )
        .group_by(["account_id", "hour_bucket"])
        .agg([
            (pl.col("txn_type") == "C").sum().alias("n_credits"),
            (pl.col("txn_type") == "D").sum().alias("n_debits"),
            pl.col("abs_amount").filter(pl.col("txn_type") == "C").sum().alias("credit_vol"),
            pl.col("abs_amount").filter(pl.col("txn_type") == "D").sum().alias("debit_vol"),
        ])
        .collect()
    )

    # Hours with both credits and debits = potential pass-through
    pt_hours = agg.filter(
        (pl.col("n_credits") > 0) & (pl.col("n_debits") > 0)
    )

    pt_agg = (
        pt_hours
        .group_by("account_id")
        .agg([
            pl.len().alias("passthrough_hours"),
            # Hours where credit ~= debit volume (within 20%)
            ((pl.col("credit_vol") - pl.col("debit_vol")).abs() /
             (pl.col("credit_vol") + 1e-9) < 0.2).sum().alias("matched_passthrough_hours"),
        ])
    )
    passthrough_partials.append(pt_agg)
    print(f"  {batch}: {pt_agg.shape[0]} accounts with pass-through hours")
    del agg, pt_hours
    gc.collect()

passthrough_combined = (
    pl.concat(passthrough_partials)
    .group_by("account_id")
    .agg([
        pl.col("passthrough_hours").sum(),
        pl.col("matched_passthrough_hours").sum(),
    ])
)
del passthrough_partials
gc.collect()

# ═══════════════════════════════════════════════════════════════════════
# PASS 3 — Counterparty concentration & fan-in/fan-out per window
# ═══════════════════════════════════════════════════════════════════════
cparty_partials = []

for batch in ["batch-1", "batch-2", "batch-3", "batch-4"]:
    print(f"[Counterparty] Processing {batch} ...")

    agg = (
        pl.scan_parquet(DATA / "transactions" / batch / "*.parquet")
        .filter(
            pl.col("account_id").is_in(target_ids)
            & pl.col("counterparty_id").is_not_null()
        )
        .group_by(["account_id", "counterparty_id"])
        .agg([
            pl.len().alias("pair_count"),
            pl.col("amount").abs().sum().alias("pair_volume"),
        ])
        .collect()
    )
    cparty_partials.append(agg)
    print(f"  {batch}: {agg.shape[0]} account-counterparty pairs")

cparty_all = pl.concat(cparty_partials).group_by(["account_id", "counterparty_id"]).agg([
    pl.col("pair_count").sum(),
    pl.col("pair_volume").sum(),
])
del cparty_partials
gc.collect()

# Per-account counterparty concentration
cparty_features = (
    cparty_all
    .group_by("account_id")
    .agg([
        pl.col("pair_count").max().alias("top_cparty_txn_count"),
        pl.col("pair_volume").max().alias("top_cparty_volume"),
        pl.col("pair_count").sum().alias("total_cparty_txns"),
        pl.col("pair_volume").sum().alias("total_cparty_volume"),
        pl.len().alias("n_unique_counterparties"),
    ])
)
cparty_features = cparty_features.with_columns([
    (pl.col("top_cparty_txn_count") / (pl.col("total_cparty_txns") + 1e-9))
        .alias("top_cparty_concentration"),
    (pl.col("top_cparty_volume") / (pl.col("total_cparty_volume") + 1e-9))
        .alias("top_cparty_volume_share"),
])
del cparty_all
gc.collect()

# ═══════════════════════════════════════════════════════════════════════
# PASS 4 — First/last activity timestamps (for temporal window prediction)
# ═══════════════════════════════════════════════════════════════════════
time_range_partials = []

for batch in ["batch-1", "batch-2", "batch-3", "batch-4"]:
    print(f"[TimeRange] Processing {batch} ...")

    agg = (
        pl.scan_parquet(DATA / "transactions" / batch / "*.parquet")
        .filter(pl.col("account_id").is_in(target_ids))
        .with_columns(pl.col("transaction_timestamp").str.to_datetime().alias("ts"))
        .group_by("account_id")
        .agg([
            pl.col("ts").min().alias("first_txn_ts"),
            pl.col("ts").max().alias("last_txn_ts"),
        ])
        .collect()
    )
    time_range_partials.append(agg)

time_range = (
    pl.concat(time_range_partials)
    .group_by("account_id")
    .agg([
        pl.col("first_txn_ts").min(),
        pl.col("last_txn_ts").max(),
    ])
)
time_range = time_range.with_columns(
    (pl.col("last_txn_ts") - pl.col("first_txn_ts")).dt.total_days().alias("activity_span_days")
)
del time_range_partials
gc.collect()

# Save time_range separately (needed for temporal window prediction in step 7)
time_range.write_parquet(FEAT / "time_range.parquet")
print(f"Time range: {time_range.shape}")

# ═══════════════════════════════════════════════════════════════════════
# Merge all behavior features
# ═══════════════════════════════════════════════════════════════════════
behavior_features = (
    monthly_features
    .join(passthrough_combined, on="account_id", how="left")
    .join(cparty_features, on="account_id", how="left")
    .join(time_range.select(["account_id", "activity_span_days"]), on="account_id", how="left")
)

# Drop _right columns from outer joins
right_cols = [c for c in behavior_features.columns if c.endswith("_right")]
if right_cols:
    behavior_features = behavior_features.drop(right_cols)

behavior_features.write_parquet(FEAT / "behavior_features.parquet")
print(f"\n✓ features/behavior_features.parquet saved ({behavior_features.shape})")
