"""
Step 3 — Advanced Transaction Features
Inter-arrival time stats, round-amount ratio, structuring detection,
channel entropy, temporal patterns.
Memory-efficient: processes IAT and temporal in separate passes.
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
# PASS 1 — Inter-arrival time features (needs sort + shift → heavy RAM)
# ═══════════════════════════════════════════════════════════════════════
iat_partials = []

for batch in ["batch-1", "batch-2", "batch-3", "batch-4"]:
    print(f"[IAT] Processing transactions/{batch} ...")

    df = (
        pl.scan_parquet(DATA / "transactions" / batch / "*.parquet")
        .filter(pl.col("account_id").is_in(target_ids))
        .select(["account_id", "transaction_timestamp"])
        .with_columns(
            pl.col("transaction_timestamp").str.to_datetime().alias("ts")
        )
        .drop("transaction_timestamp")
        .sort(["account_id", "ts"])
        .collect()
    )

    df = df.with_columns(
        (pl.col("ts") - pl.col("ts").shift(1).over("account_id"))
        .dt.total_seconds()
        .alias("iat")
    )

    iat_agg = (
        df.filter(pl.col("iat").is_not_null())
        .group_by("account_id")
        .agg([
            pl.col("iat").mean().alias("iat_mean"),
            pl.col("iat").std().alias("iat_std"),
            pl.col("iat").min().alias("iat_min"),
            pl.col("iat").median().alias("iat_median"),
            (pl.col("iat") < 60).sum().alias("burst_count_60s"),
            (pl.col("iat") < 300).sum().alias("burst_count_5min"),
        ])
    )
    iat_partials.append(iat_agg)
    print(f"  {batch}: {df.shape[0]} txns, {iat_agg.shape[0]} accounts")
    del df
    gc.collect()

iat_combined = (
    pl.concat(iat_partials)
    .group_by("account_id")
    .agg([
        pl.col("iat_mean").mean(),
        pl.col("iat_std").mean(),
        pl.col("iat_min").min(),
        pl.col("iat_median").mean().alias("iat_median"),
        pl.col("burst_count_60s").sum(),
        pl.col("burst_count_5min").sum(),
    ])
)
del iat_partials
gc.collect()
print(f"IAT combined: {iat_combined.shape}")

# ═══════════════════════════════════════════════════════════════════════
# PASS 2 — Temporal / pattern features (lighter — no sort needed)
# ═══════════════════════════════════════════════════════════════════════
temporal_partials = []

for batch in ["batch-1", "batch-2", "batch-3", "batch-4"]:
    print(f"[Temporal] Processing transactions/{batch} ...")

    agg = (
        pl.scan_parquet(DATA / "transactions" / batch / "*.parquet")
        .filter(pl.col("account_id").is_in(target_ids))
        .with_columns([
            pl.col("transaction_timestamp").str.to_datetime().alias("ts"),
            pl.col("amount").abs().alias("abs_amount"),
        ])
        .with_columns([
            pl.col("ts").dt.hour().alias("hour"),
            pl.col("ts").dt.weekday().alias("weekday"),
        ])
        .group_by("account_id")
        .agg([
            ((pl.col("hour") >= 22) | (pl.col("hour") < 6))
                .sum().alias("night_txn_count"),
            (pl.col("weekday") >= 5).sum().alias("weekend_txn_count"),
            (pl.col("abs_amount") % 1000 == 0).sum().alias("round_1k_count"),
            ((pl.col("abs_amount") >= 45000) & (pl.col("abs_amount") < 50000))
                .sum().alias("structuring_count"),
            pl.len().alias("_batch_txns"),
            pl.col("channel").n_unique().alias("_n_channels_batch"),
            pl.col("mcc_code").n_unique().alias("n_mcc_codes"),
        ])
        .collect()
    )
    temporal_partials.append(agg)
    print(f"  {batch}: {agg.shape[0]} accounts")

temporal_combined = (
    pl.concat(temporal_partials)
    .group_by("account_id")
    .agg([
        pl.col("night_txn_count").sum(),
        pl.col("weekend_txn_count").sum(),
        pl.col("round_1k_count").sum(),
        pl.col("structuring_count").sum(),
        pl.col("_batch_txns").sum(),
        pl.col("n_mcc_codes").max(),
    ])
)

temporal_combined = temporal_combined.with_columns([
    (pl.col("night_txn_count") / (pl.col("_batch_txns") + 1e-9)).alias("night_ratio"),
    (pl.col("weekend_txn_count") / (pl.col("_batch_txns") + 1e-9)).alias("weekend_ratio"),
    (pl.col("round_1k_count") / (pl.col("_batch_txns") + 1e-9)).alias("round_1k_ratio"),
]).drop("_batch_txns")

# ── Merge and save ───────────────────────────────────────────────────
adv_features = iat_combined.join(temporal_combined, on="account_id", how="outer")

adv_features.write_parquet(FEAT / "txn_advanced.parquet")
print(f"\n✓ features/txn_advanced.parquet saved  ({adv_features.shape})")
