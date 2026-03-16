"""
Step 2g — Recency Features

Computes last 3-month and 6-month transaction windows vs. full history.
Data window: Jul 2020 – Jun 2025.  Cutoffs (end of dataset = 2025-07-01):
  last_3m: >= 2025-04-01
  last_6m: >= 2025-01-01

Key rationale:
  Active mule accounts show elevated RECENT activity. Dormant-then-active
  patterns and recent-vs-historical acceleration are strong AML signals.

Features produced:
  recent_3m_vol, recent_3m_txns, recent_3m_credit_vol, recent_3m_debit_vol
  recent_3m_night_txns, recent_3m_round_1k_count
  recent_6m_vol, recent_6m_txns
  recent_vol_ratio    = recent_3m_vol / (total_vol  + 1e-9)
  recent_txn_ratio    = recent_3m_txns / (total_txns + 1)
  is_active_recent    = (recent_3m_txns > 0)
  vol_acceleration    = recent_3m_vol / (recent_6m_vol - recent_3m_vol + 1e-9)
  recent_unique_cp    = distinct counterparties in recent 3m
  recent_cp_ratio     = recent_unique_cp / (n_counterparties_total + 1)

Inputs:  data/archive/transactions/batch-{1-4}/*.parquet
         features/txn_basic.parquet  (total_volume, total_txns)
Outputs: features/recency_features.parquet
"""
import polars as pl
import gc
from pathlib import Path

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

# Window boundaries (dataset ends 2025-06-30)
CUT_3M = "2025-04-01"   # >= this → last 3 months
CUT_6M = "2025-01-01"   # >= this → last 6 months

# ── Scan raw transactions once per batch ─────────────────────────────
parts_3m = []
parts_6m = []

for batch in ["batch-1", "batch-2", "batch-3", "batch-4"]:
    print(f"Processing {batch} ...")

    base = (
        pl.scan_parquet(DATA / "transactions" / batch / "*.parquet")
        .filter(pl.col("account_id").is_in(target_ids.implode()))
        .select(["account_id", "transaction_timestamp", "amount", "counterparty_id"])
        .with_columns([
            pl.col("transaction_timestamp").str.to_datetime().alias("ts"),
            pl.col("amount").cast(pl.Float64).alias("amount"),
        ])
        .with_columns([
            pl.col("ts").dt.strftime("%Y-%m-%d").alias("d"),
        ])
    )

    # Last-3m slice
    df_3m = (
        base.filter(pl.col("d") >= CUT_3M)
        .group_by("account_id")
        .agg([
            pl.col("amount").abs().sum().alias("recent_3m_vol"),
            pl.len().alias("recent_3m_txns"),
            pl.col("amount").filter(pl.col("amount") > 0).sum().alias("recent_3m_credit_vol"),
            pl.col("amount").filter(pl.col("amount") < 0).abs().sum().alias("recent_3m_debit_vol"),
            (pl.col("ts").dt.hour().is_between(22, 23) | pl.col("ts").dt.hour().is_between(0, 5))
                .sum().alias("recent_3m_night_txns"),
            (pl.col("amount").abs() % 1000 == 0).sum().alias("recent_3m_round_1k_count"),
            pl.col("counterparty_id").n_unique().alias("recent_3m_unique_cp"),
        ])
        .collect()
    )
    parts_3m.append(df_3m)

    # Last-6m slice
    df_6m = (
        base.filter(pl.col("d") >= CUT_6M)
        .group_by("account_id")
        .agg([
            pl.col("amount").abs().sum().alias("recent_6m_vol"),
            pl.len().alias("recent_6m_txns"),
        ])
        .collect()
    )
    parts_6m.append(df_6m)

    del df_3m, df_6m
    gc.collect()
    print(f"  {batch}: done")

# ── Aggregate across batches ─────────────────────────────────────────
print("Aggregating batches ...")

agg_3m = (
    pl.concat(parts_3m)
    .group_by("account_id")
    .agg([
        pl.col("recent_3m_vol").sum(),
        pl.col("recent_3m_txns").sum(),
        pl.col("recent_3m_credit_vol").sum(),
        pl.col("recent_3m_debit_vol").sum(),
        pl.col("recent_3m_night_txns").sum(),
        pl.col("recent_3m_round_1k_count").sum(),
        pl.col("recent_3m_unique_cp").sum(),   # over-counts; capped by total
    ])
)

agg_6m = (
    pl.concat(parts_6m)
    .group_by("account_id")
    .agg([
        pl.col("recent_6m_vol").sum(),
        pl.col("recent_6m_txns").sum(),
    ])
)

del parts_3m, parts_6m
gc.collect()

# ── Join with total-history totals for ratio computation ─────────────
txn_basic = pl.read_parquet(FEAT / "txn_basic.parquet").select([
    "account_id", "total_volume", "total_txns", "n_counterparties"
])

# All target accounts (fill missing → 0)
all_accounts = pl.DataFrame({"account_id": target_ids})

result = (
    all_accounts
    .join(agg_3m, on="account_id", how="left")
    .join(agg_6m, on="account_id", how="left")
    .join(txn_basic, on="account_id", how="left")
    .with_columns([
        pl.col("recent_3m_vol").fill_null(0.0),
        pl.col("recent_3m_txns").fill_null(0),
        pl.col("recent_3m_credit_vol").fill_null(0.0),
        pl.col("recent_3m_debit_vol").fill_null(0.0),
        pl.col("recent_3m_night_txns").fill_null(0),
        pl.col("recent_3m_round_1k_count").fill_null(0),
        pl.col("recent_3m_unique_cp").fill_null(0),
        pl.col("recent_6m_vol").fill_null(0.0),
        pl.col("recent_6m_txns").fill_null(0),
    ])
    .with_columns([
        # Ratio: what fraction of ALL history is in last 3 months
        (pl.col("recent_3m_vol") / (pl.col("total_volume") + 1e-9))
            .alias("recent_vol_ratio"),
        (pl.col("recent_3m_txns") / (pl.col("total_txns") + 1).cast(pl.Float64))
            .alias("recent_txn_ratio"),
        # Binary: any activity in last 3m  
        (pl.col("recent_3m_txns") > 0).cast(pl.Int8).alias("is_active_recent"),
        # Acceleration: last-3m vol vs prior-3m (6m minus 3m)
        (
            pl.col("recent_3m_vol") /
            (pl.col("recent_6m_vol") - pl.col("recent_3m_vol") + 1e-9)
        ).alias("vol_acceleration"),
        # Counterparty renewal rate
        (
            pl.col("recent_3m_unique_cp") /
            (pl.col("n_counterparties") + 1).cast(pl.Float64)
        ).alias("recent_cp_ratio"),
        # Night ratio in recent window
        (
            pl.col("recent_3m_night_txns") /
            (pl.col("recent_3m_txns") + 1).cast(pl.Float64)
        ).alias("recent_night_ratio"),
        # Round-1k ratio in recent window
        (
            pl.col("recent_3m_round_1k_count") /
            (pl.col("recent_3m_txns") + 1).cast(pl.Float64)
        ).alias("recent_round_1k_ratio"),
    ])
    .drop(["total_volume", "total_txns", "n_counterparties"])
)

print(f"\nResult shape: {result.shape}")
print("Columns:", result.columns)

# Verify no unintended nulls in derived cols
for c in result.columns:
    if c != "account_id":
        n_null = result[c].null_count()
        if n_null > 0:
            print(f"  WARNING: {c} has {n_null} nulls — filling 0")
            result = result.with_columns(pl.col(c).fill_null(0))

out_path = FEAT / "recency_features.parquet"
result.write_parquet(out_path)
print(f"\n✓ Saved {out_path}  shape={result.shape}")

# Quick stats on new features
feat_cols = [c for c in result.columns if c != "account_id"]
for c in feat_cols[:6]:
    col = result[c].cast(pl.Float64)
    print(f"  {c:35s}: mean={col.mean():.4f}  std={col.std():.4f}  max={col.max():.4f}")
