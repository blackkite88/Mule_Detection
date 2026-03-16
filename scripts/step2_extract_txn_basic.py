"""
Step 2 — Basic Transaction Aggregation
Processes 400 M rows in batches, computes volume / count / max / min / variance.
"""
import polars as pl
from pathlib import Path

ROOT = Path(r"C:\Users\ujjaw\Downloads\AML_Mule_Project")
DATA = ROOT / "data" / "archive"
FEAT = ROOT / "features"
FEAT.mkdir(parents=True, exist_ok=True)

# target accounts
target_ids = pl.concat([
    pl.read_parquet(DATA / "train_labels.parquet").select("account_id"),
    pl.read_parquet(DATA / "test_accounts.parquet").select("account_id"),
]).unique()["account_id"]

partial_aggs = []

for batch in ["batch-1", "batch-2", "batch-3", "batch-4"]:
    print(f"Processing transactions/{batch} ...")

    lf = pl.scan_parquet(DATA / "transactions" / batch / "*.parquet")

    agg = (
        lf
        .filter(pl.col("account_id").is_in(target_ids))
        .with_columns([
            pl.col("amount").abs().alias("abs_amount"),
            (pl.col("amount") < 0).cast(pl.Int64).alias("is_reversal"),
            (pl.col("txn_type") == "C").cast(pl.Int64).alias("is_credit"),
            (pl.col("txn_type") == "D").cast(pl.Int64).alias("is_debit"),
        ])
        .group_by("account_id")
        .agg([
            pl.col("abs_amount").sum().alias("total_volume"),
            (pl.col("abs_amount") ** 2).sum().alias("sum_sq_amount"),
            pl.len().alias("total_txns"),
            pl.col("abs_amount").max().alias("max_txn_amount"),
            pl.col("abs_amount").min().alias("min_txn_amount"),
            pl.col("is_reversal").sum().alias("reversal_count"),
            pl.col("is_credit").sum().alias("credit_count"),
            pl.col("is_debit").sum().alias("debit_count"),
            pl.col("abs_amount").filter(pl.col("txn_type") == "C").sum().alias("credit_volume"),
            pl.col("abs_amount").filter(pl.col("txn_type") == "D").sum().alias("debit_volume"),
            pl.col("channel").n_unique().alias("n_channels"),
            pl.col("counterparty_id").n_unique().alias("n_counterparties"),
        ])
        .collect()
    )

    partial_aggs.append(agg)
    print(f"  {batch}: {agg.shape[0]} accounts")

combined = pl.concat(partial_aggs)

# Re-aggregate across batches — use correct aggregation per metric
txn_features = (
    combined
    .group_by("account_id")
    .agg([
        pl.col("total_volume").sum(),
        pl.col("sum_sq_amount").sum(),
        pl.col("total_txns").sum(),
        pl.col("max_txn_amount").max(),      # max of maxes
        pl.col("min_txn_amount").min(),      # min of mins
        pl.col("reversal_count").sum(),
        pl.col("credit_count").sum(),
        pl.col("debit_count").sum(),
        pl.col("credit_volume").sum(),
        pl.col("debit_volume").sum(),
        pl.col("n_channels").max(),          # approximate union
        pl.col("n_counterparties").max(),    # approximate
    ])
)

# Derived features
txn_features = txn_features.with_columns([
    (pl.col("total_volume") / pl.col("total_txns")).alias("avg_txn_amount"),
    (pl.col("reversal_count") / pl.col("total_txns")).alias("reversal_ratio"),
    (pl.col("credit_volume") / (pl.col("credit_volume") + pl.col("debit_volume") + 1e-9))
        .alias("credit_debit_ratio"),
    (pl.col("max_txn_amount") / (pl.col("total_volume") / pl.col("total_txns") + 1e-9))
        .alias("max_to_avg_ratio"),
])

txn_features.write_parquet(FEAT / "txn_basic.parquet")
print(f"\n✓ features/txn_basic.parquet saved  ({txn_features.shape})")
