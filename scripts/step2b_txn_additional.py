"""
Step 2b — Features from transactions_additional
Extracts: balance volatility, IP diversity, geo-spread.
Memory-efficient: processes one txn part file at a time.
"""
import polars as pl
import gc
import os
from pathlib import Path

ROOT = Path(r"C:\Users\ujjaw\Downloads\AML_Mule_Project")
DATA = ROOT / "data" / "archive"
FEAT = ROOT / "features"
FEAT.mkdir(parents=True, exist_ok=True)

target_ids = pl.concat([
    pl.read_parquet(DATA / "train_labels.parquet").select("account_id"),
    pl.read_parquet(DATA / "test_accounts.parquet").select("account_id"),
]).unique()["account_id"]

balance_partials = []
ip_partials = []
geo_partials = []

for batch in ["batch-1", "batch-2", "batch-3", "batch-4"]:
    print(f"\nProcessing {batch} ...")

    txn_dir = DATA / "transactions" / batch
    add_dir = DATA / "transactions_additional" / batch
    txn_parts = sorted([f for f in os.listdir(txn_dir) if f.endswith(".parquet")])

    for pi, part_file in enumerate(txn_parts):
        if (pi + 1) % 25 == 0:
            print(f"  part {pi+1}/{len(txn_parts)} ...")

        # Load one txn part -> get account_id + transaction_id
        txn = pl.read_parquet(
            txn_dir / part_file,
            columns=["transaction_id", "account_id"]
        )
        txn = txn.filter(pl.col("account_id").is_in(target_ids))
        if txn.shape[0] == 0:
            continue

        txn_id_set = txn["transaction_id"]

        # Scan ALL additional parts for this batch and filter
        add = (
            pl.scan_parquet(str(add_dir / "*.parquet"))
            .filter(pl.col("transaction_id").is_in(txn_id_set))
            .select([
                "transaction_id",
                "balance_after_transaction",
                "ip_address",
                "latitude",
                "longitude",
            ])
            .collect()
        )

        if add.shape[0] == 0:
            del txn, txn_id_set
            continue

        merged = add.join(txn.select(["transaction_id", "account_id"]),
                          on="transaction_id", how="inner")
        del txn, txn_id_set, add
        gc.collect()

        # Balance
        bal = (
            merged.filter(pl.col("balance_after_transaction").is_not_null())
            .group_by("account_id")
            .agg([
                pl.col("balance_after_transaction").std().alias("balance_std"),
                pl.col("balance_after_transaction").min().alias("balance_min"),
                pl.col("balance_after_transaction").max().alias("balance_max"),
                pl.col("balance_after_transaction").mean().alias("balance_mean"),
                (pl.col("balance_after_transaction").abs() < 100).sum().alias("near_zero_balance_count"),
                (pl.col("balance_after_transaction") < 0).sum().alias("negative_balance_count"),
                pl.len().alias("_bal_txns"),
            ])
        )
        if bal.shape[0] > 0:
            balance_partials.append(bal)

        # IP
        ip = (
            merged.filter(pl.col("ip_address").is_not_null())
            .group_by("account_id")
            .agg([
                pl.col("ip_address").n_unique().alias("n_unique_ips"),
                pl.len().alias("_ip_txns"),
            ])
        )
        if ip.shape[0] > 0:
            ip_partials.append(ip)

        # Geo
        geo = (
            merged.filter(pl.col("latitude").is_not_null())
            .group_by("account_id")
            .agg([
                pl.col("latitude").std().alias("lat_std"),
                pl.col("longitude").std().alias("lon_std"),
                pl.col("latitude").min().alias("lat_min"),
                pl.col("latitude").max().alias("lat_max"),
                pl.col("longitude").min().alias("lon_min"),
                pl.col("longitude").max().alias("lon_max"),
                pl.len().alias("_geo_txns"),
            ])
        )
        if geo.shape[0] > 0:
            geo_partials.append(geo)

        del merged
        gc.collect()

    print(f"  {batch} done")

# ── Combine across all parts ─────────────────────────────────────────
print("\nCombining across all parts...")

balance_combined = (
    pl.concat(balance_partials)
    .group_by("account_id")
    .agg([
        pl.col("balance_std").mean(),
        pl.col("balance_min").min(),
        pl.col("balance_max").max(),
        pl.col("balance_mean").mean(),
        pl.col("near_zero_balance_count").sum(),
        pl.col("negative_balance_count").sum(),
        pl.col("_bal_txns").sum(),
    ])
)
balance_combined = balance_combined.with_columns([
    (pl.col("balance_max") - pl.col("balance_min")).alias("balance_range"),
    (pl.col("near_zero_balance_count") / (pl.col("_bal_txns") + 1e-9)).alias("near_zero_ratio"),
    (pl.col("balance_std") / (pl.col("balance_mean").abs() + 1e-9)).alias("balance_cv"),
]).drop("_bal_txns")

ip_combined = (
    pl.concat(ip_partials)
    .group_by("account_id")
    .agg([
        pl.col("n_unique_ips").sum(),
        pl.col("_ip_txns").sum(),
    ])
)
ip_combined = ip_combined.with_columns(
    (pl.col("n_unique_ips") / (pl.col("_ip_txns") + 1e-9)).alias("ip_per_txn_ratio")
).drop("_ip_txns")

geo_combined = (
    pl.concat(geo_partials)
    .group_by("account_id")
    .agg([
        pl.col("lat_std").mean(),
        pl.col("lon_std").mean(),
        pl.col("lat_min").min(),
        pl.col("lat_max").max(),
        pl.col("lon_min").min(),
        pl.col("lon_max").max(),
    ])
)
geo_combined = geo_combined.with_columns(
    (((pl.col("lat_max") - pl.col("lat_min")) ** 2 +
      (pl.col("lon_max") - pl.col("lon_min")) ** 2) ** 0.5)
    .alias("geo_spread")
)

# ── Merge and save ───────────────────────────────────────────────────
# Use left joins from balance (largest) to avoid duplicate key columns
all_accounts = pl.concat([
    balance_combined.select("account_id"),
    ip_combined.select("account_id"),
    geo_combined.select("account_id"),
]).unique()

additional_features = (
    all_accounts
    .join(balance_combined, on="account_id", how="left")
    .join(ip_combined, on="account_id", how="left")
    .join(geo_combined, on="account_id", how="left")
)

right_cols = [c for c in additional_features.columns if c.endswith("_right")]
if right_cols:
    additional_features = additional_features.drop(right_cols)

additional_features.write_parquet(FEAT / "txn_additional_features.parquet")
print(f"\n✓ features/txn_additional_features.parquet saved ({additional_features.shape})")
