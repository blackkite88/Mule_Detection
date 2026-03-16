import polars as pl
from pathlib import Path
DATA = Path(r"C:\Users\ujjaw\Downloads\AML_Mule_Project\data\archive")
labels = pl.read_parquet(DATA / "train_labels.parquet")
mules = labels.filter(pl.col("is_mule") == 1)
print(mules["alert_reason"].value_counts().sort("count", descending=True))
print()
print("mule_flag_date range:")
mn = mules["mule_flag_date"].min()
mx = mules["mule_flag_date"].max()
print(f"  min: {mn}")
print(f"  max: {mx}")

future = mules.filter(pl.col("mule_flag_date").str.to_date() > pl.lit("2025-06-30").str.to_date())
print(f"  Mules flagged after 2025-06-30: {future.shape[0]}")

# Check how many txn_additional parts per batch
import os
for b in ["batch-1","batch-2","batch-3","batch-4"]:
    n = len(os.listdir(DATA / "transactions_additional" / b))
    print(f"  transactions_additional/{b}: {n} parts")

# Ratio of txn_additional rows to transactions rows
tx1 = pl.read_parquet(DATA / "transactions" / "batch-1" / "part_0001.parquet")
ta1 = pl.read_parquet(DATA / "transactions_additional" / "batch-1" / "part_0001.parquet")
print(f"\ntransactions part_0001: {tx1.shape[0]} rows")
print(f"transactions_additional part_0001: {ta1.shape[0]} rows")

# Check if transaction_id is shared
shared = tx1["transaction_id"].to_list()[:5]
print(f"\nSample txn IDs from transactions: {shared}")
matched = ta1.filter(pl.col("transaction_id").is_in(shared))
print(f"Matched in additional: {matched.shape[0]}")
print(matched)
