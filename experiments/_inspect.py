import polars as pl
from pathlib import Path
DATA = Path(r"C:\Users\ujjaw\Downloads\AML_Mule_Project\data\archive")

# Check transactions_additional schema
df = pl.read_parquet(DATA / "transactions_additional" / "batch-1" / "part_0001.parquet")
print("=== transactions_additional schema ===")
print(df.schema)
print(df.head(5))
print(f"Shape: {df.shape}")

# Check transactions schema
tx = pl.read_parquet(DATA / "transactions" / "batch-1" / "part_0001.parquet")
print("\n=== transactions schema ===")
print(tx.schema)
print(f"Shape: {tx.shape}")

# Check train_labels fully
labels = pl.read_parquet(DATA / "train_labels.parquet")
print("\n=== train_labels ===")
print(labels.schema)
print(labels.head(5))
mule_count = labels.filter(pl.col("is_mule") == 1).shape[0]
nonmule_count = labels.filter(pl.col("is_mule") == 0).shape[0]
print(f"Mule count: {mule_count}")
print(f"Non-mule: {nonmule_count}")

# Check mule_flag_date distribution
mules = labels.filter(pl.col("is_mule") == 1)
print(f"\nmule_flag_date nulls: {mules['mule_flag_date'].null_count()}")
print(mules.select("mule_flag_date", "alert_reason").head(10))
print(f"\nalert_reason values:")
print(mules["alert_reason"].value_counts().sort("count", descending=True))

# flagged_by_branch
print(f"\nflagged_by_branch unique: {mules['flagged_by_branch'].n_unique()}")
