import polars as pl, os

DATA = "data/archive"

# 1. Transaction column schema
p = "data/archive/transactions/batch-1/part_0001.parquet"
df = pl.read_parquet(p, n_rows=5)
print("=== Transaction columns ===")
for c in df.columns:
    print(f"  {c}: {df[c].dtype}  e.g. {df[c][0]}")

# 2. Transaction_additional schema
p2 = "data/archive/transactions_additional/batch-1/part_0001.parquet"
df2 = pl.read_parquet(p2, n_rows=5)
print("\n=== Transaction_additional columns ===")
for c in df2.columns:
    print(f"  {c}: {df2[c].dtype}  e.g. {df2[c][0]}")

# 3. Train labels
lb = pl.read_parquet(f"{DATA}/train_labels.parquet")
print("\n=== Train labels ===")
print(lb.columns)
print(lb.head(3))
mule_ct = lb["is_mule"].sum()
total = lb.shape[0]
print(f"Mule count: {mule_ct} / {total}")

# 4. What features we currently have
train = pl.read_parquet("features/train_features_v2.parquet")
ncols = len(train.columns)
print(f"\n=== Current features ({ncols} cols) ===")
for c in sorted(train.columns):
    print(f"  {c}")

# 5. Check txn_type values
print("\n=== txn_type values ===")
print(df["txn_type"].unique())

# 6. Check channel values
print("\n=== channel values ===")
print(df["channel"].unique())

# 7. Check mcc_code cardinality
mcc = pl.scan_parquet("data/archive/transactions/batch-1/*.parquet").select("mcc_code").collect()
print(f"\n=== mcc_code unique: {mcc['mcc_code'].n_unique()} ===")
print(mcc["mcc_code"].value_counts().sort("count", descending=True).head(10))
