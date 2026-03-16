"""
Step 1 — Static Features
Joins accounts, customers, demographics, product details, branch info
into a single master_static.parquet feature table.
"""
import polars as pl
from pathlib import Path

ROOT = Path(r"C:\Users\ujjaw\Downloads\AML_Mule_Project")
DATA = ROOT / "data" / "archive"
FEAT = ROOT / "features"
FEAT.mkdir(parents=True, exist_ok=True)

# ── load tables ──────────────────────────────────────────────────────
accounts      = pl.read_parquet(DATA / "accounts.parquet")
customers     = pl.read_parquet(DATA / "customers.parquet")
linkage       = pl.read_parquet(DATA / "customer_account_linkage.parquet")
product_det   = pl.read_parquet(DATA / "product_details.parquet")
demographics  = pl.read_parquet(DATA / "demographics.parquet")
accts_add     = pl.read_parquet(DATA / "accounts-additional.parquet")
branch        = pl.read_parquet(DATA / "branch.parquet")
train_labels  = pl.read_parquet(DATA / "train_labels.parquet")
test_accounts = pl.read_parquet(DATA / "test_accounts.parquet")

print(f"accounts:     {accounts.shape}")
print(f"customers:    {customers.shape}")
print(f"linkage:      {linkage.shape}")
print(f"product_det:  {product_det.shape}")
print(f"demographics: {demographics.shape}")
print(f"accts_add:    {accts_add.shape}")
print(f"branch:       {branch.shape}")
print(f"train_labels: {train_labels.shape}")
print(f"test_accounts:{test_accounts.shape}")

# ── target account ids ───────────────────────────────────────────────
all_account_ids = pl.concat([
    train_labels.select("account_id"),
    test_accounts.select("account_id"),
]).unique()

# ── multi-account flag per customer ──────────────────────────────────
multi_acct = (
    linkage
    .group_by("customer_id")
    .agg(pl.col("account_id").count().alias("account_count_per_customer"))
)
linkage_enriched = linkage.join(multi_acct, on="customer_id")

# ── drop PII columns from demographics (name, phone, address) ───────
demo_features = demographics.drop(["name", "phone_number", "address"])

# ── build master table ───────────────────────────────────────────────
master = (
    accounts
    .join(linkage_enriched,  on="account_id",   how="left")
    .join(customers,         on="customer_id",  how="left")
    .join(product_det,       on="customer_id",  how="left")
    .join(demo_features,     on="customer_id",  how="left")
    .join(accts_add,         on="account_id",   how="left")
    .join(branch,            on="branch_code",  how="left")
)

# keep only target accounts
master = master.join(all_account_ids, on="account_id", how="inner")

# ── derived features ─────────────────────────────────────────────────
master = master.with_columns([
    # account age in days
    (pl.lit("2025-06-30").str.to_date() - pl.col("account_opening_date").str.to_date())
        .dt.total_days().alias("account_age_days"),
    # customer age from date_of_birth
    (pl.lit("2025-06-30").str.to_date() - pl.col("date_of_birth").str.to_date())
        .dt.total_days().alias("customer_age_days"),
    # was ever frozen
    pl.col("freeze_date").is_not_null().cast(pl.Int8).alias("was_frozen"),
])

print(f"\nMaster table shape: {master.shape}")
master.write_parquet(FEAT / "master_static.parquet")
print("✓ features/master_static.parquet saved")
