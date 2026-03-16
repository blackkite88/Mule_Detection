"""
Step 5 — Assemble Feature Matrix
Joins static + txn_basic + txn_advanced + graph features,
encodes categoricals, and produces train/test feature parquets.
"""
import polars as pl
from pathlib import Path

ROOT = Path(r"C:\Users\ujjaw\Downloads\AML_Mule_Project")
DATA = ROOT / "data" / "archive"
FEAT = ROOT / "features"

# ── Load feature tables ──────────────────────────────────────────────
master    = pl.read_parquet(FEAT / "master_static.parquet")
txn_basic = pl.read_parquet(FEAT / "txn_basic.parquet")
txn_adv   = pl.read_parquet(FEAT / "txn_advanced.parquet")
graph_ft  = pl.read_parquet(FEAT / "graph_features.parquet")

print(f"master:    {master.shape}")
print(f"txn_basic: {txn_basic.shape}")
print(f"txn_adv:   {txn_adv.shape}")
print(f"graph_ft:  {graph_ft.shape}")

# ── Join everything ──────────────────────────────────────────────────
features = (
    master
    .join(txn_basic, on="account_id", how="left")
    .join(txn_adv,   on="account_id", how="left")
    .join(graph_ft,  on="account_id", how="left")
)

# ── Identify column types ───────────────────────────────────────────
# Drop columns that are IDs, dates, or PII — not useful for modelling
DROP_COLS = [
    "customer_id", "branch_code", "branch_pin", "customer_pin", "permanent_pin",
    "branch_address", "branch_pin_code",
    "account_opening_date", "freeze_date", "unfreeze_date",
    "last_mobile_update_date", "last_kyc_date",
    "date_of_birth", "relationship_start_date",
    "address_last_update_date", "passbook_last_update_date",
]
# Also drop any spurious "_right" join-key columns
DROP_COLS += [c for c in features.columns if c.endswith("_right")]
# Only drop those that exist
drop_existing = [c for c in DROP_COLS if c in features.columns]
features = features.drop(drop_existing)

# ── Encode categorical / string columns ──────────────────────────────
# Binary Y/N flags → 0/1
yn_cols = [c for c in features.columns if features[c].dtype == pl.Utf8
           and features[c].drop_nulls().unique().to_list()
           in [["Y", "N"], ["N", "Y"], ["Y"], ["N"]]]

for c in yn_cols:
    features = features.with_columns(
        (pl.col(c) == "Y").cast(pl.Int8).alias(c)
    )

# Remaining string columns → integer codes for LightGBM
str_cols = [c for c in features.columns
            if features[c].dtype == pl.Utf8 and c != "account_id"]

print(f"\nEncoding {len(str_cols)} string columns: {str_cols}")
for c in str_cols:
    features = features.with_columns(
        pl.col(c).cast(pl.Categorical).to_physical().alias(c)
    )

# ── Split train / test ──────────────────────────────────────────────
labels = pl.read_parquet(DATA / "train_labels.parquet")
test_ids = pl.read_parquet(DATA / "test_accounts.parquet")

train_df = features.join(labels, on="account_id", how="inner")
test_df  = features.join(test_ids, on="account_id", how="inner")

# Drop label leakage columns from train
label_leak = ["mule_flag_date", "alert_reason", "flagged_by_branch"]
label_leak_exist = [c for c in label_leak if c in train_df.columns]
train_df = train_df.drop(label_leak_exist)

print(f"\ntrain_df: {train_df.shape}  (is_mule mean: {train_df['is_mule'].mean():.4f})")
print(f"test_df:  {test_df.shape}")

train_df.write_parquet(FEAT / "train_features.parquet")
test_df.write_parquet(FEAT / "test_features.parquet")
print("\n✓ features/train_features.parquet saved")
print("✓ features/test_features.parquet saved")
