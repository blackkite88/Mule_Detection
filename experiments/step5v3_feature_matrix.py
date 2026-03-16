"""
Step 5v3 — V3 Feature Matrix Assembly

Identical to step5v2 but adds recency_features.parquet (step2g output).
Produces train_features_v3.parquet / test_features_v3.parquet.

Run after:  python scripts/step2g_recency_features.py
"""
import polars as pl
from pathlib import Path

ROOT = Path(r"C:\Users\ujjaw\Downloads\AML_Mule_Project")
DATA = ROOT / "data" / "archive"
FEAT = ROOT / "features"

# ── Load feature tables ──────────────────────────────────────────────
master         = pl.read_parquet(FEAT / "master_static.parquet")
txn_basic      = pl.read_parquet(FEAT / "txn_basic.parquet")
txn_adv        = pl.read_parquet(FEAT / "txn_advanced.parquet")
graph_ft       = pl.read_parquet(FEAT / "graph_features.parquet")
graph_label_ft = pl.read_parquet(FEAT / "graph_label_features.parquet")
add_ft         = pl.read_parquet(FEAT / "txn_additional_features.parquet")
behav_ft       = pl.read_parquet(FEAT / "behavior_features.parquet")
te_ft          = pl.read_parquet(FEAT / "txn_oof_te_features.parquet")
neighbor       = pl.read_parquet(FEAT / "neighbor_risk.parquet")
recency_ft     = pl.read_parquet(FEAT / "recency_features.parquet")   # ← NEW

print(f"master:     {master.shape}")
print(f"txn_basic:  {txn_basic.shape}")
print(f"txn_adv:    {txn_adv.shape}")
print(f"graph_ft:   {graph_ft.shape}")
print(f"add_ft:     {add_ft.shape}")
print(f"behav_ft:   {behav_ft.shape}")
print(f"te_ft:      {te_ft.shape}")
print(f"recency_ft: {recency_ft.shape}")

# ── Join all ─────────────────────────────────────────────────────────
features = (
    master
    .join(txn_basic,       on="account_id", how="left")
    .join(txn_adv,         on="account_id", how="left")
    .join(graph_ft,        on="account_id", how="left")
    .join(graph_label_ft,  on="account_id", how="left")
    .join(add_ft,          on="account_id", how="left")
    .join(behav_ft,        on="account_id", how="left")
    .join(te_ft,           on="account_id", how="left")
    .join(neighbor,        on="account_id", how="left")
    .join(recency_ft,      on="account_id", how="left")   # ← NEW
)

print(f"\nJoined: {features.shape}")

# ── Drop IDs, dates, PII ─────────────────────────────────────────────
DROP_COLS = [
    "customer_id", "branch_code", "branch_pin", "customer_pin", "permanent_pin",
    "branch_address", "branch_pin_code",
    "account_opening_date", "freeze_date", "unfreeze_date",
    "last_mobile_update_date", "last_kyc_date",
    "date_of_birth", "relationship_start_date",
    "address_last_update_date", "passbook_last_update_date",
    # drop totals used only as ratio denominators (already encoded in ratios)
    "total_volume", "total_txns", "n_counterparties",
]
DROP_COLS += [c for c in features.columns if c.endswith("_right")]
drop_existing = [c for c in DROP_COLS if c in features.columns]
features = features.drop(drop_existing)

print(f"After dropping IDs/dates: {features.shape}")

# ── Encode Y/N binary flags → 0/1 ───────────────────────────────────
for c in features.columns:
    if features[c].dtype == pl.Utf8:
        unique_vals = set(features[c].drop_nulls().unique().to_list())
        if unique_vals <= {"Y", "N"}:
            features = features.with_columns(
                (pl.col(c) == "Y").cast(pl.Int8).alias(c)
            )

# Remaining string columns → integer codes
str_cols = [c for c in features.columns
            if features[c].dtype == pl.Utf8 and c != "account_id"]
print(f"\nEncoding {len(str_cols)} string columns: {str_cols}")
for c in str_cols:
    features = features.with_columns(
        pl.col(c).cast(pl.Categorical).to_physical().alias(c)
    )

# ── Split train / test ───────────────────────────────────────────────
labels   = pl.read_parquet(DATA / "train_labels.parquet")
test_ids = pl.read_parquet(DATA / "test_accounts.parquet")

train_df = features.join(labels,    on="account_id", how="inner")
test_df  = features.join(test_ids,  on="account_id", how="inner")

# Drop label leakage
label_leak = ["mule_flag_date", "alert_reason", "flagged_by_branch"]
label_leak_exist = [c for c in label_leak if c in train_df.columns]
train_df = train_df.drop(label_leak_exist)

mule_rate = train_df["is_mule"].mean()
print(f"\ntrain_df: {train_df.shape}  (is_mule mean: {mule_rate:.4f})")
print(f"test_df:  {test_df.shape}")

feat_cols = [c for c in train_df.columns if c not in {"account_id", "is_mule"}]
print(f"Feature count: {len(feat_cols)}")
new_feats = [c for c in feat_cols if c.startswith("recent") or c in {"is_active_recent", "vol_acceleration"}]
print(f"New recency features: {new_feats}")

train_df.write_parquet(FEAT / "train_features_v3.parquet")
test_df.write_parquet(FEAT / "test_features_v3.parquet")
print("\n✓ features/train_features_v3.parquet saved")
print("✓ features/test_features_v3.parquet saved")
