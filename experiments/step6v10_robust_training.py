"""
Step 6v10 — Robust Training (Phase 7: Noisy Label Avoidance)
===========================================================
Trains a robust 5-fold CV ensemble of LightGBM and XGBoost.
Implements:
1. Ghost Mule Detection: Down-weights suspicious is_mule=0 accounts.
2. Regularization: Increases min_child requirements to ignore noisy labels.

Reads:  features/train_features_v2.parquet, features/test_features_v2.parquet
        features/shap_recommended_features.parquet
        features/txn_advanced.parquet
Writes: models/oof_preds_v10.npy, models/test_preds_v10.npy
"""
import polars as pl
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from pathlib import Path
import json, time, gc

ROOT   = Path(r"C:\Users\ujjaw\Downloads\AML_Mule_Project")
FEAT   = ROOT / "features"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

# 1. LOAD DATA & ENGINEER GHOST WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  Step 6v10 — Robust Training (The Winner's Polish)")
print("=" * 70)

# Load core tables
train_v2 = pl.read_parquet(FEAT / "train_features_v2.parquet")
test_v2  = pl.read_parquet(FEAT / "test_features_v2.parquet")
shap_ft  = pl.read_parquet(FEAT / "shap_recommended_features.parquet")
txn_adv  = pl.read_parquet(FEAT / "txn_advanced.parquet").select(["account_id", "burst_count_5min"])

# Merge
train = train_v2.join(shap_ft, on="account_id", how="left").join(txn_adv, on="account_id", how="left")
test  = test_v2.join(shap_ft, on="account_id", how="left").join(txn_adv, on="account_id", how="left")

# Identify Ghost Mules (Those labeled 0 but acting like extreme 1s)
# Thresholds derived from 95th percentile of actual mules
PASS_THROUGH_THRES = 0.18
BURST_THRES = 300

train = train.with_columns([
    pl.when((pl.col("is_mule") == 0) & ((pl.col("pass_through_ratio") > PASS_THROUGH_THRES) | (pl.col("burst_count_5min") > BURST_THRES)))
      .then(0.7)
      .otherwise(1.0)
      .alias("sample_weight")
])

n_ghosts = train.filter(pl.col("sample_weight") < 1.0).height / train.height
print(f"  Ghost Mules identified: {train.filter(pl.col('sample_weight') < 1.0).height} ({n_ghosts*100:.2f}%)")

# 2. FEATURE SELECTION (Pruned set + SHAP additions)
# ─────────────────────────────────────────────────────────────────────────────
FEATURES = [
    "customer_age", "branch_id", "account_age_days", "city_tier",
    "was_frozen", "missing_pan", "missing_aadhaar", "missing_mobile_upd",
    "txn_count_30d", "txn_count_7d", "unique_cp_30d", "unique_cp_7d",
    "total_credit_30d", "total_debit_30d", "credit_debit_ratio_30d",
    "avg_txn_interval_days", "balance_turnover_30d", "avg_balance", 
    "balance_volatility", "weekend_txn_ratio", "night_txn_ratio", "round_amount_ratio",
    "in_degree", "out_degree", "total_volume_in", "total_volume_out",
    "eigenvector_centrality", "pagerank", "hub_score", "authority_score",
    "network_ratio", "te_branch_id_mean", "te_mcc_mean", "te_mcc_std",
    "fe_branch_id", "fe_mcc", "txn_velocity_30d", "cp_concentration",
    "rolling_7d_zscore_max", "rolling_7d_zscore_mean", "cp_entropy",
    "days_to_first_txn", "turnover_ratio", "peak_day_conc_txn", "peak_day_conc_vol",
    "pass_through_ratio", "burst_count_5min"
]
FEATURES = [c for c in FEATURES if c in train.columns]

X_train = train.select(FEATURES).to_numpy().astype(np.float32)
y_train = train["is_mule"].to_numpy().astype(np.int32)
sw_train = train["sample_weight"].to_numpy().astype(np.float32)
X_test  = test.select(FEATURES).to_numpy().astype(np.float32)

X_train = np.nan_to_num(X_train, nan=0.0)
X_test  = np.nan_to_num(X_test,  nan=0.0)

n_pos = y_train.sum()
n_neg = len(y_train) - n_pos
scale_pos_weight = n_neg / max(n_pos, 1)

# 3. PARAMETERS (With Increased Regularization)
# ─────────────────────────────────────────────────────────────────────────────
# Bumping min_child parameters to 45 to ignore small noise clusters
lgb_params = {
    "n_estimators": 2000, "learning_rate": 0.02, "num_leaves": 127,
    "max_depth": -1, "min_child_samples": 45, "subsample": 0.8,
    "colsample_bytree": 0.7, "reg_alpha": 0.5, "reg_lambda": 2.0,
    "scale_pos_weight": scale_pos_weight, "random_state": 42,
    "n_jobs": -1, "verbose": -1,
}

xgb_params = {
    "n_estimators": 2000, "learning_rate": 0.02, "max_depth": 8,
    "min_child_weight": 45, "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.5, "reg_lambda": 2.0, "scale_pos_weight": scale_pos_weight,
    "random_state": 42, "n_jobs": -1, "tree_method": "hist",
    "eval_metric": "auc", "early_stopping_rounds": 100, "verbosity": 0,
}

# 4. 5-FOLD CV
# ─────────────────────────────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(y_train))
oof_xgb = np.zeros(len(y_train))
test_lgb = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    Xtr, Xval = X_train[train_idx], X_train[val_idx]
    ytr, yval = y_train[train_idx], y_train[val_idx]
    swtr = sw_train[train_idx]

    print(f"\nFOLD {fold} {'─'*45}")

    # LightGBM
    print("  Training LightGBM (Robust)...")
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(Xtr, ytr, sample_weight=swtr, eval_set=[(Xval, yval)], 
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)])
    oof_lgb[val_idx] = model_lgb.predict_proba(Xval)[:, 1]
    test_lgb += model_lgb.predict_proba(X_test)[:, 1] / 5

    # XGBoost
    print("  Training XGBoost (Robust)...")
    model_xgb = xgb.XGBClassifier(**xgb_params)
    model_xgb.fit(Xtr, ytr, sample_weight=swtr, eval_set=[(Xval, yval)], verbose=500)
    oof_xgb[val_idx] = model_xgb.predict_proba(Xval)[:, 1]
    test_xgb += model_xgb.predict_proba(X_test)[:, 1] / 5

    fold_mean = (oof_lgb[val_idx] + oof_xgb[val_idx]) / 2
    print(f"  FOLD {fold} AUC: {roc_auc_score(yval, fold_mean):.5f}")

# 5. BLENDING
# ─────────────────────────────────────────────────────────────────────────────
oof_final = (oof_lgb + oof_xgb) / 2
test_final = (test_lgb + test_xgb) / 2

print(f"\nFINAL OOF AUC: {roc_auc_score(y_train, oof_final):.6f}")

# Save
np.save(MODELS / "oof_preds_v10.npy", oof_final)
np.save(MODELS / "test_preds_v10.npy", test_final)

print(f"\n✓ Saved robust V10 artifacts to {MODELS}")
