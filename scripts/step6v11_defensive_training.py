"""
Step 6v11 — Defensive Training (Phase 8: Trap Immunity)
======================================================
Trains a highly robust model to pass all competition traps.
1. Feature Pruning: Strictly EXCLUDES Target Encoding (memorization trap).
2. Standardized Scaling: Uses fixed pass_through_ratio (0-1 scale).
3. Aggressive Ghost Detection: Weight = 0.2 for suspicious legit accounts.
4. Full Behavioral Signal: Restores the full set of 108 high-signal features.
"""
import polars as pl
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pathlib import Path
import time

ROOT   = Path(r"C:\Users\ujjaw\Downloads\AML_Mule_Project")
FEAT   = ROOT / "features"
MODELS = ROOT / "models"

# 1. LOAD DATA 
# ─────────────────────────────────────────────────────────────────────────────
print("Loading features...")
train_v2 = pl.read_parquet(FEAT / "train_features_v2.parquet")
test_v2  = pl.read_parquet(FEAT / "test_features_v2.parquet")
shap_v2  = pl.read_parquet(FEAT / "shap_recommended_features_v2.parquet")
txn_adv  = pl.read_parquet(FEAT / "txn_advanced.parquet").select(["account_id", "burst_count_5min"])

train = train_v2.join(shap_v2, on="account_id", how="left").join(txn_adv, on="account_id", how="left")
test  = test_v2.join(shap_v2, on="account_id", how="left").join(txn_adv, on="account_id", how="left")

# 2. DEFENSIVE GHOST DETECTION
# ─────────────────────────────────────────────────────────────────────────────
# Criteria for "Ghost Mules" using standardized scale (0.95 and high bursts)
PASS_THROUGH_THRES = 0.35  # Equivalent to very high pass-through on my 0-1 scale
BURST_THRES = 300

train = train.with_columns([
    pl.when((pl.col("is_mule") == 0) & ((pl.col("pass_through_ratio") > PASS_THROUGH_THRES) | (pl.col("burst_count_5min") > BURST_THRES)))
      .then(0.1)  # Aggressive down-weighting
      .otherwise(1.0)
      .alias("sample_weight")
])

n_ghosts = train.filter(pl.col("sample_weight") < 1.0).height
print(f"Ghost Mules identified: {n_ghosts} ({n_ghosts/len(train)*100:.2f}%)")

# 3. FEATURE SELECTION (Behavioral Only - No Target Encoding)
# ─────────────────────────────────────────────────────────────────────────────
# We exclude 'te_' and 'fe_' features which are trap vectors
all_cols = train.columns
EXCLUDE = ["account_id", "is_mule", "sample_weight"]
# Exclude Target Encoding and Frequency Encoding to stop memorization
EXCLUDE += [c for c in all_cols if c.startswith("te_") or c.startswith("fe_")]
# Exclude categorical raw strings
EXCLUDE += [c for c in all_cols if train[c].dtype == pl.Utf8]

FEATURES = [c for c in all_cols if c not in EXCLUDE]
print(f"Using {len(FEATURES)} behavioral features. (Target Encoding Removed)")

X_train = train.select(FEATURES).to_numpy().astype(np.float32)
y_train = train["is_mule"].to_numpy().astype(np.int32)
sw_train = train["sample_weight"].to_numpy().astype(np.float32)
X_test  = test.select(FEATURES).to_numpy().astype(np.float32)

X_train = np.nan_to_num(X_train, nan=0.0)
X_test  = np.nan_to_num(X_test,  nan=0.0)

# 4. ROBUST PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
lgb_params = {
    "n_estimators": 2500, "learning_rate": 0.015, "num_leaves": 95,
    "max_depth": 8, "min_child_samples": 50, "subsample": 0.7,
    "colsample_bytree": 0.6, "reg_alpha": 1.0, "reg_lambda": 5.0,
    "scale_pos_weight": (len(y_train)-y_train.sum())/y_train.sum(),
    "random_state": 42, "n_jobs": -1, "verbose": -1,
}

xgb_params = {
    "n_estimators": 2500, "learning_rate": 0.015, "max_depth": 6,
    "min_child_weight": 50, "subsample": 0.7, "colsample_bytree": 0.6,
    "reg_alpha": 1.0, "reg_lambda": 5.0, "scale_pos_weight": (len(y_train)-y_train.sum())/y_train.sum(),
    "random_state": 42, "n_jobs": -1, "tree_method": "hist",
    "eval_metric": "auc", "early_stopping_rounds": 100, "verbosity": 0,
}

# 5. TRAINING
# ─────────────────────────────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_final = np.zeros(len(y_train))
test_final = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"\nFOLD {fold}...")
    Xtr, Xval = X_train[train_idx], X_train[val_idx]
    ytr, yval = y_train[train_idx], y_train[val_idx]
    swtr = sw_train[train_idx]

    # LGB
    m_lgb = lgb.LGBMClassifier(**lgb_params)
    m_lgb.fit(Xtr, ytr, sample_weight=swtr, eval_set=[(Xval, yval)], 
              callbacks=[lgb.early_stopping(100)])
    
    # XGB
    m_xgb = xgb.XGBClassifier(**xgb_params)
    m_xgb.fit(Xtr, ytr, sample_weight=swtr, eval_set=[(Xval, yval)], verbose=False)

    p_lgb = m_lgb.predict_proba(Xval)[:, 1]
    p_xgb = m_xgb.predict_proba(Xval)[:, 1]
    oof_final[val_idx] = (p_lgb + p_xgb) / 2
    
    test_final += (m_lgb.predict_proba(X_test)[:, 1] + m_xgb.predict_proba(X_test)[:, 1]) / 10
    print(f"  Fold AUC: {roc_auc_score(yval, oof_final[val_idx]):.5f}")

print(f"\nFINAL OOF AUC: {roc_auc_score(y_train, oof_final):.6f}")

np.save(MODELS / "oof_preds_v11.npy", oof_final)
np.save(MODELS / "test_preds_v11.npy", test_final)
print("✓ Saved V11 artifacts.")
