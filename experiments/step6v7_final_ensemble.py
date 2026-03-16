"""
Step 6v7 — Final Triple Ensemble (LGB + XGB + CAT)
==================================================
Trains a 5-fold Stratified CV ensemble of LightGBM, XGBoost, and CatBoost.
Automatically uses all numeric features from v2 and SHAP tables.

Reads:  features/train_features_v2.parquet, features/test_features_v2.parquet
        features/shap_recommended_features.parquet
Writes: models/oof_preds_v7.npy, models/test_preds_v7.npy
"""
import polars as pl
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from scipy.stats import rankdata
from pathlib import Path
import json, time, gc

ROOT   = Path(r"C:\Users\ujjaw\Downloads\AML_Mule_Project")
FEAT   = ROOT / "features"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  Step 6v7 — Final Triple Ensemble (LGB + XGB + CAT)")
print("=" * 70)

train_v2 = pl.read_parquet(FEAT / "train_features_v2.parquet")
test_v2  = pl.read_parquet(FEAT / "test_features_v2.parquet")
shap_ft  = pl.read_parquet(FEAT / "shap_recommended_features.parquet")

# Merge
train = train_v2.join(shap_ft, on="account_id", how="left")
test  = test_v2.join(shap_ft, on="account_id", how="left")

# Automatically identify numeric features
EXCLUDE = {"account_id", "is_mule", "mule_flag_date", "alert_reason", "flagged_by_branch"}
ALL_COLS = train.columns
FEATURES = []
for c in ALL_COLS:
    if c in EXCLUDE: continue
    dt = train[c].dtype
    if dt in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
              pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
              pl.Float32, pl.Float64):
        FEATURES.append(c)

print(f"  Total Features: {len(FEATURES)}")

X_train = train.select(FEATURES).to_numpy().astype(np.float32)
y_train = train["is_mule"].to_numpy().astype(np.int32)
X_test  = test.select(FEATURES).to_numpy().astype(np.float32)

X_train = np.nan_to_num(X_train, nan=0.0)
X_test  = np.nan_to_num(X_test,  nan=0.0)

n_pos = y_train.sum()
n_neg = len(y_train) - n_pos
scale_pos_weight = n_neg / max(n_pos, 1)

print(f"  Train:    {X_train.shape}")
print(f"  Pos/Neg:  {n_pos}/{n_neg} (Scale: {scale_pos_weight:.1f})")

# 2. PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
# Refined based on previous Turn performance
lgb_params = {
    "n_estimators": 2000, "learning_rate": 0.015, "num_leaves": 127,
    "max_depth": -1, "min_child_samples": 30, "subsample": 0.8,
    "colsample_bytree": 0.7, "reg_alpha": 0.5, "reg_lambda": 2.0,
    "scale_pos_weight": scale_pos_weight, "random_state": 42,
    "n_jobs": -1, "verbose": -1,
}

xgb_params = {
    "n_estimators": 2000, "learning_rate": 0.015, "max_depth": 8,
    "min_child_weight": 30, "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.5, "reg_lambda": 2.0, "scale_pos_weight": scale_pos_weight,
    "random_state": 42, "n_jobs": -1, "tree_method": "hist",
    "eval_metric": "auc", "early_stopping_rounds": 100, "verbosity": 0,
}

cat_params = {
    "iterations": 2000, "learning_rate": 0.015, "depth": 8,
    "l2_leaf_reg": 3.0, "scale_pos_weight": scale_pos_weight,
    "random_seed": 42, "loss_function": "Logloss", "eval_metric": "AUC",
    "early_stopping_rounds": 100, "verbose": 500, "task_type": "CPU",
}

# 3. 5-FOLD CV
# ─────────────────────────────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(y_train))
oof_xgb = np.zeros(len(y_train))
oof_cat = np.zeros(len(y_train))

test_lgb = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))
test_cat = np.zeros(len(X_test))

start_time = time.time()

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    Xtr, Xval = X_train[train_idx], X_train[val_idx]
    ytr, yval = y_train[train_idx], y_train[val_idx]

    print(f"\nFOLD {fold} {'─'*45}")

    # LightGBM
    print("  Training LightGBM...")
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(Xtr, ytr, eval_set=[(Xval, yval)], 
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)])
    oof_lgb[val_idx] = model_lgb.predict_proba(Xval)[:, 1]
    test_lgb += model_lgb.predict_proba(X_test)[:, 1] / 5

    # XGBoost
    print("  Training XGBoost...")
    model_xgb = xgb.XGBClassifier(**xgb_params)
    model_xgb.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=500)
    oof_xgb[val_idx] = model_xgb.predict_proba(Xval)[:, 1]
    test_xgb += model_xgb.predict_proba(X_test)[:, 1] / 5

    # CatBoost
    print("  Training CatBoost...")
    model_cat = CatBoostClassifier(**cat_params)
    model_cat.fit(Xtr, ytr, eval_set=(Xval, yval), use_best_model=True)
    oof_cat[val_idx] = model_cat.predict_proba(Xval)[:, 1]
    test_cat += model_cat.predict_proba(X_test)[:, 1] / 5

    fold_mean = (oof_lgb[val_idx] + oof_xgb[val_idx] + oof_cat[val_idx]) / 3
    fold_auc = roc_auc_score(yval, fold_mean)
    print(f"  FOLD {fold} AUC: {fold_auc:.5f}")

total_time = time.time() - start_time
print(f"\nDone. Total time: {total_time/60:.1f} min")

# 4. BLENDING (Rank-Weighted)
# ─────────────────────────────────────────────────────────────────────────────
def rank_avg(*arrs):
    ranks = [rankdata(a) / len(a) for a in arrs]
    return np.mean(ranks, axis=0)

oof_final  = rank_avg(oof_lgb, oof_xgb, oof_cat)
test_final = rank_avg(test_lgb, test_xgb, test_cat)

total_auc = roc_auc_score(y_train, oof_final)
print(f"\nFINAL OOF AUC (Rank Blend): {total_auc:.6f}")

# Save
np.save(MODELS / "oof_preds_v7.npy", oof_final)
np.save(MODELS / "test_preds_v7.npy", test_final)

print(f"\n✓ Saved artifacts to {MODELS}")
