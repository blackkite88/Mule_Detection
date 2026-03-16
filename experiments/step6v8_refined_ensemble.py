"""
Step 6v8 — Refined Triple Ensemble (V8)
========================================
Rolls back to the successful SHAP-based feature selection logic of V6.
Ensures we use the exact 108 features that achieved 0.985 AUC.

Reads:  features/train_features_v2.parquet, features/test_features_v2.parquet
        features/shap_recommended_features.parquet
        report/shap/shap_feature_ranking.json
Writes: models/oof_preds_v8.npy, models/test_preds_v8.npy
"""
import polars as pl
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from pathlib import Path
import json, time, gc

ROOT   = Path(r"C:\Users\ujjaw\Downloads\AML_Mule_Project")
FEAT   = ROOT / "features"
MODELS = ROOT / "models"
SHAP   = ROOT / "report" / "shap"
MODELS.mkdir(parents=True, exist_ok=True)

SHAP_THRESHOLD = 0.01

# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  Step 6v8 — Refined Triple Ensemble (V8)")
print("=" * 70)

train_v2 = pl.read_parquet(FEAT / "train_features_v2.parquet")
test_v2  = pl.read_parquet(FEAT / "test_features_v2.parquet")
shap_ft  = pl.read_parquet(FEAT / "shap_recommended_features.parquet")

# Merge
train = train_v2.join(shap_ft, on="account_id", how="left")
test  = test_v2.join(shap_ft, on="account_id", how="left")

# Interaction features (proven in V5/V6)
# Assuming in_degree is in train_v2
train = train.with_columns([
    (pl.col("pass_through_ratio") * pl.col("in_degree").fill_null(0)).alias("rapid_eff_x_indeg"),
    (pl.col("rolling_7d_zscore_max") * pl.col("total_txns").fill_null(0)).alias("burst_x_zscore"),
])
test = test.with_columns([
    (pl.col("pass_through_ratio") * pl.col("in_degree").fill_null(0)).alias("rapid_eff_x_indeg"),
    (pl.col("rolling_7d_zscore_max") * pl.col("total_txns").fill_null(0)).alias("burst_x_zscore"),
])

# 2. DYNAMIC SHAP SELECTION (Exact V6 Logic)
# ─────────────────────────────────────────────────────────────────────────────
with open(SHAP / "shap_feature_ranking.json") as f:
    shap_ranking = json.load(f)

EXCLUDE = {"account_id", "is_mule"}
ALL_COLS = train.columns
FEATURES = []
for c in ALL_COLS:
    if c in EXCLUDE: continue
    # Only numeric
    dt = train[c].dtype
    if dt not in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
                  pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                  pl.Float32, pl.Float64):
        continue
        
    # Check SHAP threshold
    shap_val = shap_ranking.get(c, {}).get("mean_abs_shap", 0.0)
    # The 6 new SHAP features won't be in the ranking yet, so we keep them by default
    if shap_val >= SHAP_THRESHOLD or c in shap_ft.columns or c in {"rapid_eff_x_indeg", "burst_x_zscore"}:
        FEATURES.append(c)

print(f"  ✓ Kept:    {len(FEATURES)} features")

X_train = train.select(FEATURES).to_numpy().astype(np.float32)
y_train = train["is_mule"].to_numpy().astype(np.int32)
X_test  = test.select(FEATURES).to_numpy().astype(np.float32)

X_train = np.nan_to_num(X_train, nan=0.0)
X_test  = np.nan_to_num(X_test,  nan=0.0)

n_pos = y_train.sum()
n_neg = len(y_train) - n_pos
scale_pos_weight = n_neg / max(n_pos, 1)

print(f"  Train shape: {X_train.shape}")

# 3. PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
lgb_params = {
    "n_estimators": 2000, "learning_rate": 0.02, "num_leaves": 127,
    "max_depth": -1, "min_child_samples": 30, "subsample": 0.8,
    "colsample_bytree": 0.7, "reg_alpha": 0.5, "reg_lambda": 2.0,
    "scale_pos_weight": scale_pos_weight, "random_state": 42,
    "n_jobs": -1, "verbose": -1,
}

xgb_params = {
    "n_estimators": 2000, "learning_rate": 0.02, "max_depth": 8,
    "min_child_weight": 30, "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.5, "reg_lambda": 2.0, "scale_pos_weight": scale_pos_weight,
    "random_state": 42, "n_jobs": -1, "tree_method": "hist",
    "eval_metric": "auc", "early_stopping_rounds": 100, "verbosity": 0,
}

cat_params = {
    "iterations": 2000, "learning_rate": 0.02, "depth": 8,
    "l2_leaf_reg": 3.0, "scale_pos_weight": scale_pos_weight,
    "random_seed": 42, "loss_function": "Logloss", "eval_metric": "AUC",
    "early_stopping_rounds": 100, "verbose": 500, "task_type": "CPU",
}

# 4. 5-FOLD CV
# ─────────────────────────────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(y_train))
oof_xgb = np.zeros(len(y_train))
oof_cat = np.zeros(len(y_train))

test_lgb = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))
test_cat = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    Xtr, Xval = X_train[train_idx], X_train[val_idx]
    ytr, yval = y_train[train_idx], y_train[val_idx]

    print(f"\nFOLD {fold} {'─'*45}")

    # LightGBM
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(Xtr, ytr, eval_set=[(Xval, yval)], 
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)])
    oof_lgb[val_idx] = model_lgb.predict_proba(Xval)[:, 1]
    test_lgb += model_lgb.predict_proba(X_test)[:, 1] / 5

    # XGBoost
    model_xgb = xgb.XGBClassifier(**xgb_params)
    model_xgb.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=500)
    oof_xgb[val_idx] = model_xgb.predict_proba(Xval)[:, 1]
    test_xgb += model_xgb.predict_proba(X_test)[:, 1] / 5

    # CatBoost
    model_cat = CatBoostClassifier(**cat_params)
    model_cat.fit(Xtr, ytr, eval_set=(Xval, yval), use_best_model=True)
    oof_cat[val_idx] = model_cat.predict_proba(Xval)[:, 1]
    test_cat += model_cat.predict_proba(X_test)[:, 1] / 5

    fold_auc = roc_auc_score(yval, (oof_lgb[val_idx] + oof_xgb[val_idx] + oof_cat[val_idx])/3)
    print(f"  Fold {fold} AUC: {fold_auc:.5f}")

# 5. BLENDING
# ─────────────────────────────────────────────────────────────────────────────
def rank_avg(*arrs):
    ranks = [rankdata(a) / len(a) for a in arrs]
    return np.mean(ranks, axis=0)

oof_final  = rank_avg(oof_lgb, oof_xgb, oof_cat)
test_final = rank_avg(test_lgb, test_xgb, test_cat)

print(f"\nFINAL OOF AUC: {roc_auc_score(y_train, oof_final):.6f}")

# Save
np.save(MODELS / "oof_preds_v8.npy", oof_final)
np.save(MODELS / "test_preds_v8.npy", test_final)
print(f"\n✓ Saved V8 ensemble artifacts to {MODELS}")
