"""
Step 6v2 — Improved Model Training
- LightGBM + XGBoost + CatBoost
- Rank-aware blending with weight optimization
- 5-fold stratified CV
- Class imbalance handling
- Threshold optimization for F1
"""
import polars as pl
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from scipy.optimize import minimize
from scipy.stats import rankdata
from pathlib import Path
import json

ROOT = Path(r"C:\Users\ujjaw\Downloads\AML_Mule_Project")
FEAT = ROOT / "features"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────
train = pl.read_parquet(FEAT / "train_features_v2.parquet")
test = pl.read_parquet(FEAT / "test_features_v2.parquet")

EXCLUDE = {"account_id", "is_mule"}
FEATURES = [c for c in train.columns if c not in EXCLUDE]

# Verify all features are numeric
non_numeric = []
for c in FEATURES:
    dt = train[c].dtype
    if dt not in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                  pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                  pl.Float32, pl.Float64):
        non_numeric.append(c)

for c in non_numeric:
    print(f"WARNING: dropping non-numeric column '{c}' (dtype={train[c].dtype})")
    FEATURES.remove(c)

print(f"Using {len(FEATURES)} features")

X_train = train.select(FEATURES).to_numpy().astype(np.float32)
y_train = train["is_mule"].to_numpy().astype(np.int32)
X_test = test.select(FEATURES).to_numpy().astype(np.float32)

# Replace inf/nan
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

n_pos = y_train.sum()
n_neg = len(y_train) - n_pos
scale_pos_weight = n_neg / max(n_pos, 1)

print(f"X_train: {X_train.shape}, y_train mean: {y_train.mean():.4f}")
print(f"X_test:  {X_test.shape}")
print(f"Pos: {n_pos}, Neg: {n_neg}, scale_pos_weight: {scale_pos_weight:.1f}")

# ═══════════════════════════════════════════════════════════════════════
# Model parameters
# ═══════════════════════════════════════════════════════════════════════

lgb_params = {
    "n_estimators": 2000,
    "learning_rate": 0.02,
    "num_leaves": 127,
    "max_depth": -1,
    "min_child_samples": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "scale_pos_weight": scale_pos_weight,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

xgb_params = {
    "n_estimators": 2000,
    "learning_rate": 0.02,
    "max_depth": 8,
    "min_child_weight": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "scale_pos_weight": scale_pos_weight,
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
    "eval_metric": "logloss",
    "verbosity": 0,
}

cat_params = {
    "iterations": 2500,
    "learning_rate": 0.02,
    "depth": 8,
    "l2_leaf_reg": 6.0,
    "subsample": 0.8,
    "colsample_bylevel": 0.7,
    "scale_pos_weight": scale_pos_weight,
    "eval_metric": "AUC",
    "loss_function": "Logloss",
    "early_stopping_rounds": 200,
    "random_seed": 42,
    "verbose": 0,
    "task_type": "CPU",
    "bootstrap_type": "Bernoulli",
}

# ═══════════════════════════════════════════════════════════════════════
# 5-Fold Stratified CV
# ═══════════════════════════════════════════════════════════════════════
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(y_train))
oof_xgb = np.zeros(len(y_train))
oof_cat = np.zeros(len(y_train))

test_lgb = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))
test_cat = np.zeros(len(X_test))

print("\n── Cross-Validation ──")
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    Xtr, Xval = X_train[train_idx], X_train[val_idx]
    ytr, yval = y_train[train_idx], y_train[val_idx]

    # LightGBM
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(
        Xtr, ytr,
        eval_set=[(Xval, yval)],
        callbacks=[lgb.log_evaluation(200), lgb.early_stopping(100)],
    )
    val_lgb = lgb_model.predict_proba(Xval)[:, 1]
    oof_lgb[val_idx] = val_lgb
    test_lgb += lgb_model.predict_proba(X_test)[:, 1] / 5

    # XGBoost
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(
        Xtr, ytr,
        eval_set=[(Xval, yval)],
        verbose=False,
    )
    val_xgb = xgb_model.predict_proba(Xval)[:, 1]
    oof_xgb[val_idx] = val_xgb
    test_xgb += xgb_model.predict_proba(X_test)[:, 1] / 5

    # CatBoost
    cat_model = CatBoostClassifier(**cat_params)
    cat_model.fit(
        Xtr, ytr,
        eval_set=(Xval, yval),
        verbose=0,
    )
    val_cat = cat_model.predict_proba(Xval)[:, 1]
    oof_cat[val_idx] = val_cat
    test_cat += cat_model.predict_proba(X_test)[:, 1] / 5

    val_blend = (val_lgb + val_xgb + val_cat) / 3
    print(
        f"  Fold {fold}: LGB={roc_auc_score(yval, val_lgb):.5f}  "
        f"XGB={roc_auc_score(yval, val_xgb):.5f}  "
        f"CAT={roc_auc_score(yval, val_cat):.5f}  "
        f"Blend={roc_auc_score(yval, val_blend):.5f}"
    )

# ═══════════════════════════════════════════════════════════════════════
# Blending (prob and rank)
# ═══════════════════════════════════════════════════════════════════════
print(f"\n  OOF LGB AUC:   {roc_auc_score(y_train, oof_lgb):.5f}")
print(f"  OOF XGB AUC:   {roc_auc_score(y_train, oof_xgb):.5f}")
print(f"  OOF CAT AUC:   {roc_auc_score(y_train, oof_cat):.5f}")

def neg_auc(weights):
    w = np.abs(weights)
    w = np.clip(w, 0.1, None)
    w = w / w.sum()
    blend = w[0]*oof_lgb + w[1]*oof_xgb + w[2]*oof_cat
    return -roc_auc_score(y_train, blend)

res = minimize(neg_auc, x0=[1/3, 1/3, 1/3], method="Nelder-Mead", options={"maxiter":1000})
w = np.abs(res.x)
w = np.clip(w, 0.1, None)
w = w / w.sum()
print(f"\n  Optimized weights: LGB={w[0]:.3f} XGB={w[1]:.3f} CAT={w[2]:.3f}")

# Strategies

oof_simple = (oof_lgb + oof_xgb + oof_cat) / 3
test_simple = (test_lgb + test_xgb + test_cat) / 3

oof_weighted = w[0]*oof_lgb + w[1]*oof_xgb + w[2]*oof_cat
test_weighted = w[0]*test_lgb + w[1]*test_xgb + w[2]*test_cat

def rank_avg(*arrs):
    ranks = [rankdata(a) / len(a) for a in arrs]
    return np.mean(ranks, axis=0)

oof_rank = rank_avg(oof_lgb, oof_xgb, oof_cat)
test_rank = rank_avg(test_lgb, test_xgb, test_cat)

def rank_weighted(weights, *arrs):
    ranks = [rankdata(a) / len(a) for a in arrs]
    return sum(wi * ri for wi, ri in zip(weights, ranks))

oof_rank_w = rank_weighted(w, oof_lgb, oof_xgb, oof_cat)
test_rank_w = rank_weighted(w, test_lgb, test_xgb, test_cat)

strategies = {
    "simple": (oof_simple, test_simple),
    "weighted": (oof_weighted, test_weighted),
    "rank": (oof_rank, test_rank),
    "rank_weighted": (oof_rank_w, test_rank_w),
}

best_name, best_auc = None, -1
for name, (oof_pred, _) in strategies.items():
    auc = roc_auc_score(y_train, oof_pred)
    marker = ""
    if auc > best_auc:
        best_auc = auc
        best_name = name
        marker = " ← BEST"
    print(f"  {name:14s}: OOF AUC={auc:.5f}{marker}")

oof_blend, test_blend = strategies[best_name]
print(f"\n  Selected strategy: {best_name} (AUC={best_auc:.5f})")

# ── Optimal F1 threshold ────────────────────────────────────────────
best_f1, best_thr = 0, 0
for thr in np.linspace(0, 1, 1001):
    f1 = f1_score(y_train, (oof_blend > thr).astype(int))
    if f1 > best_f1:
        best_f1, best_thr = f1, thr

print(f"\n  Best OOF F1: {best_f1:.5f} at threshold {best_thr:.2f}")

# ═══════════════════════════════════════════════════════════════════════
# Feature importance (from last LGB fold)
# ═══════════════════════════════════════════════════════════════════════
importance = dict(zip(FEATURES, lgb_model.feature_importances_.tolist()))
importance = dict(sorted(importance.items(), key=lambda x: -x[1]))
print("\nTop 25 features:")
for i, (feat, imp) in enumerate(list(importance.items())[:25]):
    print(f"  {i+1:2d}. {feat:40s} {imp}")

with open(MODELS / "feature_importance_v2.json", "w") as f:
    json.dump(importance, f, indent=2)

# ── Save predictions ─────────────────────────────────────────────────
np.save(MODELS / "test_preds_v2.npy", test_blend)
np.save(MODELS / "oof_preds_v2.npy", oof_blend)
np.save(MODELS / "test_lgb.npy", test_lgb)
np.save(MODELS / "test_xgb.npy", test_xgb)
np.save(MODELS / "test_cat.npy", test_cat)

# Save test account IDs in feature-matrix order (critical for alignment)
test["account_id"].to_frame().write_parquet(MODELS / "test_ids_order.parquet")

# Save optimal threshold
with open(MODELS / "best_threshold.txt", "w") as f:
    f.write(f"{best_thr:.4f}")

print(f"\n✓ models/test_preds_v2.npy saved")
print(f"✓ models/oof_preds_v2.npy saved")
print(f"✓ Best threshold: {best_thr:.4f}")
