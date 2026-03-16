"""
Step 6v3 — Improved Model Training

Key changes vs v2:
  • Uses train/test_features_v3.parquet (v2 + recency features)
  • LGB:  3000 trees, lr=0.01, num_leaves=255, stronger regularisation
  • XGB:  3000 trees, lr=0.01, max_leaves=255 (lossguide), stronger reg
  • CAT:  2000 iter, lr=0.02, depth=7, conservative l2=5.0
  • Rank-based blending (neutralises probability scale mismatch)
  • Auto-selects best blend strategy by OOF AUC (same as v2)
  • SAFETY GATE: only saves v3 artefacts if OOF AUC > v2 baseline (0.9511)

Outputs (if gate passes):
  models/test_preds_v3.npy      ← final test predictions
  models/oof_preds_v3.npy
  models/test_lgb_v3.npy
  models/test_xgb_v3.npy
  models/test_cat_v3.npy
  models/best_threshold_v3.txt
  models/feature_importance_v3.json
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

ROOT   = Path(r"C:\Users\ujjaw\Downloads\AML_Mule_Project")
FEAT   = ROOT / "features"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

V2_OOF_BASELINE = 0.9511   # beat this to save v3 artefacts

# ── Prefer v3 features; fall back to v2 ─────────────────────────────
v3_train_path = FEAT / "train_features_v3.parquet"
v3_test_path  = FEAT / "test_features_v3.parquet"
if v3_train_path.exists() and v3_test_path.exists():
    print("Loading V3 feature matrix ...")
    train = pl.read_parquet(v3_train_path)
    test  = pl.read_parquet(v3_test_path)
    version_tag = "v3"
else:
    print("WARNING: V3 features not found – falling back to V2.")
    train = pl.read_parquet(FEAT / "train_features_v2.parquet")
    test  = pl.read_parquet(FEAT / "test_features_v2.parquet")
    version_tag = "v2-fallback"

EXCLUDE  = {"account_id", "is_mule"}
FEATURES = [c for c in train.columns if c not in EXCLUDE]

# Drop any non-numeric columns
non_numeric = [
    c for c in FEATURES
    if train[c].dtype not in (
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64
    )
]
for c in non_numeric:
    print(f"WARNING: dropping non-numeric column '{c}' (dtype={train[c].dtype})")
    FEATURES.remove(c)

print(f"Using {len(FEATURES)} features  [{version_tag}]")

X_train = train.select(FEATURES).to_numpy().astype(np.float32)
y_train = train["is_mule"].to_numpy().astype(np.int32)
X_test  = test.select(FEATURES).to_numpy().astype(np.float32)

X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_test  = np.nan_to_num(X_test,  nan=0.0, posinf=0.0, neginf=0.0)

n_pos = y_train.sum()
n_neg = len(y_train) - n_pos
scale_pos_weight = n_neg / max(n_pos, 1)

print(f"X_train: {X_train.shape},  mule rate: {y_train.mean():.4f}")
print(f"X_test:  {X_test.shape}")
print(f"scale_pos_weight: {scale_pos_weight:.1f}")

# ═══════════════════════════════════════════════════════════════════════
# Improved hyperparameters
# ═══════════════════════════════════════════════════════════════════════

lgb_params = {
    "n_estimators":      3000,
    "learning_rate":     0.01,    # slower LR → more thorough search
    "num_leaves":        255,     # more capacity (vs 127 in v2)
    "max_depth":         -1,
    "min_child_samples": 40,      # slightly more conservative
    "subsample":         0.8,
    "colsample_bytree":  0.7,
    "reg_alpha":         1.0,     # stronger L1 (vs 0.5)
    "reg_lambda":        3.0,     # stronger L2 (vs 2.0)
    "scale_pos_weight":  scale_pos_weight,
    "random_state":      42,
    "n_jobs":            -1,
    "verbose":           -1,
}

xgb_params = {
    "n_estimators":     3000,
    "learning_rate":    0.01,
    "max_depth":        0,          # unlimited when using max_leaves
    "max_leaves":       255,
    "grow_policy":      "lossguide",
    "min_child_weight": 40,
    "subsample":        0.8,
    "colsample_bytree": 0.7,
    "reg_alpha":        1.0,
    "reg_lambda":       3.0,
    "scale_pos_weight": scale_pos_weight,
    "random_state":     42,
    "n_jobs":           -1,
    "tree_method":      "hist",
    "eval_metric":      "logloss",
    "verbosity":        0,
}

cat_params = {
    "iterations":             2000,
    "learning_rate":          0.02,
    "depth":                  7,     # conservative (vs 8 which overfit before)
    "l2_leaf_reg":            5.0,
    "subsample":              0.8,
    "colsample_bylevel":      0.7,
    "scale_pos_weight":       scale_pos_weight,
    "eval_metric":            "AUC",
    "loss_function":          "Logloss",
    "early_stopping_rounds":  150,
    "random_seed":            42,
    "verbose":                0,
    "task_type":              "CPU",
    "bootstrap_type":         "Bernoulli",
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

print("\n── 5-Fold Cross-Validation ──")
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    Xtr, Xval = X_train[train_idx], X_train[val_idx]
    ytr, yval = y_train[train_idx], y_train[val_idx]

    # ── LightGBM ────────────────────────────────────────────────────
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(
        Xtr, ytr,
        eval_set=[(Xval, yval)],
        callbacks=[lgb.log_evaluation(500), lgb.early_stopping(150)],
    )
    val_lgb = lgb_model.predict_proba(Xval)[:, 1]
    oof_lgb[val_idx] = val_lgb
    test_lgb += lgb_model.predict_proba(X_test)[:, 1] / 5

    # ── XGBoost ─────────────────────────────────────────────────────
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(
        Xtr, ytr,
        eval_set=[(Xval, yval)],
        verbose=False,
    )
    val_xgb = xgb_model.predict_proba(Xval)[:, 1]
    oof_xgb[val_idx] = val_xgb
    test_xgb += xgb_model.predict_proba(X_test)[:, 1] / 5

    # ── CatBoost ────────────────────────────────────────────────────
    cat_model = CatBoostClassifier(**cat_params)
    cat_model.fit(
        Xtr, ytr,
        eval_set=(Xval, yval),
        verbose=0,
    )
    val_cat = cat_model.predict_proba(Xval)[:, 1]
    oof_cat[val_idx] = val_cat
    test_cat += cat_model.predict_proba(X_test)[:, 1] / 5

    auc_lgb   = roc_auc_score(yval, val_lgb)
    auc_xgb   = roc_auc_score(yval, val_xgb)
    auc_cat   = roc_auc_score(yval, val_cat)
    val_rank  = (rankdata(val_lgb) + rankdata(val_xgb)) / (2 * len(val_lgb))
    auc_blend = roc_auc_score(yval, val_rank)
    print(
        f"  Fold {fold}: LGB={auc_lgb:.5f}  XGB={auc_xgb:.5f}  "
        f"CAT={auc_cat:.5f}  Rank-LX={auc_blend:.5f}"
    )

# ═══════════════════════════════════════════════════════════════════════
# Blend strategy selection (rank-based to avoid probability scale issues)
# ═══════════════════════════════════════════════════════════════════════
print("\n── OOF Results ──")
auc_lgb_oof = roc_auc_score(y_train, oof_lgb)
auc_xgb_oof = roc_auc_score(y_train, oof_xgb)
auc_cat_oof = roc_auc_score(y_train, oof_cat)
print(f"  LGB OOF AUC: {auc_lgb_oof:.5f}")
print(f"  XGB OOF AUC: {auc_xgb_oof:.5f}")
print(f"  CAT OOF AUC: {auc_cat_oof:.5f}")

def rank_avg(*arrs):
    """Normalised rank average – scale-invariant blend."""
    ranks = [rankdata(a) / len(a) for a in arrs]
    return np.mean(ranks, axis=0)

def rank_weighted_blend(weights, *arrs):
    ranks = [rankdata(a) / len(a) for a in arrs]
    return sum(w * r for w, r in zip(weights, ranks))

# Optimise rank-blend weights on OOF
def neg_auc_rank(logits):
    w = np.exp(logits)           # always positive
    w = w / w.sum()
    blend = rank_weighted_blend(w, oof_lgb, oof_xgb, oof_cat)
    return -roc_auc_score(y_train, blend)

res = minimize(neg_auc_rank, x0=[0.0, 0.0, -1.6],  # start: 40/40/20
               method="Nelder-Mead", options={"maxiter": 2000})
w_opt = np.exp(res.x); w_opt = w_opt / w_opt.sum()
# Cap CAT at 20% to prevent overfitting by a single strong-but-overfit model
w_opt[2] = min(w_opt[2], 0.20)
w_opt = w_opt / w_opt.sum()
print(f"\n  Optimised rank weights: LGB={w_opt[0]:.3f} XGB={w_opt[1]:.3f} CAT={w_opt[2]:.3f}")

# Candidate strategies
strategies = {
    "rank_lgb_xgb":    (
        rank_avg(oof_lgb, oof_xgb),
        rank_avg(test_lgb, test_xgb),
    ),
    "rank_all_equal":  (
        rank_avg(oof_lgb, oof_xgb, oof_cat),
        rank_avg(test_lgb, test_xgb, test_cat),
    ),
    "rank_weighted":   (
        rank_weighted_blend(w_opt, oof_lgb, oof_xgb, oof_cat),
        rank_weighted_blend(w_opt, test_lgb, test_xgb, test_cat),
    ),
    "prob_lgb_xgb":    (
        (oof_lgb + oof_xgb) / 2,
        (test_lgb + test_xgb) / 2,
    ),
}

best_name, best_auc = None, -1
print("\n── Strategy Comparison ──")
for name, (oof_pred, _) in strategies.items():
    auc = roc_auc_score(y_train, oof_pred)
    marker = ""
    if auc > best_auc:
        best_auc = auc
        best_name = name
        marker = " ← BEST"
    print(f"  {name:20s}: OOF AUC={auc:.5f}{marker}")

oof_blend, test_blend = strategies[best_name]
print(f"\n  Selected: {best_name}  (OOF AUC={best_auc:.5f})")

# ── Safety gate ──────────────────────────────────────────────────────
print(f"\n── Safety Gate ──")
print(f"  V3 OOF AUC:       {best_auc:.5f}")
print(f"  V2 OOF baseline:  {V2_OOF_BASELINE:.5f}")

if best_auc <= V2_OOF_BASELINE:
    print(
        f"\n  ✗ V3 did NOT beat v2 baseline "
        f"(Δ={best_auc - V2_OOF_BASELINE:+.5f}). "
        "V3 artefacts NOT saved. Use v2 predictions."
    )
    raise SystemExit(0)

print(f"\n  ✓ V3 beats v2 by Δ={best_auc - V2_OOF_BASELINE:+.5f}. Saving artefacts.")

# ── Optimal F1 threshold ─────────────────────────────────────────────
best_f1, best_thr = 0.0, 0.5
for thr in np.linspace(0.1, 0.9, 801):
    f1 = f1_score(y_train, (oof_blend > thr).astype(int), zero_division=0)
    if f1 > best_f1:
        best_f1, best_thr = f1, thr

print(f"  Best OOF F1: {best_f1:.5f} at threshold {best_thr:.4f}")

# ── Feature importance (last LGB fold) ──────────────────────────────
importance = dict(zip(FEATURES, lgb_model.feature_importances_.tolist()))
importance = dict(sorted(importance.items(), key=lambda x: -x[1]))
print("\nTop 20 features:")
for i, (feat, imp) in enumerate(list(importance.items())[:20]):
    print(f"  {i+1:2d}. {feat:45s} {imp}")

with open(MODELS / "feature_importance_v3.json", "w") as f:
    json.dump(importance, f, indent=2)

# ── Save predictions ─────────────────────────────────────────────────
np.save(MODELS / "test_preds_v3.npy",  test_blend)
np.save(MODELS / "oof_preds_v3.npy",   oof_blend)
np.save(MODELS / "test_lgb_v3.npy",    test_lgb)
np.save(MODELS / "test_xgb_v3.npy",    test_xgb)
np.save(MODELS / "test_cat_v3.npy",    test_cat)

with open(MODELS / "best_threshold_v3.txt", "w") as f:
    f.write(f"{best_thr:.4f}")

# test_ids_order is the same (same account set)
print(f"\n✓ models/test_preds_v3.npy saved")
print(f"✓ models/oof_preds_v3.npy saved")
print(f"✓ best_threshold_v3.txt = {best_thr:.4f}")
print(f"✓ feature_importance_v3.json saved")
