"""
Step 6v4 — SHAP-Pruned LightGBM + XGBoost Training
====================================================
Drops features with mean |SHAP| < 0.01, then retrains LGB + XGB
with 5-fold StratifiedCV + optimized blending.

Reads:  features/train_features_v2.parquet, features/test_features_v2.parquet
        report/shap/shap_feature_ranking.json
Writes: models/oof_preds_v4.npy, models/test_preds_v4.npy, submission.csv, etc.
"""
import polars as pl
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from scipy.optimize import minimize
from scipy.stats import rankdata
from pathlib import Path
from datetime import timedelta
import pandas as pd
import json, time

ROOT   = Path(r"C:\Users\ujjaw\Downloads\AML_Mule_Project")
DATA   = ROOT / "data" / "archive"
FEAT   = ROOT / "features"
MODELS = ROOT / "models"
SHAP   = ROOT / "report" / "shap"
MODELS.mkdir(parents=True, exist_ok=True)

SHAP_THRESHOLD = 0.01   # drop features with mean |SHAP| below this

# ═══════════════════════════════════════════════════════════════════════
# 1. LOAD DATA + SHAP-BASED FEATURE SELECTION
# ═══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  Step 6v4 — SHAP-Pruned Model Training")
print("=" * 70)

train = pl.read_parquet(FEAT / "train_features_v2.parquet")
test  = pl.read_parquet(FEAT / "test_features_v2.parquet")

# Load SHAP ranking
with open(SHAP / "shap_feature_ranking.json") as f:
    shap_ranking = json.load(f)

EXCLUDE = {"account_id", "is_mule"}
ALL_FEATURES = [c for c in train.columns if c not in EXCLUDE]

# Filter non-numeric
non_numeric = []
for c in ALL_FEATURES:
    dt = train[c].dtype
    if dt not in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                  pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                  pl.Float32, pl.Float64):
        non_numeric.append(c)
for c in non_numeric:
    ALL_FEATURES.remove(c)

# Apply SHAP threshold
kept_features = []
dropped_features = []
for feat in ALL_FEATURES:
    shap_val = shap_ranking.get(feat, {}).get("mean_abs_shap", 0.0)
    if shap_val >= SHAP_THRESHOLD:
        kept_features.append(feat)
    else:
        dropped_features.append(feat)

FEATURES = kept_features

print(f"\n  Total numeric features: {len(ALL_FEATURES)}")
print(f"  SHAP threshold: {SHAP_THRESHOLD}")
print(f"  ✓ Kept:    {len(FEATURES)} features (mean |SHAP| ≥ {SHAP_THRESHOLD})")
print(f"  ✗ Dropped: {len(dropped_features)} features")
print(f"\n  Dropped features:")
for feat in dropped_features:
    shap_val = shap_ranking.get(feat, {}).get("mean_abs_shap", 0.0)
    print(f"    {feat:<40s} SHAP={shap_val:.6f}")

X_train = train.select(FEATURES).to_numpy().astype(np.float32)
y_train = train["is_mule"].to_numpy().astype(np.int32)
X_test  = test.select(FEATURES).to_numpy().astype(np.float32)

X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_test  = np.nan_to_num(X_test,  nan=0.0, posinf=0.0, neginf=0.0)

n_pos = y_train.sum()
n_neg = len(y_train) - n_pos
scale_pos_weight = n_neg / max(n_pos, 1)

print(f"\n  X_train: {X_train.shape}, y_train mean: {y_train.mean():.4f}")
print(f"  X_test:  {X_test.shape}")
print(f"  Pos: {n_pos}, Neg: {n_neg}, scale_pos_weight: {scale_pos_weight:.1f}")

# ═══════════════════════════════════════════════════════════════════════
# 2. MODEL PARAMETERS (same as v3)
# ═══════════════════════════════════════════════════════════════════════
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
    "eval_metric": "logloss", "early_stopping_rounds": 100, "verbosity": 0,
}

# ═══════════════════════════════════════════════════════════════════════
# 3. 5-FOLD STRATIFIED CV
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Training LightGBM + XGBoost (5-fold CV, pruned features)")
print("=" * 70)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(y_train))
oof_xgb = np.zeros(len(y_train))
test_lgb = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))

total_start = time.time()

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    fold_start = time.time()
    Xtr, Xval = X_train[train_idx], X_train[val_idx]
    ytr, yval = y_train[train_idx], y_train[val_idx]

    print(f"\n{'─'*60}")
    print(f"FOLD {fold} | train={len(train_idx)} val={len(val_idx)} "
          f"(mule rate: {yval.mean():.4f})")
    print(f"{'─'*60}")

    # ── LightGBM ─────────────────────────────────────────────────
    print(f"\n  [LGB] Training...")
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(
        Xtr, ytr,
        eval_set=[(Xtr, ytr), (Xval, yval)],
        eval_names=["train", "valid"],
        eval_metric="auc",
        callbacks=[lgb.log_evaluation(100), lgb.early_stopping(100)],
    )
    val_lgb = lgb_model.predict_proba(Xval)[:, 1]
    oof_lgb[val_idx] = val_lgb
    test_lgb += lgb_model.predict_proba(X_test)[:, 1] / 5

    lgb_auc = roc_auc_score(yval, val_lgb)
    lgb_f1 = f1_score(yval, (val_lgb > 0.5).astype(int))
    print(f"  [LGB] Fold {fold}: AUC={lgb_auc:.5f}  F1={lgb_f1:.5f}  "
          f"best_iter={lgb_model.best_iteration_}")

    # ── XGBoost ──────────────────────────────────────────────────
    print(f"\n  [XGB] Training...")
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xval, yval)], verbose=100)
    val_xgb = xgb_model.predict_proba(Xval)[:, 1]
    oof_xgb[val_idx] = val_xgb
    test_xgb += xgb_model.predict_proba(X_test)[:, 1] / 5

    xgb_auc = roc_auc_score(yval, val_xgb)
    xgb_f1 = f1_score(yval, (val_xgb > 0.5).astype(int))
    xgb_best = getattr(xgb_model, 'best_iteration', xgb_params['n_estimators'])
    print(f"  [XGB] Fold {fold}: AUC={xgb_auc:.5f}  F1={xgb_f1:.5f}  "
          f"best_iter={xgb_best}")

    # ── Fold blend ───────────────────────────────────────────────
    val_blend = (val_lgb + val_xgb) / 2
    blend_auc = roc_auc_score(yval, val_blend)
    print(f"\n  ★ Fold {fold} Blend: AUC={blend_auc:.5f}")

    fold_time = time.time() - fold_start
    print(f"  ⏱ Fold time: {fold_time:.0f}s")

total_time = time.time() - total_start
print(f"\n⏱ Total training time: {total_time:.0f}s ({total_time/60:.1f}min)")

# ═══════════════════════════════════════════════════════════════════════
# 4. BLENDING
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Blending & Evaluation")
print("=" * 70)

print(f"\n  OOF LGB AUC:  {roc_auc_score(y_train, oof_lgb):.5f}")
print(f"  OOF XGB AUC:  {roc_auc_score(y_train, oof_xgb):.5f}")

# Optimize blend weight
def neg_auc(weights):
    w = np.abs(weights)
    w = np.clip(w, 0.1, None)
    w = w / w.sum()
    blend = w[0] * oof_lgb + w[1] * oof_xgb
    return -roc_auc_score(y_train, blend)

res = minimize(neg_auc, x0=[0.5, 0.5], method="Nelder-Mead",
               options={"maxiter": 1000})
w = np.abs(res.x)
w = np.clip(w, 0.1, None)
w = w / w.sum()
print(f"\n  Optimized weights: LGB={w[0]:.3f}  XGB={w[1]:.3f}")

# All blending strategies
oof_simple  = (oof_lgb + oof_xgb) / 2
test_simple = (test_lgb + test_xgb) / 2

oof_weighted  = w[0] * oof_lgb + w[1] * oof_xgb
test_weighted = w[0] * test_lgb + w[1] * test_xgb

def rank_avg(*arrs):
    ranks = [rankdata(a) / len(a) for a in arrs]
    return np.mean(ranks, axis=0)

oof_rank  = rank_avg(oof_lgb, oof_xgb)
test_rank = rank_avg(test_lgb, test_xgb)

def rank_weighted(weights, *arrs):
    ranks = [rankdata(a) / len(a) for a in arrs]
    return sum(wi * ri for wi, ri in zip(weights, ranks))

oof_rank_w  = rank_weighted(w, oof_lgb, oof_xgb)
test_rank_w = rank_weighted(w, test_lgb, test_xgb)

strategies = {
    "simple_avg":    (oof_simple, test_simple),
    "weighted":      (oof_weighted, test_weighted),
    "rank_avg":      (oof_rank, test_rank),
    "rank_weighted": (oof_rank_w, test_rank_w),
}

print(f"\n  Blending strategies:")
best_name, best_auc = None, -1
for name, (oof_pred, _) in strategies.items():
    auc = roc_auc_score(y_train, oof_pred)
    marker = ""
    if auc > best_auc:
        best_auc = auc
        best_name = name
        marker = "  ← BEST"
    print(f"    {name:16s}: OOF AUC = {auc:.5f}{marker}")

oof_blend, test_blend = strategies[best_name]
print(f"\n  ★ Selected: {best_name} (AUC={best_auc:.5f})")

# Compare with v3
try:
    v3_oof = np.load(MODELS / "oof_preds_v3.npy")
    v3_auc = roc_auc_score(y_train, v3_oof)
    print(f"  ★ v3 reference OOF AUC: {v3_auc:.5f}")
    diff = best_auc - v3_auc
    symbol = "BETTER" if diff > 0 else "WORSE" if diff < 0 else "SAME"
    print(f"    Δ = {diff:+.5f} ({symbol})")
except Exception:
    pass

# ── F1 threshold optimization ────────────────────────────────────────
best_f1, best_thr = 0, 0
for thr in np.linspace(0, 1, 1001):
    f1 = f1_score(y_train, (oof_blend > thr).astype(int))
    if f1 > best_f1:
        best_f1, best_thr = f1, thr

print(f"\n  Best OOF F1: {best_f1:.5f} at threshold {best_thr:.4f}")

# ═══════════════════════════════════════════════════════════════════════
# 5. SAVE ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Saving artifacts...")
print("=" * 70)

np.save(MODELS / "oof_lgb_v4.npy", oof_lgb)
np.save(MODELS / "oof_xgb_v4.npy", oof_xgb)
np.save(MODELS / "oof_preds_v4.npy", oof_blend)
np.save(MODELS / "test_preds_v4.npy", test_blend)
np.save(MODELS / "test_lgb_v4.npy", test_lgb)
np.save(MODELS / "test_xgb_v4.npy", test_xgb)

test.select("account_id").write_parquet(MODELS / "test_ids_order_v4.parquet")

with open(MODELS / "best_threshold_v4.txt", "w") as f:
    f.write(f"{best_thr:.4f}")

# Feature importance
importance = dict(zip(FEATURES, lgb_model.feature_importances_.tolist()))
importance = dict(sorted(importance.items(), key=lambda x: -x[1]))
with open(MODELS / "feature_importance_v4.json", "w") as f:
    json.dump(importance, f, indent=2)

# Save the pruned feature list
with open(MODELS / "pruned_features_v4.json", "w") as f:
    json.dump({
        "shap_threshold": SHAP_THRESHOLD,
        "n_kept": len(FEATURES),
        "n_dropped": len(dropped_features),
        "kept": FEATURES,
        "dropped": dropped_features,
    }, f, indent=2)

print(f"\n  Top 20 features:")
for i, (feat, imp) in enumerate(list(importance.items())[:20]):
    print(f"    {i+1:2d}. {feat:40s} {imp}")

print(f"\n✓ All v4 artifacts saved (pruned from {len(ALL_FEATURES)} → {len(FEATURES)} features)")

# ═══════════════════════════════════════════════════════════════════════
# 6. DYNAMIC TEMPORAL WINDOWS + SUBMISSION
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Generating submission with dynamic temporal windows...")
print("=" * 70)

test_ids_df = test.select("account_id").with_columns(pl.Series("pred", test_blend))

predicted_mule_ids = (
    test_ids_df
    .filter(pl.col("pred") > best_thr)["account_id"]
    .to_list()
)
print(f"  Predicted mules (threshold {best_thr:.4f}): {len(predicted_mule_ids)}")

# Scan daily volumes
daily_parts = []
if predicted_mule_ids:
    target_mule_ids = pl.Series("account_id", predicted_mule_ids)
    for batch in ["batch-1", "batch-2", "batch-3", "batch-4"]:
        print(f"  Scanning {batch}...")
        df = (
            pl.scan_parquet(DATA / "transactions" / batch / "*.parquet")
            .filter(pl.col("account_id").is_in(target_mule_ids))
            .with_columns(pl.col("transaction_timestamp").str.to_datetime().alias("ts"))
            .with_columns(pl.col("ts").dt.date().alias("d"))
            .group_by(["account_id", "d"])
            .agg([
                pl.col("amount").abs().sum().alias("daily_vol"),
                pl.len().alias("daily_txns"),
            ])
            .collect()
        )
        daily_parts.append(df)

# Peak detection
windows = {}
if daily_parts:
    daily = (
        pl.concat(daily_parts)
        .group_by(["account_id", "d"])
        .agg([pl.col("daily_vol").sum(), pl.col("daily_txns").sum()])
    )

    for acc_id in predicted_mule_ids:
        acc_df = daily.filter(pl.col("account_id") == acc_id).sort("d")
        if acc_df.is_empty():
            continue

        pdf = acc_df.to_pandas()
        dates = pd.to_datetime(pdf["d"]).tolist()
        vols = pdf["daily_vol"].values.astype(float)

        if len(dates) < 2:
            windows[acc_id] = (
                dates[0].strftime("%Y-%m-%d %H:%M:%S"),
                (dates[0] + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
            )
            continue

        date_range = pd.date_range(start=min(dates), end=max(dates), freq="D")
        vol_series = pd.Series(0.0, index=date_range)
        for dt, v in zip(dates, vols):
            vol_series[dt] = v

        rolling_avg = vol_series.rolling(window=7, min_periods=1, center=True).mean()
        peak_idx = rolling_avg.argmax()

        baseline = vol_series.median()
        if baseline <= 0:
            baseline = vol_series.mean()
        if baseline <= 0:
            baseline = 1.0

        rolling_vals = rolling_avg.values
        left = peak_idx
        while left > 0 and rolling_vals[left - 1] >= baseline:
            left -= 1
        right = peak_idx
        while right < len(rolling_vals) - 1 and rolling_vals[right + 1] >= baseline:
            right += 1

        start_date = vol_series.index[left]
        end_date   = vol_series.index[right] + timedelta(days=1)

        windows[acc_id] = (
            start_date.strftime("%Y-%m-%d %H:%M:%S"),
            end_date.strftime("%Y-%m-%d %H:%M:%S"),
        )

# Window stats
if windows:
    wdays = [(pd.Timestamp(e) - pd.Timestamp(s)).days for s, e in windows.values()]
    wdays = np.array(wdays)
    print(f"\n  Window stats: mean={wdays.mean():.0f}d, median={np.median(wdays):.0f}d, "
          f"min={wdays.min()}d, max={wdays.max()}d")

# Build submission
sub_df = test_ids_df.select(["account_id", "pred"]).rename({"pred": "is_mule"})

sus_starts, sus_ends = [], []
for row in sub_df.iter_rows(named=True):
    acc_id = row["account_id"]
    if row["is_mule"] > best_thr and acc_id in windows:
        s, e = windows[acc_id]
        sus_starts.append(s)
        sus_ends.append(e)
    else:
        sus_starts.append("")
        sus_ends.append("")

submission = pd.DataFrame({
    "account_id": sub_df["account_id"].to_list(),
    "is_mule": sub_df["is_mule"].to_list(),
    "suspicious_start": sus_starts,
    "suspicious_end": sus_ends,
})

out_path = ROOT / "submission.csv"
submission.to_csv(out_path, index=False)

n_windows = sum(1 for s in sus_starts if s)
arr = submission["is_mule"].values
print(f"\n✓ submission.csv saved ({submission.shape})")
print(f"  is_mule: mean={arr.mean():.4f}, min={arr.min():.4f}, max={arr.max():.4f}")
print(f"  Temporal windows: {n_windows}")
print(f"\nSample mules:")
print(submission[submission["is_mule"] > best_thr].head(10))
