"""
Step 10 — Pseudo-Label Retrain with Rank Blend + Dynamic Temporal

Full pipeline:
  1. Load train + test features
  2. Select high-confidence pseudo-labels from rank-blended v3 predictions
     - Top 0.5% → is_mule = 1
     - Bottom 10% → is_mule = 0
  3. Augment training data with pseudo-labeled test rows
  4. Retrain LightGBM + XGBoost + CatBoost (5-fold stratified CV)
  5. Save individual OOF predictions (oof_lgb_v4, oof_xgb_v4, oof_cat_v4)
  6. Rank-blend all predictions
  7. Optimise F1 threshold
  8. Apply dynamic temporal windows (7-day rolling peak detection)
  9. Generate final submission.csv

Inputs:  features/train_features_v2.parquet
         features/test_features_v2.parquet
         models/test_preds_v3_rank.npy  (from step 8)
Outputs: models/oof_lgb_v4.npy, oof_xgb_v4.npy, oof_cat_v4.npy
         models/oof_preds_v4.npy
         models/test_preds_v4.npy
         models/test_ids_order_v4.parquet
         models/best_threshold_v4.txt
         submission.csv
"""
import polars as pl
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from scipy.stats import rankdata
from pathlib import Path
from datetime import timedelta
import json

ROOT = Path(r"C:\Users\ujjaw\Downloads\AML_Mule_Project")
DATA = ROOT / "data" / "archive"
FEAT = ROOT / "features"
MODELS = ROOT / "models"

# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: Load data + create pseudo-labels
# ═══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PHASE 1: Loading data and creating pseudo-labels")
print("=" * 70)

train = pl.read_parquet(FEAT / "train_features_v2.parquet")
test = pl.read_parquet(FEAT / "test_features_v2.parquet")

EXCLUDE = {"account_id", "is_mule"}
FEATURES = [c for c in train.columns if c not in EXCLUDE]

# Remove non-numeric columns
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

X_train_orig = train.select(FEATURES).to_numpy().astype(np.float32)
y_train_orig = train["is_mule"].to_numpy().astype(np.int32)
X_test = test.select(FEATURES).to_numpy().astype(np.float32)
train_account_ids = train["account_id"].to_list()
test_account_ids = test["account_id"].to_list()

# Clean inf/nan
X_train_orig = np.nan_to_num(X_train_orig, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

print(f"Original train: {X_train_orig.shape}, mule rate: {y_train_orig.mean():.4f}")
print(f"Test:           {X_test.shape}")

# ── Create pseudo-labels from rank-blended predictions ───────────────
rank_preds = np.load(MODELS / "test_preds_v3_rank.npy")

# Top 0.5% → mule, Bottom 10% → legit
top_threshold = np.percentile(rank_preds, 99.5)   # top 0.5%
bottom_threshold = np.percentile(rank_preds, 10.0)  # bottom 10%

pseudo_mule_mask = rank_preds >= top_threshold
pseudo_legit_mask = rank_preds <= bottom_threshold

n_pseudo_mule = pseudo_mule_mask.sum()
n_pseudo_legit = pseudo_legit_mask.sum()

print(f"\nPseudo-labeling:")
print(f"  Top 0.5% threshold (rank): {top_threshold:.4f} → {n_pseudo_mule} pseudo-mules")
print(f"  Bottom 10% threshold (rank): {bottom_threshold:.4f} → {n_pseudo_legit} pseudo-legit")

# Build pseudo-labeled arrays
pseudo_mask = pseudo_mule_mask | pseudo_legit_mask
X_pseudo = X_test[pseudo_mask]
y_pseudo = np.where(rank_preds[pseudo_mask] >= top_threshold, 1, 0).astype(np.int32)

print(f"  Pseudo-labeled samples: {X_pseudo.shape[0]} (mule={y_pseudo.sum()}, legit={len(y_pseudo)-y_pseudo.sum()})")

# Augment training data
X_train_aug = np.vstack([X_train_orig, X_pseudo])
y_train_aug = np.concatenate([y_train_orig, y_pseudo])

print(f"\nAugmented train: {X_train_aug.shape}, mule rate: {y_train_aug.mean():.4f}")

# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: Retrain LGB + XGB + CAT with 5-fold CV
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 2: Retraining with pseudo-labels")
print("=" * 70)

n_pos = y_train_aug.sum()
n_neg = len(y_train_aug) - n_pos
scale_pos_weight = n_neg / max(n_pos, 1)
print(f"Pos: {n_pos}, Neg: {n_neg}, scale_pos_weight: {scale_pos_weight:.1f}")

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

# 5-Fold CV on augmented data
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(y_train_aug))
oof_xgb = np.zeros(len(y_train_aug))
oof_cat = np.zeros(len(y_train_aug))

test_lgb_arr = np.zeros(len(X_test))
test_xgb_arr = np.zeros(len(X_test))
test_cat_arr = np.zeros(len(X_test))

print("\n── Cross-Validation ──")
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_aug, y_train_aug)):
    Xtr, Xval = X_train_aug[train_idx], X_train_aug[val_idx]
    ytr, yval = y_train_aug[train_idx], y_train_aug[val_idx]

    # LightGBM
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(
        Xtr, ytr,
        eval_set=[(Xval, yval)],
        callbacks=[lgb.log_evaluation(200), lgb.early_stopping(100)],
    )
    val_lgb = lgb_model.predict_proba(Xval)[:, 1]
    oof_lgb[val_idx] = val_lgb
    test_lgb_arr += lgb_model.predict_proba(X_test)[:, 1] / 5

    # XGBoost
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(
        Xtr, ytr,
        eval_set=[(Xval, yval)],
        verbose=False,
    )
    val_xgb = xgb_model.predict_proba(Xval)[:, 1]
    oof_xgb[val_idx] = val_xgb
    test_xgb_arr += xgb_model.predict_proba(X_test)[:, 1] / 5

    # CatBoost
    cat_model = CatBoostClassifier(**cat_params)
    cat_model.fit(
        Xtr, ytr,
        eval_set=(Xval, yval),
        verbose=0,
    )
    val_cat = cat_model.predict_proba(Xval)[:, 1]
    oof_cat[val_idx] = val_cat
    test_cat_arr += cat_model.predict_proba(X_test)[:, 1] / 5

    val_blend = (val_lgb + val_xgb + val_cat) / 3
    print(
        f"  Fold {fold}: LGB={roc_auc_score(yval, val_lgb):.5f}  "
        f"XGB={roc_auc_score(yval, val_xgb):.5f}  "
        f"CAT={roc_auc_score(yval, val_cat):.5f}  "
        f"Blend={roc_auc_score(yval, val_blend):.5f}"
    )

# ═══════════════════════════════════════════════════════════════════════
# PHASE 3: Rank-blend and evaluate
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 3: Rank blending and evaluation")
print("=" * 70)

# OOF evaluation on ORIGINAL train samples only (exclude pseudo-labeled)
n_orig = len(y_train_orig)
oof_lgb_orig = oof_lgb[:n_orig]
oof_xgb_orig = oof_xgb[:n_orig]
oof_cat_orig = oof_cat[:n_orig]

print(f"\nOOF AUC on ORIGINAL train ({n_orig} samples):")
print(f"  LGB: {roc_auc_score(y_train_orig, oof_lgb_orig):.5f}")
print(f"  XGB: {roc_auc_score(y_train_orig, oof_xgb_orig):.5f}")
print(f"  CAT: {roc_auc_score(y_train_orig, oof_cat_orig):.5f}")

# Probability blend on OOF
oof_prob_blend = (oof_lgb_orig + oof_xgb_orig + oof_cat_orig) / 3
print(f"  Prob blend: {roc_auc_score(y_train_orig, oof_prob_blend):.5f}")

# Rank blend on OOF
oof_rank_lgb = rankdata(oof_lgb_orig) / n_orig
oof_rank_xgb = rankdata(oof_xgb_orig) / n_orig
oof_rank_cat = rankdata(oof_cat_orig) / n_orig
oof_rank_blend = (oof_rank_lgb + oof_rank_xgb + oof_rank_cat) / 3
print(f"  Rank blend: {roc_auc_score(y_train_orig, oof_rank_blend):.5f}")

# Compare with v2 OOF
try:
    v2_oof = np.load(MODELS / "oof_preds_v2.npy")
    print(f"\n  v2 OOF AUC (reference): {roc_auc_score(y_train_orig, v2_oof):.5f}")
except Exception:
    pass

# ── Rank blend on test ───────────────────────────────────────────────
n_test = len(X_test)
test_rank_lgb = rankdata(test_lgb_arr) / n_test
test_rank_xgb = rankdata(test_xgb_arr) / n_test
test_rank_cat = rankdata(test_cat_arr) / n_test
test_rank_blend = (test_rank_lgb + test_rank_xgb + test_rank_cat) / 3

# ── Optimal F1 threshold on OOF ─────────────────────────────────────
best_f1, best_thr = 0, 0
for thr in np.linspace(0, 1, 1001):
    f1 = f1_score(y_train_orig, (oof_rank_blend > thr).astype(int))
    if f1 > best_f1:
        best_f1, best_thr = f1, thr

print(f"\n  Best OOF F1: {best_f1:.5f} at rank threshold {best_thr:.4f}")

# ── Save model artifacts ─────────────────────────────────────────────
np.save(MODELS / "oof_lgb_v4.npy", oof_lgb_orig)
np.save(MODELS / "oof_xgb_v4.npy", oof_xgb_orig)
np.save(MODELS / "oof_cat_v4.npy", oof_cat_orig)
np.save(MODELS / "oof_preds_v4.npy", oof_rank_blend)
np.save(MODELS / "test_preds_v4.npy", test_rank_blend)
np.save(MODELS / "test_lgb_v4.npy", test_lgb_arr)
np.save(MODELS / "test_xgb_v4.npy", test_xgb_arr)
np.save(MODELS / "test_cat_v4.npy", test_cat_arr)

test.select("account_id").write_parquet(MODELS / "test_ids_order_v4.parquet")

with open(MODELS / "best_threshold_v4.txt", "w") as f:
    f.write(f"{best_thr:.4f}")

# Feature importance from last LGB fold
importance = dict(zip(FEATURES, lgb_model.feature_importances_.tolist()))
importance = dict(sorted(importance.items(), key=lambda x: -x[1]))
with open(MODELS / "feature_importance_v4.json", "w") as f:
    json.dump(importance, f, indent=2)

print("\n  Top 20 features:")
for i, (feat, imp) in enumerate(list(importance.items())[:20]):
    print(f"    {i+1:2d}. {feat:40s} {imp}")

print(f"\n✓ All v4 model artifacts saved")

# ═══════════════════════════════════════════════════════════════════════
# PHASE 4: Dynamic temporal windows + submission
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 4: Dynamic temporal windows + submission")
print("=" * 70)

test_ids_df = test.select("account_id").with_columns(pl.Series("pred", test_rank_blend))

# Predicted mules using v4 threshold
predicted_mule_ids = (
    test_ids_df
    .filter(pl.col("pred") > best_thr)["account_id"]
    .to_list()
)
print(f"Predicted mules (threshold {best_thr:.4f}): {len(predicted_mule_ids)}")

# ── Scan daily transaction volumes for predicted mules ───────────────
print("Scanning transactions for temporal windows...")
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

# ── Peak-detection temporal windows ──────────────────────────────────
windows = {}
if daily_parts:
    daily = (
        pl.concat(daily_parts)
        .group_by(["account_id", "d"])
        .agg([
            pl.col("daily_vol").sum(),
            pl.col("daily_txns").sum(),
        ])
    )

    print(f"Building dynamic windows for {len(predicted_mule_ids)} accounts...")
    processed = 0

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
            processed += 1
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

        all_dates = vol_series.index
        rolling_vals = rolling_avg.values

        left = peak_idx
        while left > 0 and rolling_vals[left - 1] >= baseline:
            left -= 1

        right = peak_idx
        while right < len(rolling_vals) - 1 and rolling_vals[right + 1] >= baseline:
            right += 1

        start_date = all_dates[left]
        end_date = all_dates[right] + timedelta(days=1)

        windows[acc_id] = (
            start_date.strftime("%Y-%m-%d %H:%M:%S"),
            end_date.strftime("%Y-%m-%d %H:%M:%S"),
        )

        processed += 1
        if processed % 500 == 0:
            print(f"  Processed {processed}/{len(predicted_mule_ids)} accounts...")

    print(f"  Processed {processed}/{len(predicted_mule_ids)} accounts total")

# ── Window stats ─────────────────────────────────────────────────────
if windows:
    window_days = []
    for acc_id, (s, e) in windows.items():
        dt_s = pd.Timestamp(s)
        dt_e = pd.Timestamp(e)
        window_days.append((dt_e - dt_s).days)
    window_days = np.array(window_days)
    print(f"\nWindow size stats (days):")
    print(f"  Mean: {window_days.mean():.1f}, Median: {np.median(window_days):.1f}")
    print(f"  Min: {window_days.min()}, Max: {window_days.max()}")

# ── Build submission ─────────────────────────────────────────────────
sub_df = test_ids_df.select(["account_id", "pred"]).rename({"pred": "is_mule"})

suspicious_starts = []
suspicious_ends = []
for row in sub_df.iter_rows(named=True):
    acc_id = row["account_id"]
    prob = row["is_mule"]
    if prob > best_thr and acc_id in windows:
        s, e = windows[acc_id]
        suspicious_starts.append(s)
        suspicious_ends.append(e)
    else:
        suspicious_starts.append("")
        suspicious_ends.append("")

submission = pd.DataFrame({
    "account_id": sub_df["account_id"].to_list(),
    "is_mule": sub_df["is_mule"].to_list(),
    "suspicious_start": suspicious_starts,
    "suspicious_end": suspicious_ends,
})

out_path = ROOT / "submission.csv"
submission.to_csv(out_path, index=False)

n_with_windows = sum(1 for s in suspicious_starts if s)
is_mule_arr = submission["is_mule"].values
print(f"\n✓ submission.csv saved ({submission.shape})")
print(f"  is_mule stats: mean={is_mule_arr.mean():.4f}, min={is_mule_arr.min():.4f}, max={is_mule_arr.max():.4f}")
print(f"  Accounts with temporal windows: {n_with_windows}")
print(f"\n  Sample predicted mules:")
print(submission[submission["is_mule"] > best_thr].head(10))
