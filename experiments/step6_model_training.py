"""
Step 6 — Model Training
LightGBM with 5-fold CV, class-imbalance handling, feature importance.
"""
import polars as pl
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from pathlib import Path
import json

ROOT = Path(r"C:\Users\ujjaw\Downloads\AML_Mule_Project")
FEAT = ROOT / "features"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────
train = pl.read_parquet(FEAT / "train_features.parquet")
test  = pl.read_parquet(FEAT / "test_features.parquet")

EXCLUDE = {"account_id", "is_mule"}
FEATURES = [c for c in train.columns if c not in EXCLUDE]

# Verify all features are numeric
for c in FEATURES:
    dt = train[c].dtype
    if dt not in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                  pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                  pl.Float32, pl.Float64):
        print(f"WARNING: dropping non-numeric column '{c}' (dtype={dt})")
        FEATURES.remove(c)

print(f"Using {len(FEATURES)} features")

X_train = train.select(FEATURES).to_numpy().astype(np.float32)
y_train = train["is_mule"].to_numpy().astype(np.int32)
X_test  = test.select(FEATURES).to_numpy().astype(np.float32)

print(f"X_train: {X_train.shape}, y_train mean: {y_train.mean():.4f}")
print(f"X_test:  {X_test.shape}")

# ── LightGBM parameters ─────────────────────────────────────────────
params = {
    "n_estimators": 800,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "is_unbalance": True,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

# ── 5-fold Stratified CV ────────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(y_train))
test_preds = np.zeros(len(X_test))

print("\n── Cross-Validation ──")
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    Xtr, Xval = X_train[train_idx], X_train[val_idx]
    ytr, yval = y_train[train_idx], y_train[val_idx]

    model = lgb.LGBMClassifier(**params)
    model.fit(
        Xtr, ytr,
        eval_set=[(Xval, yval)],
        callbacks=[lgb.log_evaluation(100), lgb.early_stopping(50)],
    )

    val_pred = model.predict_proba(Xval)[:, 1]
    oof_preds[val_idx] = val_pred

    auc = roc_auc_score(yval, val_pred)
    f1  = f1_score(yval, (val_pred > 0.5).astype(int))
    print(f"  Fold {fold}: AUC={auc:.5f}  F1={f1:.5f}")

    test_preds += model.predict_proba(X_test)[:, 1] / 5

# ── Overall OOF metrics ─────────────────────────────────────────────
oof_auc = roc_auc_score(y_train, oof_preds)
oof_f1  = f1_score(y_train, (oof_preds > 0.5).astype(int))
print(f"\n  OOF AUC:  {oof_auc:.5f}")
print(f"  OOF F1:   {oof_f1:.5f}")

# ── Feature importance ───────────────────────────────────────────────
importance = dict(zip(FEATURES, model.feature_importances_.tolist()))
importance = dict(sorted(importance.items(), key=lambda x: -x[1]))
print("\nTop 20 features:")
for i, (feat, imp) in enumerate(list(importance.items())[:20]):
    print(f"  {i+1:2d}. {feat:35s} {imp}")

with open(MODELS / "feature_importance.json", "w") as f:
    json.dump(importance, f, indent=2)

# ── Save predictions ─────────────────────────────────────────────────
np.save(MODELS / "test_preds.npy", test_preds)
np.save(MODELS / "oof_preds.npy",  oof_preds)

print(f"\n✓ models/test_preds.npy saved")
print(f"✓ models/oof_preds.npy saved")
print(f"✓ models/feature_importance.json saved")
