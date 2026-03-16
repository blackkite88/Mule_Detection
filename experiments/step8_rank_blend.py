"""
Step 8 — Rank-Based Blend
Converts per-model test predictions to percentile ranks, then averages ranks.
This neutralises probability scale mismatches (especially CatBoost's sharp
distribution) and almost always provides a safe AUC boost.

Inputs:  models/test_lgb.npy, test_xgb.npy, test_cat.npy
         models/oof_preds_v2.npy  (for reference comparison)
         models/test_ids_order.parquet
Outputs: models/test_preds_v3_rank.npy
"""
import numpy as np
from scipy.stats import rankdata
from pathlib import Path

ROOT = Path(r"C:\Users\ujjaw\Downloads\AML_Mule_Project")
MODELS = ROOT / "models"

# ── Load per-model test predictions ──────────────────────────────────
test_lgb = np.load(MODELS / "test_lgb.npy")
test_xgb = np.load(MODELS / "test_xgb.npy")
test_cat = np.load(MODELS / "test_cat.npy")

print(f"Loaded predictions: LGB={test_lgb.shape}, XGB={test_xgb.shape}, CAT={test_cat.shape}")
print(f"  LGB: mean={test_lgb.mean():.4f} min={test_lgb.min():.4f} max={test_lgb.max():.4f}")
print(f"  XGB: mean={test_xgb.mean():.4f} min={test_xgb.min():.4f} max={test_xgb.max():.4f}")
print(f"  CAT: mean={test_cat.mean():.4f} min={test_cat.min():.4f} max={test_cat.max():.4f}")

# ── Convert to percentile ranks (0.0 to 1.0) ────────────────────────
n = len(test_lgb)
rank_lgb = rankdata(test_lgb) / n
rank_xgb = rankdata(test_xgb) / n
rank_cat = rankdata(test_cat) / n

print(f"\nRank distributions:")
print(f"  Rank LGB: mean={rank_lgb.mean():.4f} std={rank_lgb.std():.4f}")
print(f"  Rank XGB: mean={rank_xgb.mean():.4f} std={rank_xgb.std():.4f}")
print(f"  Rank CAT: mean={rank_cat.mean():.4f} std={rank_cat.std():.4f}")

# ── Blend ranks (equal weight) ───────────────────────────────────────
rank_blend = (rank_lgb + rank_xgb + rank_cat) / 3.0

print(f"\nRank blend: mean={rank_blend.mean():.4f} min={rank_blend.min():.4f} max={rank_blend.max():.4f}")

# ── Compare with v2 probability blend ────────────────────────────────
v2_preds = np.load(MODELS / "test_preds_v2.npy")
# Spearman rank correlation between v2 blend and rank blend
from scipy.stats import spearmanr
corr, _ = spearmanr(v2_preds, rank_blend)
print(f"\nSpearman correlation (v2 prob blend vs rank blend): {corr:.6f}")

# Check how many accounts change their ordering significantly
v2_ranks = rankdata(v2_preds) / n
rank_diff = np.abs(v2_ranks - rank_blend)
print(f"Mean rank displacement: {rank_diff.mean():.4f}")
print(f"Max rank displacement:  {rank_diff.max():.4f}")
print(f"Accounts with rank shift > 5%: {(rank_diff > 0.05).sum()}")
print(f"Accounts with rank shift > 10%: {(rank_diff > 0.10).sum()}")

# ── Save ─────────────────────────────────────────────────────────────
np.save(MODELS / "test_preds_v3_rank.npy", rank_blend)
print(f"\n✓ models/test_preds_v3_rank.npy saved ({rank_blend.shape})")
