"""
Complete SHAP Analysis Suite (V11)
==================================
Generates every major SHAP visualization for forensic model explainability.
1. Global Importance (Bar)
2. Beeswarm (Distribution)
3. Waterfall (Local Explanation - Mule)
4. Waterfall (Local Explanation - Legit)
5. Dependence Plots (Top 2 Features)
6. Heatmap (Sample Patterns)
7. Decision Plot (Cumulative Logic)
"""
import polars as pl
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# Executive Theme
THEME_BG = '#ffffff'
THEME_TEXT = '#1e293b'
THEME_GRID = '#e2e8f0'

plt.rcParams.update({
    'text.color': THEME_TEXT, 'axes.labelcolor': THEME_TEXT,
    'xtick.color': THEME_TEXT, 'ytick.color': THEME_TEXT,
    'axes.facecolor': THEME_BG, 'figure.facecolor': THEME_BG,
    'axes.edgecolor': THEME_GRID, 'font.family': 'sans-serif'
})

ROOT = Path(r"C:\Users\ujjaw\Downloads\AML_Mule_Project")
FEAT = ROOT / "features"
MODELS = ROOT / "models"
ASSETS = ROOT / "report_assets"
ASSETS.mkdir(parents=True, exist_ok=True)

print("🚀 Starting Complete SHAP Analysis Suite...")

# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
train_df = pl.read_parquet(FEAT / "train_features_v2.parquet")
shap_rec = pl.read_parquet(FEAT / "shap_recommended_features_v2.parquet")
df = train_df.join(shap_rec, on="account_id", how="left")

FEATURES = [c for c in df.columns if c not in ["account_id", "is_mule", "sample_weight"]]
FEATURES = [c for c in FEATURES if df[c].dtype != pl.Utf8 and not c.startswith("te_")]

sample_size = 1000
df_sample = df.sample(n=sample_size, seed=42)
X = df_sample.select(FEATURES).to_numpy().astype(np.float32)
y = df_sample["is_mule"].to_numpy().astype(np.int32)
X = np.nan_to_num(X, nan=0.0)

import lightgbm as lgb
model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, verbose=-1)
model.fit(X, y)

explainer = shap.TreeExplainer(model)
s_values_raw = explainer.shap_values(X)
expected_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
s_vals_class1 = s_values_raw[1] if isinstance(s_values_raw, list) else s_values_raw

# Create Explanation object for newer plots
from shap import Explanation
shap_exp = Explanation(values=s_vals_class1, base_values=expected_val, data=X, feature_names=FEATURES)

print("\n--- Generating Plots ---")

# A. Global Bar
plt.figure(figsize=(10, 7))
shap.plots.bar(shap_exp, show=False)
plt.title("Global Feature Importance", fontsize=16, pad=20)
plt.tight_layout(); plt.savefig(ASSETS / "shap_importance_final.png", dpi=200)

# B. Beeswarm
plt.figure(figsize=(10, 8))
shap.plots.beeswarm(shap_exp, show=False)
plt.title("Impact Distribution", fontsize=16, pad=20)
plt.tight_layout(); plt.savefig(ASSETS / "shap_beeswarm_final.png", dpi=200)

# C & D. Waterfall
mule_idx = int(np.argmax(model.predict_proba(X)[:, 1]))
legit_idx = int(np.argmin(model.predict_proba(X)[:, 1]))

plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap_exp[mule_idx], show=False)
plt.title(f"Mule Local Explanation: {df_sample['account_id'][mule_idx]}", pad=20)
plt.tight_layout(); plt.savefig(ASSETS / "shap_waterfall_mule.png")

plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap_exp[legit_idx], show=False)
plt.title(f"Legitimate Local Explanation: {df_sample['account_id'][legit_idx]}", pad=20)
plt.tight_layout(); plt.savefig(ASSETS / "shap_waterfall_legit.png")

# E. Dependence Plot (Legacy API is more stable for this)
top_idx = np.abs(s_vals_class1).mean(0).argsort()[::-1][0]
plt.figure(figsize=(8, 6))
shap.dependence_plot(top_idx, s_vals_class1, X, feature_names=FEATURES, show=False)
plt.title(f"Dependence: {FEATURES[top_idx]}", pad=15)
plt.tight_layout(); plt.savefig(ASSETS / "shap_dependence_0.png")

# F. Heatmap
plt.figure(figsize=(12, 8))
shap.plots.heatmap(shap_exp[:300], show=False)
plt.tight_layout(); plt.savefig(ASSETS / "shap_heatmap.png")

# G. Decision Plot
plt.figure(figsize=(10, 8))
shap.decision_plot(expected_val, s_vals_class1[:50], feature_names=FEATURES, show=False)
plt.tight_layout(); plt.savefig(ASSETS / "shap_decision.png")

print("✓ All Plots Generated.")
