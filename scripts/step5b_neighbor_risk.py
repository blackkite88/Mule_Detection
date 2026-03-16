"""
Step 2f — Neighbor Mule Risk Feature
Computes 1-hop risk by averaging v2 model scores of counterparties.
Uses OOF preds for train accounts and test preds for test accounts.
"""
import polars as pl
import numpy as np
from pathlib import Path

ROOT = Path(r"C:\Users\ujjaw\Downloads\AML_Mule_Project")
DATA = ROOT / "data" / "archive"
MODELS = ROOT / "models"
FEAT = ROOT / "features"

# Load account lists and scores
train = pl.read_parquet(FEAT / "train_features_v2.parquet").select(["account_id", "is_mule"])
test = pl.read_parquet(FEAT / "test_features_v2.parquet").select(["account_id"])

oof = np.load(MODELS / "oof_preds_v2.npy")
test_preds = np.load(MODELS / "test_preds_v2.npy")

train_scores = train.with_columns(pl.Series("score", oof))
test_scores = test.with_columns(pl.Series("score", test_preds))

score_map = pl.concat([
    train_scores.select(pl.col("account_id").alias("counterparty_id"), pl.col("score")),
    test_scores.select(pl.col("account_id").alias("counterparty_id"), pl.col("score")),
])

score_map = score_map.filter(pl.col("counterparty_id").str.starts_with("ACCT"))

print(f"Scores loaded for {score_map.height} counterparties")

parts = []
for batch in ["batch-1", "batch-2", "batch-3", "batch-4"]:
    print(f"Processing {batch} ...")
    df = (
        pl.scan_parquet(DATA / "transactions" / batch / "*.parquet")
        .select(["account_id", "counterparty_id", "amount"])
        .filter(pl.col("counterparty_id").str.starts_with("ACCT"))
        .join(score_map.lazy(), on="counterparty_id", how="inner")
        .group_by("account_id")
        .agg([
            pl.col("score").mean().alias("neighbor_score_mean"),
            pl.col("score").max().alias("neighbor_score_max"),
            (pl.col("score") * pl.col("amount").abs()).sum().alias("neighbor_score_weighted_sum"),
            pl.col("score").count().alias("neighbor_score_count"),
        ])
    ).collect()
    parts.append(df)

print("Aggregating across batches...")
combined = pl.concat(parts).group_by("account_id").agg([
    pl.col("neighbor_score_mean").mean(),
    pl.col("neighbor_score_max").max(),
    pl.col("neighbor_score_weighted_sum").sum(),
    pl.col("neighbor_score_count").sum(),
])

combined = combined.with_columns([
    (pl.col("neighbor_score_weighted_sum") / (pl.col("neighbor_score_count") + 1e-3)).alias("neighbor_score_weighted_mean"),
])

combined = combined.rename({
    "neighbor_score_mean": "neighbor_score_mean",
    "neighbor_score_max": "neighbor_score_max",
    "neighbor_score_weighted_mean": "neighbor_score_weighted_mean",
    "neighbor_score_count": "neighbor_score_count",
})

out_path = FEAT / "neighbor_risk.parquet"
combined.write_parquet(out_path)
print(f"✓ Saved {out_path} with shape {combined.shape}")
