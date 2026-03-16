"""
Step 2d — OOF Pair Target-Encoding Features
Builds leakage-safe account-level risk features using OOF target encoding for
(counterparty_id, channel).
"""
from pathlib import Path
import gc
import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedKFold

ROOT = Path(r"C:\Users\ujjaw\Downloads\AML_Mule_Project")
DATA = ROOT / "data" / "archive"
FEAT = ROOT / "features"

N_FOLDS = 5
SEED = 42
ALPHA_PAIR = 60.0
HIGH_RISK_THR = 0.20


def add_fold_assignments(labels: pl.DataFrame) -> pl.DataFrame:
    y = labels["is_mule"].to_numpy()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = np.zeros(len(labels), dtype=np.int8)
    for f, (_, va_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        folds[va_idx] = f
    return labels.with_columns(pl.Series("fold", folds))


def combine_stats(dfs: list[pl.DataFrame], keys: list[str]) -> pl.DataFrame:
    return (
        pl.concat(dfs)
        .group_by(keys)
        .agg(
            pl.col("target_sum").sum().alias("target_sum"),
            pl.col("target_cnt").sum().alias("target_cnt"),
        )
    )


def make_fold_te_table(global_stats: pl.DataFrame, fold_stats: pl.DataFrame, prior: float) -> pl.DataFrame:
    out = []
    for f in range(N_FOLDS):
        fs = (
            fold_stats
            .filter(pl.col("fold") == f)
            .select([
                "counterparty_id", "channel",
                pl.col("target_sum").alias("fold_sum"),
                pl.col("target_cnt").alias("fold_cnt"),
            ])
        )
        te_f = (
            global_stats
            .join(fs, on=["counterparty_id", "channel"], how="left")
            .fill_null(0)
            .with_columns(
                ((pl.col("target_sum") - pl.col("fold_sum") + ALPHA_PAIR * prior)
                 / (pl.col("target_cnt") - pl.col("fold_cnt") + ALPHA_PAIR)).alias("te_pair")
            )
            .select(["counterparty_id", "channel", "te_pair"])
            .with_columns(pl.lit(f).alias("fold"))
            .select(["fold", "counterparty_id", "channel", "te_pair"])
        )
        out.append(te_f)
    return pl.concat(out)


def make_full_te_table(global_stats: pl.DataFrame, prior: float) -> pl.DataFrame:
    return global_stats.with_columns(
        ((pl.col("target_sum") + ALPHA_PAIR * prior) / (pl.col("target_cnt") + ALPHA_PAIR)).alias("te_pair")
    ).select(["counterparty_id", "channel", "te_pair"])


def agg_batch(df: pl.LazyFrame) -> pl.DataFrame:
    return (
        df.group_by("account_id")
        .agg(
            pl.len().alias("txn_cnt"),
            pl.col("te_pair").sum().alias("sum_te"),
            (pl.col("te_pair") * pl.col("te_pair")).sum().alias("sumsq_te"),
            pl.col("te_pair").max().alias("max_te"),
            (pl.col("amount") * pl.col("te_pair")).sum().alias("sum_amt_te"),
            pl.col("amount").sum().alias("sum_amt"),
            (pl.col("te_pair") >= HIGH_RISK_THR).cast(pl.Int32).sum().alias("high_risk_cnt"),
        )
        .collect()
    )


def finalize(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by("account_id")
        .agg(
            pl.col("txn_cnt").sum().alias("txn_cnt"),
            pl.col("sum_te").sum().alias("sum_te"),
            pl.col("sumsq_te").sum().alias("sumsq_te"),
            pl.col("max_te").max().alias("max_te"),
            pl.col("sum_amt_te").sum().alias("sum_amt_te"),
            pl.col("sum_amt").sum().alias("sum_amt"),
            pl.col("high_risk_cnt").sum().alias("high_risk_cnt"),
        )
        .with_columns(
            (pl.col("sum_te") / (pl.col("txn_cnt") + 1e-9)).alias("te_pair_mean"),
            ((pl.col("sumsq_te") / (pl.col("txn_cnt") + 1e-9) -
              (pl.col("sum_te") / (pl.col("txn_cnt") + 1e-9)) ** 2)
             .clip(lower_bound=0)
             .sqrt()).alias("te_pair_std"),
            pl.col("max_te").alias("te_pair_max"),
            (pl.col("sum_amt_te") / (pl.col("sum_amt") + 1e-9)).alias("te_pair_amt_weighted"),
            (pl.col("high_risk_cnt") / (pl.col("txn_cnt") + 1e-9)).alias("te_pair_high_risk_ratio"),
        )
        .select([
            "account_id",
            "te_pair_mean", "te_pair_std", "te_pair_max",
            "te_pair_amt_weighted", "te_pair_high_risk_ratio",
        ])
    )


labels = pl.read_parquet(DATA / "train_labels.parquet").select(["account_id", "is_mule"])
test_ids = pl.read_parquet(DATA / "test_accounts.parquet").select(["account_id"])
prior = float(labels["is_mule"].mean())
print(f"Train: {labels.shape[0]}  Test: {test_ids.shape[0]}  Prior: {prior:.5f}")

labels_f = add_fold_assignments(labels)
fold_map_lf = labels_f.select(["account_id", "fold", "is_mule"]).lazy()

# Pass A: stats from labeled accounts only
pair_global_parts, pair_fold_parts = [], []
for b in range(1, 5):
    bdir = DATA / "transactions" / f"batch-{b}"
    print(f"[Pass A] batch-{b}...")
    dfl = (
        pl.scan_parquet(str(bdir / "*.parquet"))
        .select(["account_id", "counterparty_id", "channel"])
        .join(fold_map_lf, on="account_id", how="inner")
    )
    pair_global_parts.append(
        dfl.group_by(["counterparty_id", "channel"]).agg(
            pl.col("is_mule").sum().alias("target_sum"),
            pl.len().alias("target_cnt"),
        ).collect()
    )
    pair_fold_parts.append(
        dfl.group_by(["fold", "counterparty_id", "channel"]).agg(
            pl.col("is_mule").sum().alias("target_sum"),
            pl.len().alias("target_cnt"),
        ).collect()
    )
    gc.collect()

pair_global = combine_stats(pair_global_parts, ["counterparty_id", "channel"])
pair_fold = combine_stats(pair_fold_parts, ["fold", "counterparty_id", "channel"])

print("Building pair TE lookups...")
pair_te_fold = make_fold_te_table(pair_global, pair_fold, prior).lazy()
pair_te_full = make_full_te_table(pair_global, prior).lazy()

# Pass B: apply TE and aggregate per account
train_parts, test_parts = [], []
for b in range(1, 5):
    bdir = DATA / "transactions" / f"batch-{b}"
    print(f"[Pass B] batch-{b}...")
    lf = pl.scan_parquet(str(bdir / "*.parquet")).select([
        "account_id", "counterparty_id", "channel", "amount"
    ])

    train_lf = (
        lf.join(fold_map_lf.select(["account_id", "fold"]), on="account_id", how="inner")
        .join(pair_te_fold, on=["fold", "counterparty_id", "channel"], how="left")
        .with_columns(pl.col("te_pair").fill_null(prior))
    )
    train_parts.append(agg_batch(train_lf))

    test_lf = (
        lf.join(test_ids.lazy(), on="account_id", how="inner")
        .join(pair_te_full, on=["counterparty_id", "channel"], how="left")
        .with_columns(pl.col("te_pair").fill_null(prior))
    )
    test_parts.append(agg_batch(test_lf))

    gc.collect()

train_feat = finalize(pl.concat(train_parts))
test_feat = finalize(pl.concat(test_parts))

all_ids = pl.concat([labels.select("account_id"), test_ids]).unique()
out = (
    all_ids
    .join(train_feat, on="account_id", how="left")
    .join(test_feat, on="account_id", how="left", suffix="_test")
    .with_columns(
        *[
            pl.coalesce(pl.col(c), pl.col(f"{c}_test")).alias(c)
            for c in [
                "te_pair_mean", "te_pair_std", "te_pair_max",
                "te_pair_amt_weighted", "te_pair_high_risk_ratio",
            ]
        ]
    )
    .select([
        "account_id",
        "te_pair_mean", "te_pair_std", "te_pair_max",
        "te_pair_amt_weighted", "te_pair_high_risk_ratio",
    ])
    .fill_null(prior)
)

print(f"Pair TE features: {out.shape}")
out.write_parquet(FEAT / "txn_oof_pair_te_features.parquet")
print(f"✓ {(FEAT / 'txn_oof_pair_te_features.parquet')} saved")
