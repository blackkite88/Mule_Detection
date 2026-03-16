"""
Step 2c — OOF Transaction Target-Encoding Features
Builds leakage-safe account-level transaction risk features using OOF target
encodings for:
- counterparty_id
- mcc_code
- channel

Approach:
1) Build global and per-fold label stats per key from labeled accounts only.
2) Convert those into fold-specific TE lookup tables.
3) Apply fold-specific TEs to train accounts (OOF), and full TEs to test accounts.
4) Aggregate per-account transaction risk stats.
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
ALPHA_CPARTY = 50.0
ALPHA_MCC = 20.0
ALPHA_CHANNEL = 20.0
HIGH_RISK_THR = 0.20


def add_fold_assignments(labels: pl.DataFrame) -> pl.DataFrame:
    y = labels["is_mule"].to_numpy()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = np.zeros(len(labels), dtype=np.int8)
    for f, (_, va_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        folds[va_idx] = f
    return labels.with_columns(pl.Series("fold", folds))


def combine_stats(dfs: list[pl.DataFrame], keys: list[str]) -> pl.DataFrame:
    if not dfs:
        return pl.DataFrame({k: [] for k in keys} | {"target_sum": [], "target_cnt": []})
    return (
        pl.concat(dfs)
        .group_by(keys)
        .agg(
            pl.col("target_sum").sum().alias("target_sum"),
            pl.col("target_cnt").sum().alias("target_cnt"),
        )
    )


def make_fold_te_table(
    global_stats: pl.DataFrame,
    fold_stats: pl.DataFrame,
    key_col: str,
    prior: float,
    alpha: float,
) -> pl.DataFrame:
    out = []
    for f in range(N_FOLDS):
        fs = (
            fold_stats
            .filter(pl.col("fold") == f)
            .select([
                pl.col(key_col),
                pl.col("target_sum").alias("fold_sum"),
                pl.col("target_cnt").alias("fold_cnt"),
            ])
        )
        te_f = (
            global_stats
            .join(fs, on=key_col, how="left")
            .fill_null(0)
            .with_columns(
                ((pl.col("target_sum") - pl.col("fold_sum") + alpha * prior)
                 / (pl.col("target_cnt") - pl.col("fold_cnt") + alpha)).alias("te")
            )
            .select(["te", key_col])
            .with_columns(pl.lit(f).alias("fold"))
            .select(["fold", key_col, "te"])
        )
        out.append(te_f)
    return pl.concat(out)


def make_full_te_table(
    global_stats: pl.DataFrame,
    key_col: str,
    prior: float,
    alpha: float,
) -> pl.DataFrame:
    return global_stats.with_columns(
        ((pl.col("target_sum") + alpha * prior) / (pl.col("target_cnt") + alpha)).alias("te")
    ).select([key_col, "te"])


def agg_account_stats(df: pl.LazyFrame, te_cp_col: str, te_mcc_col: str, te_ch_col: str) -> pl.DataFrame:
    return (
        df.group_by("account_id")
        .agg(
            pl.len().alias("txn_cnt"),
            pl.col(te_cp_col).sum().alias("sum_te_cp"),
            (pl.col(te_cp_col) * pl.col(te_cp_col)).sum().alias("sumsq_te_cp"),
            pl.col(te_cp_col).max().alias("max_te_cp"),
            pl.col(te_mcc_col).sum().alias("sum_te_mcc"),
            pl.col(te_ch_col).sum().alias("sum_te_ch"),
            (pl.col("amount") * pl.col(te_cp_col)).sum().alias("sum_amt_te_cp"),
            pl.col("amount").sum().alias("sum_amt"),
            (pl.col(te_cp_col) >= HIGH_RISK_THR).cast(pl.Int32).sum().alias("high_risk_cnt"),
        )
        .collect()
    )


def finalize_account_features(df: pl.DataFrame, prefix: str = "te") -> pl.DataFrame:
    out = (
        df.group_by("account_id")
        .agg(
            pl.col("txn_cnt").sum().alias("txn_cnt"),
            pl.col("sum_te_cp").sum().alias("sum_te_cp"),
            pl.col("sumsq_te_cp").sum().alias("sumsq_te_cp"),
            pl.col("max_te_cp").max().alias("max_te_cp"),
            pl.col("sum_te_mcc").sum().alias("sum_te_mcc"),
            pl.col("sum_te_ch").sum().alias("sum_te_ch"),
            pl.col("sum_amt_te_cp").sum().alias("sum_amt_te_cp"),
            pl.col("sum_amt").sum().alias("sum_amt"),
            pl.col("high_risk_cnt").sum().alias("high_risk_cnt"),
        )
        .with_columns(
            (pl.col("sum_te_cp") / (pl.col("txn_cnt") + 1e-9)).alias(f"{prefix}_cp_mean"),
            ((pl.col("sumsq_te_cp") / (pl.col("txn_cnt") + 1e-9) -
              (pl.col("sum_te_cp") / (pl.col("txn_cnt") + 1e-9)) ** 2)
             .clip(lower_bound=0)
             .sqrt()).alias(f"{prefix}_cp_std"),
            pl.col("max_te_cp").alias(f"{prefix}_cp_max"),
            (pl.col("sum_te_mcc") / (pl.col("txn_cnt") + 1e-9)).alias(f"{prefix}_mcc_mean"),
            (pl.col("sum_te_ch") / (pl.col("txn_cnt") + 1e-9)).alias(f"{prefix}_channel_mean"),
            (pl.col("sum_amt_te_cp") / (pl.col("sum_amt") + 1e-9)).alias(f"{prefix}_cp_amt_weighted"),
            (pl.col("high_risk_cnt") / (pl.col("txn_cnt") + 1e-9)).alias(f"{prefix}_high_risk_ratio"),
        )
        .select([
            "account_id",
            f"{prefix}_cp_mean",
            f"{prefix}_cp_std",
            f"{prefix}_cp_max",
            f"{prefix}_mcc_mean",
            f"{prefix}_channel_mean",
            f"{prefix}_cp_amt_weighted",
            f"{prefix}_high_risk_ratio",
        ])
    )
    return out


print("Loading labels and IDs...")
labels = pl.read_parquet(DATA / "train_labels.parquet").select(["account_id", "is_mule"])
test_ids = pl.read_parquet(DATA / "test_accounts.parquet").select(["account_id"])
prior = float(labels["is_mule"].mean())
print(f"Train accounts: {labels.shape[0]}  Test accounts: {test_ids.shape[0]}  Prior: {prior:.5f}")

labels_f = add_fold_assignments(labels)
fold_map = labels_f.select(["account_id", "is_mule", "fold"])
fold_map_lf = fold_map.lazy()

# Pass A: key stats from labeled accounts only
cp_global_parts, cp_fold_parts = [], []
mcc_global_parts, mcc_fold_parts = [], []
ch_global_parts, ch_fold_parts = [], []

for b in range(1, 5):
    bdir = DATA / "transactions" / f"batch-{b}"
    print(f"[Pass A] batch-{b}...")

    lf = pl.scan_parquet(str(bdir / "*.parquet")).select(
        ["account_id", "counterparty_id", "mcc_code", "channel", "amount"]
    )
    dfl = lf.join(fold_map_lf, on="account_id", how="inner")

    cp_global_parts.append(
        dfl.group_by("counterparty_id").agg(
            pl.col("is_mule").sum().alias("target_sum"),
            pl.len().alias("target_cnt"),
        ).collect()
    )
    cp_fold_parts.append(
        dfl.group_by(["fold", "counterparty_id"]).agg(
            pl.col("is_mule").sum().alias("target_sum"),
            pl.len().alias("target_cnt"),
        ).collect()
    )

    mcc_global_parts.append(
        dfl.group_by("mcc_code").agg(
            pl.col("is_mule").sum().alias("target_sum"),
            pl.len().alias("target_cnt"),
        ).collect()
    )
    mcc_fold_parts.append(
        dfl.group_by(["fold", "mcc_code"]).agg(
            pl.col("is_mule").sum().alias("target_sum"),
            pl.len().alias("target_cnt"),
        ).collect()
    )

    ch_global_parts.append(
        dfl.group_by("channel").agg(
            pl.col("is_mule").sum().alias("target_sum"),
            pl.len().alias("target_cnt"),
        ).collect()
    )
    ch_fold_parts.append(
        dfl.group_by(["fold", "channel"]).agg(
            pl.col("is_mule").sum().alias("target_sum"),
            pl.len().alias("target_cnt"),
        ).collect()
    )

    gc.collect()

cp_global = combine_stats(cp_global_parts, ["counterparty_id"])
cp_fold = combine_stats(cp_fold_parts, ["fold", "counterparty_id"])
mcc_global = combine_stats(mcc_global_parts, ["mcc_code"])
mcc_fold = combine_stats(mcc_fold_parts, ["fold", "mcc_code"])
ch_global = combine_stats(ch_global_parts, ["channel"])
ch_fold = combine_stats(ch_fold_parts, ["fold", "channel"])

print("Creating TE lookup tables...")
cp_te_fold = make_fold_te_table(cp_global, cp_fold, "counterparty_id", prior, ALPHA_CPARTY)
mcc_te_fold = make_fold_te_table(mcc_global, mcc_fold, "mcc_code", prior, ALPHA_MCC)
ch_te_fold = make_fold_te_table(ch_global, ch_fold, "channel", prior, ALPHA_CHANNEL)

cp_te_full = make_full_te_table(cp_global, "counterparty_id", prior, ALPHA_CPARTY)
mcc_te_full = make_full_te_table(mcc_global, "mcc_code", prior, ALPHA_MCC)
ch_te_full = make_full_te_table(ch_global, "channel", prior, ALPHA_CHANNEL)

# Pass B: apply TEs and aggregate to account-level features
cp_te_fold_lf = cp_te_fold.lazy().rename({"te": "te_cp"})
mcc_te_fold_lf = mcc_te_fold.lazy().rename({"te": "te_mcc"})
ch_te_fold_lf = ch_te_fold.lazy().rename({"te": "te_ch"})

cp_te_full_lf = cp_te_full.lazy().rename({"te": "te_cp"})
mcc_te_full_lf = mcc_te_full.lazy().rename({"te": "te_mcc"})
ch_te_full_lf = ch_te_full.lazy().rename({"te": "te_ch"})

test_ids_lf = test_ids.lazy()

train_parts = []
test_parts = []

for b in range(1, 5):
    bdir = DATA / "transactions" / f"batch-{b}"
    print(f"[Pass B] batch-{b}...")

    lf = pl.scan_parquet(str(bdir / "*.parquet")).select(
        ["account_id", "counterparty_id", "mcc_code", "channel", "amount"]
    )

    train_lf = (
        lf.join(fold_map_lf.select(["account_id", "fold"]), on="account_id", how="inner")
        .join(cp_te_fold_lf, on=["fold", "counterparty_id"], how="left")
        .join(mcc_te_fold_lf, on=["fold", "mcc_code"], how="left")
        .join(ch_te_fold_lf, on=["fold", "channel"], how="left")
        .with_columns(
            pl.col("te_cp").fill_null(prior),
            pl.col("te_mcc").fill_null(prior),
            pl.col("te_ch").fill_null(prior),
        )
    )
    train_parts.append(agg_account_stats(train_lf, "te_cp", "te_mcc", "te_ch"))

    test_lf = (
        lf.join(test_ids_lf, on="account_id", how="inner")
        .join(cp_te_full_lf, on="counterparty_id", how="left")
        .join(mcc_te_full_lf, on="mcc_code", how="left")
        .join(ch_te_full_lf, on="channel", how="left")
        .with_columns(
            pl.col("te_cp").fill_null(prior),
            pl.col("te_mcc").fill_null(prior),
            pl.col("te_ch").fill_null(prior),
        )
    )
    test_parts.append(agg_account_stats(test_lf, "te_cp", "te_mcc", "te_ch"))

    gc.collect()

print("Finalizing train/test TE features...")
train_te = finalize_account_features(pl.concat(train_parts), prefix="te")
test_te = finalize_account_features(pl.concat(test_parts), prefix="te")

all_ids = pl.concat([labels.select("account_id"), test_ids]).unique()
out = (
    all_ids
    .join(train_te, on="account_id", how="left")
    .join(test_te, on="account_id", how="left", suffix="_test")
    .with_columns(
        # Use train OOF values when present; else test values for test accounts.
        *[
            pl.coalesce(pl.col(c), pl.col(f"{c}_test")).alias(c)
            for c in [
                "te_cp_mean", "te_cp_std", "te_cp_max", "te_mcc_mean",
                "te_channel_mean", "te_cp_amt_weighted", "te_high_risk_ratio"
            ]
        ]
    )
    .select([
        "account_id",
        "te_cp_mean", "te_cp_std", "te_cp_max", "te_mcc_mean",
        "te_channel_mean", "te_cp_amt_weighted", "te_high_risk_ratio",
    ])
    .fill_null(prior)
)

print(f"TE feature table: {out.shape}")
out.write_parquet(FEAT / "txn_oof_te_features.parquet")
print(f"✓ {(FEAT / 'txn_oof_te_features.parquet')} saved")
