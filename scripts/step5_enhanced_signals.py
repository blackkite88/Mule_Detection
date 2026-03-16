"""
Step 2e — Enhanced Features (txn_type TE, frequency encoding, IP sharing, sub-type TE)
Builds multiple high-value feature families in one script to push AUC higher.

Feature families:
A) OOF TE on txn_type, mcc_code x txn_type, counterparty_id x txn_type
B) Counterparty frequency encoding (how common each counterparty is)
C) IP-based mule exposure (shared IPs with known mules) from transactions_additional
D) Transaction sub-type features from transactions_additional
"""
from pathlib import Path
import gc, os
import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedKFold

ROOT = Path(r"C:\Users\ujjaw\Downloads\AML_Mule_Project")
DATA = ROOT / "data" / "archive"
FEAT = ROOT / "features"

N_FOLDS = 5
SEED = 42

# ── Load labels and fold assignments ─────────────────────────────────
labels = pl.read_parquet(DATA / "train_labels.parquet").select(["account_id", "is_mule"])
test_ids = pl.read_parquet(DATA / "test_accounts.parquet").select(["account_id"])
prior = float(labels["is_mule"].mean())

y = labels["is_mule"].to_numpy()
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = np.zeros(len(labels), dtype=np.int8)
for f, (_, va_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
    folds[va_idx] = f
fold_map = labels.with_columns(pl.Series("fold", folds))
fold_map_lf = fold_map.lazy()
all_ids = pl.concat([labels.select("account_id"), test_ids]).unique()

print(f"Train: {labels.shape[0]}  Test: {test_ids.shape[0]}  Prior: {prior:.5f}")


# ═════════════════════════════════════════════════════════════════════
# FAMILY A: OOF Target Encoding on txn_type and interactions
# ═════════════════════════════════════════════════════════════════════
def smoothed_te(target_sum, target_cnt, prior, alpha):
    return (target_sum + alpha * prior) / (target_cnt + alpha)


def build_te_family(key_cols, alpha, tag):
    """Build OOF TE for given key columns from transaction data."""
    print(f"\n[TE-{tag}] Building target encoding for {key_cols}...")

    # Pass A: collect stats from labeled accounts
    global_parts, fold_parts = [], []
    for b in range(1, 5):
        bdir = DATA / "transactions" / f"batch-{b}"
        dfl = (
            pl.scan_parquet(str(bdir / "*.parquet"))
            .select(["account_id"] + key_cols)
            .join(fold_map_lf, on="account_id", how="inner")
        )
        global_parts.append(
            dfl.group_by(key_cols).agg(
                pl.col("is_mule").sum().alias("target_sum"),
                pl.len().alias("target_cnt"),
            ).collect()
        )
        fold_parts.append(
            dfl.group_by(["fold"] + key_cols).agg(
                pl.col("is_mule").sum().alias("target_sum"),
                pl.len().alias("target_cnt"),
            ).collect()
        )
        gc.collect()

    gstats = (
        pl.concat(global_parts).group_by(key_cols)
        .agg(pl.col("target_sum").sum(), pl.col("target_cnt").sum())
    )
    fstats = (
        pl.concat(fold_parts).group_by(["fold"] + key_cols)
        .agg(pl.col("target_sum").sum(), pl.col("target_cnt").sum())
    )

    # Build fold-specific TE lookup
    fold_te_parts = []
    for f in range(N_FOLDS):
        fs = fstats.filter(pl.col("fold") == f).select(
            key_cols + [
                pl.col("target_sum").alias("fold_sum"),
                pl.col("target_cnt").alias("fold_cnt"),
            ]
        )
        te_f = (
            gstats.join(fs, on=key_cols, how="left").fill_null(0)
            .with_columns(
                smoothed_te(
                    pl.col("target_sum") - pl.col("fold_sum"),
                    pl.col("target_cnt") - pl.col("fold_cnt"),
                    prior, alpha
                ).alias(f"te_{tag}")
            )
            .select(key_cols + [f"te_{tag}"])
            .with_columns(pl.lit(f).alias("fold"))
        )
        fold_te_parts.append(te_f)
    fold_te = pl.concat(fold_te_parts).lazy()

    # Full TE for test
    full_te = gstats.with_columns(
        smoothed_te(pl.col("target_sum"), pl.col("target_cnt"), prior, alpha).alias(f"te_{tag}")
    ).select(key_cols + [f"te_{tag}"]).lazy()

    # Pass B: apply TE and aggregate per account
    te_col = f"te_{tag}"
    train_parts, test_parts = [], []
    for b in range(1, 5):
        bdir = DATA / "transactions" / f"batch-{b}"
        print(f"  [TE-{tag}] batch-{b}...")
        lf = pl.scan_parquet(str(bdir / "*.parquet")).select(["account_id", "amount"] + key_cols)

        tr = (
            lf.join(fold_map_lf.select(["account_id", "fold"]), on="account_id", how="inner")
            .join(fold_te, on=["fold"] + key_cols, how="left")
            .with_columns(pl.col(te_col).fill_null(prior))
            .group_by("account_id").agg(
                pl.col(te_col).mean().alias(f"{te_col}_mean"),
                pl.col(te_col).max().alias(f"{te_col}_max"),
                pl.col(te_col).std().alias(f"{te_col}_std"),
                (pl.col("amount") * pl.col(te_col)).sum().alias(f"{te_col}_amt_sum"),
                pl.col("amount").sum().alias("_amt_sum"),
            ).collect()
        )
        train_parts.append(tr)

        te = (
            lf.join(test_ids.lazy(), on="account_id", how="inner")
            .join(full_te, on=key_cols, how="left")
            .with_columns(pl.col(te_col).fill_null(prior))
            .group_by("account_id").agg(
                pl.col(te_col).mean().alias(f"{te_col}_mean"),
                pl.col(te_col).max().alias(f"{te_col}_max"),
                pl.col(te_col).std().alias(f"{te_col}_std"),
                (pl.col("amount") * pl.col(te_col)).sum().alias(f"{te_col}_amt_sum"),
                pl.col("amount").sum().alias("_amt_sum"),
            ).collect()
        )
        test_parts.append(te)
        gc.collect()

    # Finalize: aggregate across batches
    def finalize_te(parts):
        combined = pl.concat(parts)
        return (
            combined.group_by("account_id").agg(
                pl.col(f"{te_col}_mean").mean(),
                pl.col(f"{te_col}_max").max(),
                pl.col(f"{te_col}_std").mean(),
                pl.col(f"{te_col}_amt_sum").sum(),
                pl.col("_amt_sum").sum(),
            ).with_columns(
                (pl.col(f"{te_col}_amt_sum") / (pl.col("_amt_sum") + 1e-9)).alias(f"{te_col}_amtw")
            ).drop([f"{te_col}_amt_sum", "_amt_sum"])
        )

    train_te = finalize_te(train_parts)
    test_te = finalize_te(test_parts)

    out_cols = [f"{te_col}_mean", f"{te_col}_max", f"{te_col}_std", f"{te_col}_amtw"]

    out = (
        all_ids
        .join(train_te, on="account_id", how="left")
        .join(test_te, on="account_id", how="left", suffix="_t")
        .with_columns(
            *[pl.coalesce(pl.col(c), pl.col(f"{c}_t")).alias(c) for c in out_cols]
        )
        .select(["account_id"] + out_cols)
        .fill_null(prior)
    )
    print(f"  [TE-{tag}] done: {out.shape}")
    return out


# Build TE features for multiple key combinations
te_txntype = build_te_family(["txn_type"], alpha=20.0, tag="txntype")
te_mcc_txn = build_te_family(["mcc_code", "txn_type"], alpha=40.0, tag="mcc_txn")
te_cp_txn = build_te_family(["counterparty_id", "txn_type"], alpha=50.0, tag="cp_txn")

te_combined = (
    te_txntype
    .join(te_mcc_txn, on="account_id", how="outer_coalesce")
    .join(te_cp_txn, on="account_id", how="outer_coalesce")
    .fill_null(prior)
)
print(f"\nTE combined: {te_combined.shape}")


# ═════════════════════════════════════════════════════════════════════
# FAMILY B: Counterparty Frequency Encoding
# ═════════════════════════════════════════════════════════════════════
print("\n[FreqEnc] Building counterparty frequency features...")
cp_freq_parts = []
for b in range(1, 5):
    bdir = DATA / "transactions" / f"batch-{b}"
    cp_freq_parts.append(
        pl.scan_parquet(str(bdir / "*.parquet"))
        .select(["counterparty_id", "account_id"])
        .group_by("counterparty_id")
        .agg(
            pl.col("account_id").n_unique().alias("cp_n_accounts"),
            pl.len().alias("cp_total_txns"),
        ).collect()
    )
    gc.collect()

cp_freq = (
    pl.concat(cp_freq_parts)
    .group_by("counterparty_id")
    .agg(pl.col("cp_n_accounts").max(), pl.col("cp_total_txns").sum())
)
print(f"  Unique counterparties: {cp_freq.shape[0]}")

# For each account: stats about their counterparties' popularity
freq_train_parts, freq_test_parts = [], []
for b in range(1, 5):
    bdir = DATA / "transactions" / f"batch-{b}"
    print(f"  [FreqEnc] batch-{b}...")
    lf = pl.scan_parquet(str(bdir / "*.parquet")).select(["account_id", "counterparty_id"])

    joined = lf.join(cp_freq.lazy(), on="counterparty_id", how="left")

    for ids, parts in [(fold_map_lf.select("account_id"), freq_train_parts),
                       (test_ids.lazy(), freq_test_parts)]:
        agg = (
            joined.join(ids, on="account_id", how="inner")
            .group_by("account_id").agg(
                pl.col("cp_n_accounts").mean().alias("cp_freq_mean"),
                pl.col("cp_n_accounts").max().alias("cp_freq_max"),
                pl.col("cp_n_accounts").min().alias("cp_freq_min"),
                pl.col("cp_n_accounts").std().alias("cp_freq_std"),
                (pl.col("cp_n_accounts") == 1).cast(pl.Int32).sum().alias("cp_exclusive_cnt"),
            ).collect()
        )
        parts.append(agg)
    gc.collect()

def finalize_freq(parts):
    return (
        pl.concat(parts).group_by("account_id").agg(
            pl.col("cp_freq_mean").mean(),
            pl.col("cp_freq_max").max(),
            pl.col("cp_freq_min").min(),
            pl.col("cp_freq_std").mean(),
            pl.col("cp_exclusive_cnt").sum(),
        )
    )

freq_out = (
    all_ids
    .join(finalize_freq(freq_train_parts), on="account_id", how="left")
    .join(finalize_freq(freq_test_parts), on="account_id", how="left", suffix="_t")
    .with_columns(
        *[pl.coalesce(pl.col(c), pl.col(f"{c}_t")).alias(c)
          for c in ["cp_freq_mean", "cp_freq_max", "cp_freq_min", "cp_freq_std", "cp_exclusive_cnt"]]
    )
    .select(["account_id", "cp_freq_mean", "cp_freq_max", "cp_freq_min", "cp_freq_std", "cp_exclusive_cnt"])
    .fill_null(0)
)
print(f"  Freq features: {freq_out.shape}")


# ═════════════════════════════════════════════════════════════════════
# FAMILY C: IP-based mule exposure from transactions_additional
# ═════════════════════════════════════════════════════════════════════
print("\n[IP] Building IP-sharing features...")
mule_ids = set(labels.filter(pl.col("is_mule") == 1)["account_id"].to_list())

# Step 1: Find IPs used by mules
mule_ips_parts = []
for b in range(1, 5):
    bdir = DATA / "transactions_additional" / f"batch-{b}"
    if not bdir.exists():
        continue
    # transactions_additional links via transaction_id; we need account_id
    # Join with main transactions to get account_id
    txn_bdir = DATA / "transactions" / f"batch-{b}"
    parts = sorted(f for f in os.listdir(str(bdir)) if f.endswith(".parquet"))
    print(f"  [IP] batch-{b}: {len(parts)} parts...")

    for pf in parts:
        try:
            add = pl.read_parquet(str(bdir / pf), columns=["transaction_id", "ip_address"])
            main = pl.read_parquet(str(txn_bdir / pf), columns=["transaction_id", "account_id"])
            merged = add.join(main, on="transaction_id", how="inner").drop_nulls("ip_address")
            mule_rows = merged.filter(pl.col("account_id").is_in(list(mule_ids)))
            if mule_rows.shape[0] > 0:
                mule_ips_parts.append(mule_rows.select(["ip_address", "account_id"]).unique())
        except Exception:
            pass
    gc.collect()

if mule_ips_parts:
    mule_ip_set = set(pl.concat(mule_ips_parts)["ip_address"].unique().to_list())
    print(f"  Mule IPs found: {len(mule_ip_set)}")

    # Step 2: For each account, count txns from mule IPs
    ip_train_parts, ip_test_parts = [], []
    for b in range(1, 5):
        bdir = DATA / "transactions_additional" / f"batch-{b}"
        txn_bdir = DATA / "transactions" / f"batch-{b}"
        parts = sorted(f for f in os.listdir(str(bdir)) if f.endswith(".parquet"))
        print(f"  [IP-agg] batch-{b}...")

        batch_rows = []
        for pf in parts:
            try:
                add = pl.read_parquet(str(bdir / pf), columns=["transaction_id", "ip_address"])
                main = pl.read_parquet(str(txn_bdir / pf), columns=["transaction_id", "account_id"])
                merged = add.join(main, on="transaction_id", how="inner").drop_nulls("ip_address")
                merged = merged.with_columns(
                    pl.col("ip_address").is_in(list(mule_ip_set)).cast(pl.Int8).alias("is_mule_ip")
                )
                batch_rows.append(
                    merged.group_by("account_id").agg(
                        pl.col("is_mule_ip").sum().alias("mule_ip_txns"),
                        pl.len().alias("total_ip_txns"),
                    )
                )
            except Exception:
                pass

        if batch_rows:
            batch_agg = pl.concat(batch_rows).group_by("account_id").agg(
                pl.col("mule_ip_txns").sum(), pl.col("total_ip_txns").sum()
            )
            ip_train_parts.append(batch_agg)
            ip_test_parts.append(batch_agg)
        gc.collect()

    ip_agg = (
        pl.concat(ip_train_parts).group_by("account_id").agg(
            pl.col("mule_ip_txns").sum(), pl.col("total_ip_txns").sum()
        ).with_columns(
            (pl.col("mule_ip_txns") / (pl.col("total_ip_txns") + 1)).alias("mule_ip_ratio")
        )
    )
    ip_out = (
        all_ids.join(ip_agg, on="account_id", how="left")
        .fill_null(0)
        .select(["account_id", "mule_ip_txns", "mule_ip_ratio"])
    )
    print(f"  IP features: {ip_out.shape}")
else:
    ip_out = all_ids.with_columns(
        pl.lit(0).alias("mule_ip_txns"),
        pl.lit(0.0).alias("mule_ip_ratio"),
    )
    print("  No mule IPs found, using zeros")


# ═════════════════════════════════════════════════════════════════════
# FAMILY D: Transaction sub-type features from transactions_additional
# ═════════════════════════════════════════════════════════════════════
print("\n[SubType] Building transaction sub-type features...")
subtype_parts = []
for b in range(1, 5):
    bdir = DATA / "transactions_additional" / f"batch-{b}"
    txn_bdir = DATA / "transactions" / f"batch-{b}"
    parts = sorted(f for f in os.listdir(str(bdir)) if f.endswith(".parquet"))
    print(f"  [SubType] batch-{b}: {len(parts)} parts...")

    batch_rows = []
    for pf in parts:
        try:
            add = pl.read_parquet(str(bdir / pf),
                columns=["transaction_id", "transaction_sub_type", "part_transaction_type"])
            main = pl.read_parquet(str(txn_bdir / pf), columns=["transaction_id", "account_id"])
            merged = add.join(main, on="transaction_id", how="inner")
            batch_rows.append(
                merged.group_by("account_id").agg(
                    pl.len().alias("n_addl_txns"),
                    pl.col("transaction_sub_type").n_unique().alias("n_sub_types"),
                    pl.col("part_transaction_type").n_unique().alias("n_part_types"),
                    (pl.col("transaction_sub_type") == "normal").cast(pl.Int32).sum().alias("normal_txn_cnt"),
                    (pl.col("transaction_sub_type") != "normal").cast(pl.Int32).sum().alias("special_txn_cnt"),
                )
            )
        except Exception:
            pass

    if batch_rows:
        subtype_parts.append(
            pl.concat(batch_rows).group_by("account_id").agg(
                pl.col("n_addl_txns").sum(),
                pl.col("n_sub_types").max(),
                pl.col("n_part_types").max(),
                pl.col("normal_txn_cnt").sum(),
                pl.col("special_txn_cnt").sum(),
            )
        )
    gc.collect()

if subtype_parts:
    subtype_agg = (
        pl.concat(subtype_parts).group_by("account_id").agg(
            pl.col("n_addl_txns").sum(),
            pl.col("n_sub_types").max(),
            pl.col("n_part_types").max(),
            pl.col("normal_txn_cnt").sum(),
            pl.col("special_txn_cnt").sum(),
        ).with_columns(
            (pl.col("special_txn_cnt") / (pl.col("n_addl_txns") + 1)).alias("special_txn_ratio")
        )
    )
    subtype_out = (
        all_ids.join(subtype_agg, on="account_id", how="left")
        .fill_null(0)
        .select(["account_id", "n_sub_types", "n_part_types", "special_txn_cnt", "special_txn_ratio"])
    )
    print(f"  SubType features: {subtype_out.shape}")
else:
    subtype_out = all_ids.with_columns(
        pl.lit(0).alias("n_sub_types"),
        pl.lit(0).alias("n_part_types"),
        pl.lit(0).alias("special_txn_cnt"),
        pl.lit(0.0).alias("special_txn_ratio"),
    )


# ═════════════════════════════════════════════════════════════════════
# SAVE ALL
# ═════════════════════════════════════════════════════════════════════
final = (
    te_combined
    .join(freq_out, on="account_id", how="outer_coalesce")
    .join(ip_out, on="account_id", how="outer_coalesce")
    .join(subtype_out, on="account_id", how="outer_coalesce")
    .fill_null(0)
)

print(f"\nFinal enhanced features: {final.shape}")
final.write_parquet(FEAT / "enhanced_features.parquet")
print(f"✓ {FEAT / 'enhanced_features.parquet'} saved")
