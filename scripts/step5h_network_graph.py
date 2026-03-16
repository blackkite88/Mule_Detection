"""
Step 4 — Graph Features
Computes per-account graph metrics from the counterparty network:
  in-degree, out-degree, unique counterparties, fan-in/fan-out ratio.
"""
import polars as pl
from pathlib import Path

ROOT = Path(r"C:\Users\ujjaw\Downloads\AML_Mule_Project")
DATA = ROOT / "data" / "archive"
FEAT = ROOT / "features"
FEAT.mkdir(parents=True, exist_ok=True)

target_ids = pl.concat([
    pl.read_parquet(DATA / "train_labels.parquet").select("account_id"),
    pl.read_parquet(DATA / "test_accounts.parquet").select("account_id"),
]).unique()["account_id"]

out_partials = []   # account_id -> counterparty (outgoing = debits)
in_partials  = []   # counterparty -> account_id (incoming = credits)

for batch in ["batch-1", "batch-2", "batch-3", "batch-4"]:
    print(f"Processing transactions/{batch} ...")

    df = (
        pl.scan_parquet(DATA / "transactions" / batch / "*.parquet")
        .filter(
            pl.col("account_id").is_in(target_ids)
            & pl.col("counterparty_id").is_not_null()
        )
        .select(["account_id", "counterparty_id", "txn_type", "amount"])
        .collect()
    )

    # ── Out-degree: debits from this account ─────────────────────────
    out_agg = (
        df.filter(pl.col("txn_type") == "D")
        .group_by("account_id")
        .agg([
            pl.col("counterparty_id").n_unique().alias("out_degree"),
            pl.len().alias("out_txn_count"),
            pl.col("amount").abs().sum().alias("out_volume"),
        ])
    )
    out_partials.append(out_agg)

    # ── In-degree: credits to this account ───────────────────────────
    in_agg = (
        df.filter(pl.col("txn_type") == "C")
        .group_by("account_id")
        .agg([
            pl.col("counterparty_id").n_unique().alias("in_degree"),
            pl.len().alias("in_txn_count"),
            pl.col("amount").abs().sum().alias("in_volume"),
        ])
    )
    in_partials.append(in_agg)

    print(f"  {batch}: {df.shape[0]} edges")

# ── Combine across batches ───────────────────────────────────────────
out_combined = (
    pl.concat(out_partials)
    .group_by("account_id")
    .agg([
        pl.col("out_degree").max(),       # approx unique
        pl.col("out_txn_count").sum(),
        pl.col("out_volume").sum(),
    ])
)

in_combined = (
    pl.concat(in_partials)
    .group_by("account_id")
    .agg([
        pl.col("in_degree").max(),
        pl.col("in_txn_count").sum(),
        pl.col("in_volume").sum(),
    ])
)

# ── Merge in/out into one table ──────────────────────────────────────
graph_features = out_combined.join(in_combined, on="account_id", how="outer")

graph_features = graph_features.with_columns([
    # fan-in/fan-out ratio  (high fan-in + low fan-out = collector pattern)
    (pl.col("in_degree").fill_null(0) / (pl.col("out_degree").fill_null(0) + 1))
        .alias("fan_in_out_ratio"),
    # volume asymmetry
    ((pl.col("in_volume").fill_null(0) - pl.col("out_volume").fill_null(0))
     / (pl.col("in_volume").fill_null(0) + pl.col("out_volume").fill_null(0) + 1e-9))
        .alias("volume_asymmetry"),
    # total unique counterparties
    (pl.col("in_degree").fill_null(0) + pl.col("out_degree").fill_null(0))
        .alias("total_degree"),
])

graph_features.write_parquet(FEAT / "graph_features.parquet")
print(f"\n✓ features/graph_features.parquet saved  ({graph_features.shape})")
