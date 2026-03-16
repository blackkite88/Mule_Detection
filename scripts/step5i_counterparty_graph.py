"""
Step 4b — Graph Label Features (1-hop mule exposure)
For each account, measures how many counterparties are known mules and
what fraction of transaction volume flows to/from mule accounts.
Uses training labels to identify mules.
"""
import polars as pl
import os, gc

BASE = r"C:\Users\ujjaw\Downloads\AML_Mule_Project"
DATA = f"{BASE}/data/archive"
OUT  = f"{BASE}/features"

# ── Load training mule IDs ──────────────────────────────────────────
labels = pl.read_parquet(f"{DATA}/train_labels.parquet")
mule_ids = labels.filter(pl.col("is_mule") == 1)["account_id"].to_list()
print(f"Known mules: {len(mule_ids)}")

# Accumulators (one per batch, aggregated at end)
sent_batches = []
recv_batches = []
pair_batches = []

for b in range(1, 5):
    bdir = f"{DATA}/transactions/batch-{b}"
    parts = sorted(f for f in os.listdir(bdir) if f.endswith(".parquet"))
    print(f"\n[Batch-{b}] {len(parts)} parts")

    ps, pr, pm = [], [], []

    for pf in parts:
        df = pl.read_parquet(
            f"{bdir}/{pf}",
            columns=["account_id", "counterparty_id", "amount"],
        )
        df = df.with_columns(
            pl.col("counterparty_id").is_in(mule_ids).cast(pl.Int8).alias("cparty_is_mule"),
            pl.col("account_id").is_in(mule_ids).cast(pl.Int8).alias("acct_is_mule"),
        )

        # Outgoing (account_id → counterparty_id)
        ps.append(
            df.group_by("account_id").agg(
                pl.col("amount").sum().alias("vol_sent"),
                pl.len().alias("txn_sent"),
                (pl.col("amount") * pl.col("cparty_is_mule")).sum().alias("vol_sent_mule"),
                pl.col("cparty_is_mule").sum().alias("txn_sent_mule"),
            )
        )

        # Incoming (counterparty_id → account_id)
        pr.append(
            df.group_by("counterparty_id").agg(
                pl.col("amount").sum().alias("vol_recv"),
                pl.len().alias("txn_recv"),
                (pl.col("amount") * pl.col("acct_is_mule")).sum().alias("vol_recv_mule"),
                pl.col("acct_is_mule").sum().alias("txn_recv_mule"),
            ).rename({"counterparty_id": "account_id"})
        )

        # Unique (account, mule_counterparty) pairs
        m1 = df.filter(pl.col("cparty_is_mule") == 1).select(
            pl.col("account_id").alias("aid"),
            pl.col("counterparty_id").alias("mid"),
        ).unique()
        m2 = df.filter(pl.col("acct_is_mule") == 1).select(
            pl.col("counterparty_id").alias("aid"),
            pl.col("account_id").alias("mid"),
        ).unique()
        pm.append(pl.concat([m1, m2]))
        del df

    # Aggregate within batch
    sent_batches.append(
        pl.concat(ps).group_by("account_id").agg(
            pl.col("vol_sent").sum(), pl.col("txn_sent").sum(),
            pl.col("vol_sent_mule").sum(), pl.col("txn_sent_mule").sum(),
        )
    )
    recv_batches.append(
        pl.concat(pr).group_by("account_id").agg(
            pl.col("vol_recv").sum(), pl.col("txn_recv").sum(),
            pl.col("vol_recv_mule").sum(), pl.col("txn_recv_mule").sum(),
        )
    )
    pair_batches.append(pl.concat(pm).unique())
    del ps, pr, pm
    gc.collect()
    print(f"  done: sent={sent_batches[-1].shape[0]}, recv={recv_batches[-1].shape[0]}, pairs={pair_batches[-1].shape[0]}")


# ── Final aggregation across batches ─────────────────────────────────
print("\nAggregating across batches...")
sent = pl.concat(sent_batches).group_by("account_id").agg(
    pl.col("vol_sent").sum(), pl.col("txn_sent").sum(),
    pl.col("vol_sent_mule").sum(), pl.col("txn_sent_mule").sum(),
)
recv = pl.concat(recv_batches).group_by("account_id").agg(
    pl.col("vol_recv").sum(), pl.col("txn_recv").sum(),
    pl.col("vol_recv_mule").sum(), pl.col("txn_recv_mule").sum(),
)
mule_pairs = pl.concat(pair_batches).unique()
mule_counts = mule_pairs.group_by("aid").agg(
    pl.col("mid").n_unique().alias("n_mule_counterparties")
)
del sent_batches, recv_batches, pair_batches, mule_pairs
gc.collect()

# ── Combine & derive ratios ──────────────────────────────────────────
feat = (
    sent.join(recv, on="account_id", how="outer_coalesce")
    .join(mule_counts.rename({"aid": "account_id"}), on="account_id", how="left")
    .fill_null(0)
)

feat = feat.with_columns(
    (pl.col("vol_sent_mule") / (pl.col("vol_sent") + 1)).alias("frac_vol_sent_mule"),
    (pl.col("vol_recv_mule") / (pl.col("vol_recv") + 1)).alias("frac_vol_recv_mule"),
    (pl.col("txn_sent_mule") / (pl.col("txn_sent") + 1)).alias("frac_txn_sent_mule"),
    (pl.col("txn_recv_mule") / (pl.col("txn_recv") + 1)).alias("frac_txn_recv_mule"),
    ((pl.col("vol_sent_mule") + pl.col("vol_recv_mule"))
     / (pl.col("vol_sent") + pl.col("vol_recv") + 1)).alias("frac_vol_mule_total"),
    ((pl.col("txn_sent_mule") + pl.col("txn_recv_mule"))
     / (pl.col("txn_sent") + pl.col("txn_recv") + 1)).alias("frac_txn_mule_total"),
)

out = feat.select([
    "account_id",
    "n_mule_counterparties",
    "vol_sent_mule", "vol_recv_mule",
    "txn_sent_mule", "txn_recv_mule",
    "frac_vol_sent_mule", "frac_vol_recv_mule",
    "frac_txn_sent_mule", "frac_txn_recv_mule",
    "frac_vol_mule_total", "frac_txn_mule_total",
])

print(f"\nGraph label features: {out.shape}")
out.write_parquet(f"{OUT}/graph_label_features.parquet")
print(f"✓ {OUT}/graph_label_features.parquet saved")
