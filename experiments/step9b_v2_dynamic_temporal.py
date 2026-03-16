"""
Step 9b — Submission with V2 Predictions + Dynamic Temporal Windows
Uses the PROVEN v2 probability predictions (0.979 AUC) and only applies
the improved dynamic temporal windowing on top.

The rank blend (step 8) degraded AUC from 0.979 → 0.880 because v2
already optimized blend weights internally. This script reverts to
v2 predictions and only improves the temporal IoU component.

Inputs:  models/test_preds_v2.npy  (the proven 0.979 predictions)
         models/test_ids_order.parquet
         models/best_threshold.txt
         data/archive/transactions/batch-{1-4}/*.parquet
Outputs: submission.csv
"""
import polars as pl
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta

ROOT = Path(r"C:\Users\ujjaw\Downloads\AML_Mule_Project")
DATA = ROOT / "data" / "archive"
MODELS = ROOT / "models"

# ── Load PROVEN v2 predictions ───────────────────────────────────────
test_ids_order = pl.read_parquet(MODELS / "test_ids_order.parquet")
v2_preds = np.load(MODELS / "test_preds_v2.npy")

with open(MODELS / "best_threshold.txt") as f:
    threshold = float(f.read().strip())

print(f"Test accounts:  {test_ids_order.shape[0]}")
print(f"Predictions:    {len(v2_preds)}")
print(f"Threshold:      {threshold:.4f}")
print(f"Pred stats: mean={v2_preds.mean():.4f}, min={v2_preds.min():.4f}, max={v2_preds.max():.4f}")

test_accounts = test_ids_order.with_columns(pl.Series("pred", v2_preds))

# ── Identify predicted mules ────────────────────────────────────────
predicted_mule_ids = (
    test_accounts
    .filter(pl.col("pred") > threshold)["account_id"]
    .to_list()
)
print(f"Predicted mules (threshold {threshold:.4f}): {len(predicted_mule_ids)}")

# ── Compute daily transaction volumes for predicted mules ────────────
print("\nScanning transactions for temporal windows...")
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

    print(f"\nBuilding dynamic windows for {len(predicted_mule_ids)} accounts...")
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

        # Create complete date range (fill gaps with 0)
        date_range = pd.date_range(start=min(dates), end=max(dates), freq="D")
        vol_series = pd.Series(0.0, index=date_range)
        for dt, v in zip(dates, vols):
            vol_series[dt] = v

        # 7-day rolling average
        rolling_avg = vol_series.rolling(window=7, min_periods=1, center=True).mean()

        # Find peak of rolling average
        peak_idx = rolling_avg.argmax()

        # Baseline = median of daily volume (including zero-days)
        baseline = vol_series.median()
        if baseline <= 0:
            baseline = vol_series.mean()
        if baseline <= 0:
            baseline = 1.0

        all_dates = vol_series.index
        rolling_vals = rolling_avg.values

        # Expand window from peak until rolling avg drops below baseline
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
    print(f"  Mean:   {window_days.mean():.1f}")
    print(f"  Median: {np.median(window_days):.1f}")
    print(f"  Min:    {window_days.min()}")
    print(f"  Max:    {window_days.max()}")
    print(f"  <7d:    {(window_days < 7).sum()} ({(window_days < 7).mean()*100:.1f}%)")
    print(f"  <30d:   {(window_days < 30).sum()} ({(window_days < 30).mean()*100:.1f}%)")
    print(f"  <90d:   {(window_days < 90).sum()} ({(window_days < 90).mean()*100:.1f}%)")

# ── Build submission ─────────────────────────────────────────────────
sub_df = test_accounts.select(["account_id", "pred"]).rename({"pred": "is_mule"})

suspicious_starts = []
suspicious_ends = []
for row in sub_df.iter_rows(named=True):
    acc_id = row["account_id"]
    prob = row["is_mule"]
    if prob > threshold and acc_id in windows:
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
print(submission[submission["is_mule"] > threshold].head(10))
