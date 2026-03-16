# Vanguard AML System — Model Architecture

## End-to-End Pipeline

```mermaid
flowchart TB
    subgraph DATA["📦 Data Ingestion Layer"]
        direction LR
        D1["transactions/<br/>4 Batches × Parquet"]
        D2["transactions_additional/<br/>4 Batches × Parquet"]
        D3["customers.parquet<br/>accounts.parquet<br/>demographics.parquet"]
        D4["train_labels.parquet"]
    end

    subgraph STREAM["⚡ Polars Streaming Engine"]
        direction LR
        S1["scan_parquet()<br/>Lazy Evaluation"]
        S2["Projection Pushdown<br/>Column Pruning"]
        S3["collect(streaming=True)<br/>Out-of-Core Execution"]
    end

    subgraph FEATURES["🔬 Feature Engineering (5 Modules)"]
        direction TB
        subgraph F_STATIC["Module A: Static Profile"]
            FA1["account_age_days"]
            FA2["customer_age_days"]
            FA3["product_type"]
        end
        subgraph F_TXN["Module B: Transaction Physics"]
            FB1["pass_through_ratio"]
            FB2["rpte_60m"]
            FB3["turnover_ratio"]
        end
        subgraph F_GRAPH["Module C: Network Topology"]
            FC1["pagerank"]
            FC2["in_degree / out_degree"]
            FC3["clustering_coefficient"]
        end
        subgraph F_TEMPORAL["Module D: Temporal Burstiness"]
            FD1["burst_count_5min"]
            FD2["rolling_7d_zscore"]
            FD3["peak_day_concentration"]
        end
        subgraph F_CONCENTRATION["Module E: Concentration"]
            FE1["cp_entropy"]
            FE2["n_unique_counterparties"]
            FE3["txn_spike_ratio"]
        end
    end

    subgraph DEFENSE["🛡️ Defensive Preprocessing"]
        direction LR
        DEF1["Trap 1-6 Purge<br/>Drop: gender, age,<br/>branch_stats, currency"]
        DEF2["Trap 7 Surgery<br/>496 Ghost Mules<br/>sample_weight = 0.2"]
        DEF3["Structural Clipping<br/>max_depth = 5<br/>num_leaves = 31"]
    end

    subgraph ENSEMBLE["🧠 Dual-Model Ensemble"]
        direction LR
        M1["LightGBM<br/>n_est=500, lr=0.02<br/>5-Fold Stratified CV"]
        M2["XGBoost<br/>n_est=500, lr=0.02<br/>5-Fold Stratified CV"]
    end

    subgraph CALIBRATION["🎯 Calibration & Output"]
        direction TB
        C1["Rank-Based Blending<br/>rank_avg = (rank_lgb + rank_xgb) / 2"]
        C2["Prior-Based Threshold<br/>Anchor to 2.79% Mule Prior"]
        C3["Lifecycle Bounding<br/>first_burst → last_burst"]
        OUT["submission.csv<br/>account_id, is_mule,<br/>start_date, end_date"]
    end

    DATA --> STREAM
    STREAM --> FEATURES
    FEATURES --> DEFENSE
    DEFENSE --> ENSEMBLE
    M1 --> C1
    M2 --> C1
    C1 --> C2
    C2 --> C3
    C3 --> OUT

    style DATA fill:#1e3a5f,stroke:#60a5fa,color:#e0f2fe
    style STREAM fill:#1e3a5f,stroke:#60a5fa,color:#e0f2fe
    style FEATURES fill:#1a2e1a,stroke:#4ade80,color:#dcfce7
    style DEFENSE fill:#3b1010,stroke:#f87171,color:#fee2e2
    style ENSEMBLE fill:#2d1b4e,stroke:#a78bfa,color:#ede9fe
    style CALIBRATION fill:#1a2e1a,stroke:#4ade80,color:#dcfce7
```

---

## Component Specifications

### Layer 1 — Data Ingestion
| Component | Specification |
|---|---|
| **Total Volume** | 16.2 GB / 400M+ rows |
| **Format** | Apache Parquet (columnar) |
| **Sources** | 8 transaction batches + 4 static tables + labels |

### Layer 2 — Streaming Engine
| Component | Specification |
|---|---|
| **Framework** | Polars 0.20+ (Rust-backed) |
| **Execution** | LazyFrame → Streaming Collect |
| **Memory Reduction** | ~60% via dtype optimization (Float32/Int32) |

### Layer 3 — Feature Engineering (151 Features)
| Module | Count | Key Signals |
|---|---|---|
| **A: Static Profile** | 12 | Account age, product type, customer tenure |
| **B: Transaction Physics** | 38 | Pass-through ratio, RPTE, turnover |
| **C: Network Topology** | 22 | PageRank, degree ratios, clustering |
| **D: Temporal Burstiness** | 41 | Rolling Z-scores, burst counts, weekend ratios |
| **E: Concentration** | 38 | CP entropy, spike ratios, counterparty diversity |

### Layer 4 — Defensive Preprocessing
| Defense | Target | Mechanism |
|---|---|---|
| **Demographic Purge** | Traps 1–6 | Hard feature exclusion |
| **Ghost Mule Surgery** | Trap 7 (496 noisy labels) | `sample_weight = 0.2` soft-weighting |
| **Structural Clipping** | Overfitting prevention | `max_depth=5`, `num_leaves=31` |

### Layer 5 — Ensemble Architecture
| Model | Hyperparameters | CV Strategy |
|---|---|---|
| **LightGBM** | `n_estimators=500`, `lr=0.02`, `max_depth=5` | 5-Fold Stratified |
| **XGBoost** | `n_estimators=500`, `lr=0.02`, `max_depth=5` | 5-Fold Stratified |
| **Blending** | Rank-average of OOF predictions | Neutralizes probability scale mismatch |

### Layer 6 — Calibrated Output
| Step | Method | Impact |
|---|---|---|
| **Rank Blend** | [(rank_lgb + rank_xgb) / 2](file:///C:/Users/ujjaw/anaconda3/Lib/site-packages/shap/explainers/_tree.py#1904-1909) | Stable generalization |
| **Threshold** | Anchored to 2.79% prior | Maximized F1 on imbalanced data |
| **Temporal IoU** | `first_burst → last_burst` bounding | +0.26 IoU improvement |

---

## Final Metrics (V11 Defensive)

| Metric | Score |
|---|---|
| **AUC-ROC** | **0.9947** |
| **F1-Score** | **0.8612** |
| **Temporal IoU** | **0.6637** |
| **Trap Avoidance (RH 1–6)** | **> 0.95** |
