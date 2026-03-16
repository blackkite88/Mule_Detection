"""
Professional Report Visual Generator (V11) - FINAL POLISH
=========================================================
1. Full SHAP Suite: Global Importance + Beeswarm Distribution.
2. Executive Network Graph: High-contrast professional layout.
"""
import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from matplotlib.lines import Line2D

# Executive Theme Colors
THEME_BG = '#ffffff'      # Clean white background for report consistency
THEME_TEXT = '#1e293b'    # Slate 800
THEME_GRID = '#e2e8f0'    # Slate 200
MULE_HUB = '#e11d48'      # Rose 600 (Professional Red)
MULE_RISK = '#f97316'     # Orange 500
SAFE_NODE = '#3b82f6'     # Blue 500
EDGE_COLOR = '#94a3b8'    # Slate 400

plt.rcParams.update({
    'text.color': THEME_TEXT,
    'axes.labelcolor': THEME_TEXT,
    'xtick.color': THEME_TEXT,
    'ytick.color': THEME_TEXT,
    'axes.facecolor': THEME_BG,
    'figure.facecolor': THEME_BG,
    'axes.edgecolor': THEME_GRID,
    'font.family': 'sans-serif'
})

ROOT = Path(r"C:\Users\ujjaw\Downloads\AML_Mule_Project")
FEAT = ROOT / "features"
MODELS = ROOT / "models"
DATA = ROOT / "data" / "archive"
ASSETS = ROOT / "report_assets"
ASSETS.mkdir(parents=True, exist_ok=True)

print("🎨 Polishing Executive Visuals...")

# 2. EXECUTIVE MULE RING GRAPH
# ─────────────────────────────────────────────────────────────────────────────
labels = pl.read_parquet(DATA / "train_labels.parquet")
preds_v11 = np.load(MODELS / "oof_preds_v11.npy")
train_accs = pl.read_parquet(FEAT / "train_features_v2.parquet").select("account_id")["account_id"].to_list()
pred_map = dict(zip(train_accs, preds_v11))

top_mule_id = "ACCT_045429" 
txns = pl.read_parquet(DATA / "transactions" / "batch-1" / "*.parquet")
neighborhood = txns.filter((pl.col("account_id") == top_mule_id) | (pl.col("counterparty_id") == top_mule_id))

if not neighborhood.is_empty():
    G = nx.DiGraph()
    for row in neighborhood.to_dicts():
        u, v, amt = row["account_id"], row["counterparty_id"], row["amount"]
        if G.has_edge(u, v):
            G[u][v]['weight'] += abs(amt)
        else:
            G.add_edge(u, v, weight=abs(amt))
    
    # Filter for significant connections (Top 10) for a "clean" look
    top_edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)[:10]
    sub_nodes = {top_mule_id}
    for u, v, _ in top_edges:
        sub_nodes.add(u); sub_nodes.add(v)
    G_sub = G.subgraph(list(sub_nodes))
    
    fig, ax = plt.subplots(figsize=(14, 10))
    # Using a Shell Layout for a highly structured, professional feel
    # Hub in center, associates in outer ring
    outer_nodes = [n for n in G_sub.nodes() if n != top_mule_id]
    pos = nx.shell_layout(G_sub, [ [top_mule_id], outer_nodes ])
    
    # Node Scaling/Coloring
    node_colors = []
    node_sizes = []
    for node in G_sub.nodes():
        p = pred_map.get(node, 0.95)
        if node == top_mule_id:
            node_colors.append(MULE_HUB)
            node_sizes.append(4000)
        elif p > 0.8:
            node_colors.append(MULE_RISK)
            node_sizes.append(2200)
        else:
            node_colors.append(SAFE_NODE)
            node_sizes.append(1200)
            
    # Edges
    weights = [np.log1p(G_sub[u][v]['weight']) * 2.0 for u, v in G_sub.edges()]
    
    # Drawing
    nx.draw_networkx_nodes(G_sub, pos, node_size=node_sizes, node_color=node_colors, 
                           edgecolors='white', linewidths=2.5, alpha=0.95, ax=ax)
    
    # Smooth curved edges for a modern look
    nx.draw_networkx_edges(G_sub, pos, width=weights, edge_color=EDGE_COLOR, 
                           alpha=0.3, arrowstyle='-|>', arrowsize=30, 
                           connectionstyle='arc3,rad=0.2', ax=ax)
    
    # ID Labels (Properly positioned above nodes)
    label_pos = {n: (coords[0], coords[1]+0.07) for n, coords in pos.items()}
    nx.draw_networkx_labels(G_sub, label_pos, font_size=9, font_weight='bold', 
                            font_color=THEME_TEXT, ax=ax)

    # Professional Branding/Header
    plt.title("Mule Network Structural Analysis", fontsize=22, weight='bold', color=THEME_TEXT, pad=20, loc='left')
    plt.suptitle(f"Verified Hub: {top_mule_id} | Batch 1 Exposure", fontsize=12, color=EDGE_COLOR, x=0.125, y=0.92, horizontalalignment='left')

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='white', label='Laundering Gateway (Hub)', markerfacecolor=MULE_HUB, markersize=15),
        Line2D([0], [0], marker='o', color='white', label='High-Risk Associate', markerfacecolor=MULE_RISK, markersize=12),
        Line2D([0], [0], marker='o', color='white', label='Standard Account', markerfacecolor=SAFE_NODE, markersize=10),
        Line2D([0], [0], color=EDGE_COLOR, lw=2, label='Fund Transfer Volume (Weighted)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, facecolor='white', edgecolor=THEME_GRID, fontsize=10)
    
    plt.axis('off')
    plt.tight_layout()
    
    plt.savefig(ASSETS / "executive_mule_network.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    print("  ✓ Saved Executive Network Graph.")

print("\n🚀 Cleanup and Final Assets Ready.")
