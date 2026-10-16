"""
visualizer.py — Matplotlib/Seaborn plots for the BID-seq report.

All functions return a base64-encoded PNG string for embedding in HTML,
OR write directly to a file when output_path is given.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server/pipeline use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

# Shared style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PALETTE = {
    "HIGH_CONFIDENCE": "#c0392b",
    "MODERATE":        "#e67e22",
    "LOW":             "#3498db",
    "NOISE":           "#bdc3c7",
    "BACKGROUND":      "#95a5a6",
    "KNOWN_PSI":       "#8e44ad",
}
CONFIDENCE_ALPHA = {"HIGH_CONFIDENCE": 0.95, "MODERATE": 0.80, "LOW": 0.60, "NOISE": 0.25}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def _save_or_encode(fig: plt.Figure, output_path: Optional[str | Path]) -> str:
    if output_path:
        fig.savefig(output_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return str(output_path)
    return _fig_to_b64(fig)


# ─── Plot 1: Coverage ─────────────────────────────────────────────────────────

def plot_coverage(
    pileup_df: pd.DataFrame,
    min_coverage: int = 20,
    title: str = "RNU4-2 Read Coverage",
    output_path: Optional[str | Path] = None,
) -> str:
    fig, ax = plt.subplots(figsize=(14, 3.5))

    pos = pileup_df["position"].values
    cov = pileup_df["coverage"].values

    ax.fill_between(pos, cov, alpha=0.45, color="#2980b9", label="Coverage")
    ax.axhline(min_coverage, color="#e74c3c", linewidth=1.2, linestyle="--",
               label=f"Min coverage ({min_coverage}×)")
    ax.set_xlabel("Position in RNU4-2 (1-based nt)", fontsize=11)
    ax.set_ylabel("Read depth", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(1, len(pileup_df))

    # Annotate RENU region
    ax.axvspan(59, 70, alpha=0.12, color="#e74c3c", label="RENU T-loop (pos 60–70)")
    ax.legend(fontsize=9, loc="upper right")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"
    ))
    plt.tight_layout()
    return _save_or_encode(fig, output_path)


# ─── Plot 2: Deletion rate landscape ─────────────────────────────────────────

def plot_deletion_rate(
    stats_df: pd.DataFrame,
    background_rate: float = 0.06,
    title: str = "BID-seq Deletion Rate Across RNU4-2",
    output_path: Optional[str | Path] = None,
) -> str:
    fig, ax = plt.subplots(figsize=(14, 4.5))

    pos      = stats_df["position"].values
    rate     = stats_df["deletion_rate_t"].values
    conf     = stats_df["confidence"].values if "confidence" in stats_df.columns else ["NOISE"] * len(pos)

    # Background bar
    ax.axhline(background_rate, color="#7f8c8d", linewidth=1.0, linestyle=":",
               alpha=0.8, label=f"Nanopore background ({background_rate:.0%})")

    # Base deletion rate line
    ax.plot(pos, rate, color="#bdc3c7", linewidth=0.8, alpha=0.7, zorder=1)

    # Colour-coded scatter for called sites
    for tier in ("HIGH_CONFIDENCE", "MODERATE", "LOW"):
        mask = conf == tier
        if mask.any():
            ax.scatter(
                pos[mask], rate[mask],
                color=PALETTE[tier],
                s=60 if tier == "HIGH_CONFIDENCE" else 40,
                zorder=5,
                alpha=CONFIDENCE_ALPHA[tier],
                label=tier.replace("_", " "),
            )

    # Mark known Ψ sites
    if "is_known_psi" in stats_df.columns:
        known_mask = stats_df["is_known_psi"].values
        ax.scatter(
            pos[known_mask], rate[known_mask],
            marker="D", s=80, color=PALETTE["KNOWN_PSI"], zorder=6,
            edgecolors="black", linewidth=0.5, label="Known Ψ site",
        )

    # RENU region shading
    ax.axvspan(59, 70, alpha=0.10, color="#e74c3c", label="RENU T-loop (pos 60–70)")

    ax.set_xlabel("Position in RNU4-2 (1-based nt)", fontsize=11)
    ax.set_ylabel("Deletion rate", fontsize=11)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(1, max(pos) + 1)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, loc="upper right", ncol=2)
    plt.tight_layout()
    return _save_or_encode(fig, output_path)


# ─── Plot 3: Volcano / fold-change vs. significance ──────────────────────────

def plot_volcano(
    stats_df: pd.DataFrame,
    fdr_threshold: float = 0.05,
    min_fold_change: float = 3.0,
    title: str = "BID-seq Signal: Fold Change vs Significance",
    output_path: Optional[str | Path] = None,
) -> str:
    fig, ax = plt.subplots(figsize=(8, 6))

    df = stats_df[stats_df["coverage_t"] >= 20].copy()
    df["neg_log_q"] = -np.log10(df["q_value"].clip(1e-30, 1.0))

    conf = df["confidence"].values if "confidence" in df.columns else ["NOISE"] * len(df)

    for tier in ("NOISE", "LOW", "MODERATE", "HIGH_CONFIDENCE"):
        mask = conf == tier
        if mask.any():
            ax.scatter(
                np.log2(df.loc[mask, "fold_change"].clip(0.01)),
                df.loc[mask, "neg_log_q"],
                c=PALETTE[tier],
                alpha=CONFIDENCE_ALPHA[tier],
                s=35 if tier != "HIGH_CONFIDENCE" else 70,
                label=tier.replace("_", " "),
                zorder=4 if tier == "HIGH_CONFIDENCE" else 2,
            )

    # Threshold lines
    ax.axhline(-np.log10(fdr_threshold), color="#e74c3c", linewidth=1.0, linestyle="--",
               alpha=0.8, label=f"FDR={fdr_threshold}")
    ax.axvline(np.log2(min_fold_change), color="#3498db", linewidth=1.0, linestyle="--",
               alpha=0.8, label=f"FC={min_fold_change}×")

    # Label top sites
    top = df[df["confidence"].isin(("HIGH_CONFIDENCE", "MODERATE"))].nsmallest(8, "q_value")
    for _, row in top.iterrows():
        ax.annotate(
            str(int(row["position"])),
            xy=(np.log2(max(row["fold_change"], 0.01)), row["neg_log_q"]),
            xytext=(3, 3), textcoords="offset points",
            fontsize=8, color="#2c3e50",
        )

    ax.set_xlabel("log₂(fold change)", fontsize=11)
    ax.set_ylabel("–log₁₀(q-value)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return _save_or_encode(fig, output_path)


# ─── Plot 4: RENU insertion bar chart ────────────────────────────────────────

def plot_renu_summary(
    renu_summary: Dict,
    title: str = "RENU n.64_65insT Insertion Status",
    output_path: Optional[str | Path] = None,
) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: stacked bar of read categories
    ax1 = axes[0]
    categories = ["Wild-type", "RENU T-ins", "Other ins"]
    counts = [
        renu_summary.get("wild_type_reads", 0),
        renu_summary.get("renu_t_insertion_reads", 0),
        renu_summary.get("other_insertion_reads", 0),
    ]
    colors = ["#27ae60", "#c0392b", "#f39c12"]
    bars = ax1.bar(categories, counts, color=colors, edgecolor="white", width=0.5)
    ax1.set_ylabel("Read count", fontsize=11)
    ax1.set_title("Read classification", fontsize=11)
    for bar, count in zip(bars, counts):
        if count > 0:
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.5,
                str(count), ha="center", va="bottom", fontsize=10,
            )

    # Right: VAF gauge (simple pie)
    ax2 = axes[1]
    vaf = renu_summary.get("vaf", 0.0)
    zygosity = renu_summary.get("zygosity", "UNKNOWN")
    pie_vals = [vaf, 1.0 - vaf]
    pie_cols = ["#c0392b", "#ecf0f1"]
    wedges, texts = ax2.pie(
        pie_vals, colors=pie_cols, startangle=90,
        wedgeprops={"width": 0.45, "edgecolor": "white"},
    )
    ax2.text(0, 0, f"{vaf:.1%}", ha="center", va="center",
             fontsize=16, fontweight="bold", color="#c0392b")
    ax2.set_title(f"VAF  |  {zygosity}", fontsize=11)
    ax2.legend(
        wedges,
        [f"RENU insertion ({vaf:.1%})", f"Wild-type ({1-vaf:.1%})"],
        loc="lower center", fontsize=9, bbox_to_anchor=(0.5, -0.18),
    )

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    return _save_or_encode(fig, output_path)


# ─── Plot 5: Per-position heatmap ─────────────────────────────────────────────

def plot_deletion_heatmap(
    stats_df: pd.DataFrame,
    title: str = "Deletion Rate Heatmap (RNU4-2)",
    output_path: Optional[str | Path] = None,
) -> str:
    """Single-row heatmap of deletion rates — compact overview."""
    rates = stats_df.set_index("position")["deletion_rate_t"].values.reshape(1, -1)

    fig, ax = plt.subplots(figsize=(16, 1.8))
    im = ax.imshow(rates, aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.8)
    ax.set_yticks([])
    ax.set_xlabel("Position in RNU4-2 (1-based nt)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")

    n_pos = rates.shape[1]
    tick_step = max(1, n_pos // 20)
    ax.set_xticks(range(0, n_pos, tick_step))
    ax.set_xticklabels(range(1, n_pos + 1, tick_step), fontsize=8)

    plt.colorbar(im, ax=ax, orientation="horizontal",
                 pad=0.35, fraction=0.03, label="Deletion rate")
    plt.tight_layout()
    return _save_or_encode(fig, output_path)


# ─── Plot 6: Known Ψ recovery ────────────────────────────────────────────────

def plot_known_psi_recovery(
    stats_df: pd.DataFrame,
    known_psi_sites: set,
    title: str = "Known Pseudouridine Site Recovery",
    output_path: Optional[str | Path] = None,
) -> str:
    """Bar chart showing deletion rate at known Ψ positions."""
    known_list = sorted(known_psi_sites)
    rates   = []
    confs   = []
    for p in known_list:
        row = stats_df[stats_df["position"] == p]
        if row.empty:
            rates.append(0.0)
            confs.append("NOISE")
        else:
            rates.append(float(row.iloc[0]["deletion_rate_t"]))
            confs.append(str(row.iloc[0].get("confidence", "NOISE")))

    colors = [PALETTE.get(c, "#bdc3c7") for c in confs]

    fig, ax = plt.subplots(figsize=(max(6, len(known_list) * 1.5), 4))
    bars = ax.bar([str(p) for p in known_list], rates, color=colors, edgecolor="white")
    ax.set_xlabel("Known Ψ position (1-based nt)", fontsize=11)
    ax.set_ylabel("Deletion rate (treated)", fontsize=11)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title(title, fontsize=13, fontweight="bold")

    patches = [
        mpatches.Patch(color=PALETTE[t], label=t.replace("_", " "))
        for t in ("HIGH_CONFIDENCE", "MODERATE", "LOW", "NOISE")
    ]
    ax.legend(handles=patches, fontsize=9)
    plt.tight_layout()
    return _save_or_encode(fig, output_path)
