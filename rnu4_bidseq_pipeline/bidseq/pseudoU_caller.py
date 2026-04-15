"""
pseudoU_caller.py — Call pseudouridine (Ψ) sites from the deletion statistics.

This module sits on top of deletion_stats.py and translates the statistical
results into a clean, biology-facing table of Ψ calls.

Output
------
A DataFrame with one row per CALLED Ψ site (only non-NOISE calls), columns:

  position          1-based position in RNU4-2
  ref_base          reference nucleotide (must be T/U for a valid call)
  deletion_rate     treated sample deletion rate (the BID-seq signal)
  fold_change       deletion_rate / background
  q_value           BH-corrected p-value
  confidence        HIGH_CONFIDENCE | MODERATE | LOW
  is_known_psi      from literature annotation
  novel             True if not in the literature list
  homopolymer_flag  True if inside a homopolymer run (lower reliability)
  in_renu_region    True if position ± 5 nt from the RENU hotspot
  interpretation    Human-readable interpretation string
  call_reason       Why this site was (or was not) called

A separate "all_positions" table captures every position with noise
classification for complete transparency.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from bidseq.deletion_stats import run_deletion_analysis


# ─── Confidence ordering ──────────────────────────────────────────────────────

CONFIDENCE_ORDER = {
    "HIGH_CONFIDENCE": 0,
    "MODERATE":        1,
    "LOW":             2,
    "NOISE":           3,
}


# ─── Main caller ─────────────────────────────────────────────────────────────

def call_pseudoU_sites(
    stats_df: pd.DataFrame,
    known_psi_sites: Optional[set] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Translate deletion statistics into Ψ calls.

    Parameters
    ----------
    stats_df:        Output of deletion_stats.run_deletion_analysis().
    known_psi_sites: Set of 1-based positions known from literature.

    Returns
    -------
    dict with keys:
        "calls"       → DataFrame of non-NOISE positions (the actionable output)
        "all"         → Full position table (includes NOISE)
        "summary"     → Dict with call counts, fraction modified, etc.
    """
    from bidseq.reference import KNOWN_PSI_SITES
    known = known_psi_sites or KNOWN_PSI_SITES

    df = stats_df.copy()

    # ── Annotate novel vs known ───────────────────────────────────────────────
    df["is_known_psi"] = df["position"].isin(known)
    df["novel"]        = ~df["is_known_psi"] & (df["confidence"] != "NOISE")

    # ── Build human-readable interpretation ──────────────────────────────────
    df["interpretation"] = df.apply(_interpret_row, axis=1)
    df["call_reason"]    = df.apply(_call_reason, axis=1)

    # ── Split into calls vs noise ─────────────────────────────────────────────
    calls_df = (
        df[df["confidence"] != "NOISE"]
        .sort_values(["confidence", "q_value"], key=lambda s: s.map(CONFIDENCE_ORDER) if s.name == "confidence" else s)
        .reset_index(drop=True)
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = _build_summary(calls_df, df, known)
    _log_calls(calls_df, summary)

    return {
        "calls": calls_df,
        "all":   df,
        "summary": summary,
    }


# ─── Row-level helpers ────────────────────────────────────────────────────────

def _interpret_row(row: pd.Series) -> str:
    """Generate a concise biological interpretation for a position."""
    conf = row.get("confidence", "NOISE")

    if conf == "NOISE":
        rate = row.get("deletion_rate_t", 0)
        if row.get("coverage_t", 0) < 20:
            return "Insufficient coverage — cannot assess."
        if rate < 0.06:
            return "Deletion rate consistent with nanopore background noise."
        return f"Deletion rate {rate:.1%} — not significantly above background (q={row.get('q_value', 1.0):.3f})."

    rate  = row.get("deletion_rate_t", 0)
    fc    = row.get("fold_change", 0)
    q     = row.get("q_value", 1.0)
    known = row.get("is_known_psi", False)
    novel = row.get("novel", False)
    hp    = row.get("homopolymer", False)
    renu  = row.get("in_renu_region", False)

    parts = [
        f"Pseudouridine detected ({conf}).",
        f"Deletion rate {rate:.1%}  ({fc:.1f}× above background;  q={q:.4f}).",
    ]
    if known:
        parts.append("Position matches a known Ψ site in human U4 snRNA.")
    if novel:
        parts.append("Novel Ψ site not previously reported in literature.")
    if hp:
        parts.append("⚠ In homopolymer context — interpret with caution.")
    if renu:
        parts.append("📍 This position is adjacent to the RENU syndrome insertion hotspot.")
    return "  ".join(parts)


def _call_reason(row: pd.Series) -> str:
    """Explain why this position was or was not called."""
    conf  = row.get("confidence", "NOISE")
    cov   = row.get("coverage_t", 0)
    rate  = row.get("deletion_rate_t", 0)
    fc    = row.get("fold_change", 0)
    q     = row.get("q_value", 1.0)
    flags = row.get("flags", "")

    if conf != "NOISE":
        return (
            f"CALLED: q={q:.4f} ≤ threshold, "
            f"rate={rate:.1%} ≥ minimum, "
            f"fold_change={fc:.1f}×"
        )

    reasons = []
    if cov < 20:
        reasons.append(f"coverage={cov} < 20")
    if rate < 0.12:
        reasons.append(f"deletion_rate={rate:.1%} < 0.12")
    if fc < 2.0:
        reasons.append(f"fold_change={fc:.1f} < 2.0")
    if q > 0.10:
        reasons.append(f"q={q:.3f} > 0.10")
    if "non_uridine_ref" in str(flags):
        reasons.append("reference base is not T/U")
    if not reasons:
        reasons.append("did not meet combined thresholds")

    return "NOT CALLED: " + "; ".join(reasons)


def _build_summary(
    calls_df: pd.DataFrame,
    all_df: pd.DataFrame,
    known_psi: set,
) -> Dict:
    """Build a high-level summary dict for report generation."""
    n_high     = (calls_df["confidence"] == "HIGH_CONFIDENCE").sum()
    n_moderate = (calls_df["confidence"] == "MODERATE").sum()
    n_low      = (calls_df["confidence"] == "LOW").sum()
    n_known    = calls_df["is_known_psi"].sum()
    n_novel    = calls_df["novel"].sum()
    n_hp       = calls_df.get("homopolymer", pd.Series(False)).sum()
    n_renu     = calls_df.get("in_renu_region", pd.Series(False)).sum()

    called_positions = sorted(calls_df["position"].tolist())

    return {
        "total_calls":              int(n_high + n_moderate + n_low),
        "high_confidence_calls":    int(n_high),
        "moderate_calls":           int(n_moderate),
        "low_calls":                int(n_low),
        "known_psi_recovered":      int(n_known),
        "known_psi_total":          len(known_psi),
        "novel_calls":              int(n_novel),
        "homopolymer_flagged":      int(n_hp),
        "in_renu_region":           int(n_renu),
        "called_positions":         called_positions,
        "all_positions_tested":     len(all_df),
    }


def _log_calls(calls_df: pd.DataFrame, summary: Dict) -> None:
    logger.success(
        f"Pseudouridine calls:  "
        f"HIGH={summary['high_confidence_calls']}  "
        f"MODERATE={summary['moderate_calls']}  "
        f"LOW={summary['low_calls']}  "
        f"(known Ψ recovered: {summary['known_psi_recovered']}/{summary['known_psi_total']}  "
        f"novel: {summary['novel_calls']})"
    )
    if not calls_df.empty:
        logger.info(
            "Called positions (1-based): "
            + ", ".join(str(p) for p in summary["called_positions"])
        )


# ─── BED export ──────────────────────────────────────────────────────────────

def calls_to_bed(
    calls_df: pd.DataFrame,
    ref_name: str = "RNU4-2",
    output_path: Optional[str] = None,
) -> str:
    """
    Convert Ψ calls to BED6 format for IGV / genome browsers.

    BED is 0-based half-open, so position 4 (1-based) → "RNU4-2\t3\t4".
    Score column (0–1000) encodes confidence: HIGH=1000, MODERATE=600, LOW=300.
    """
    score_map = {"HIGH_CONFIDENCE": 1000, "MODERATE": 600, "LOW": 300}
    lines = ["# BID-seq pseudouridine calls (BED6)"]
    lines.append("# chrom\tstart\tend\tname\tscore\tstrand")

    for _, row in calls_df.iterrows():
        pos   = int(row["position"])
        start = pos - 1          # 0-based
        end   = pos              # half-open
        name  = f"psi_{pos}"
        score = score_map.get(row.get("confidence", "LOW"), 300)
        strand = "."
        lines.append(f"{ref_name}\t{start}\t{end}\t{name}\t{score}\t{strand}")

    bed_str = "\n".join(lines)
    if output_path:
        with open(output_path, "w") as f:
            f.write(bed_str + "\n")
        logger.debug(f"BED file written: {output_path}")
    return bed_str
