"""
deletion_stats.py — Statistical framework for discriminating real BID-seq
signal from nanopore sequencing noise.

The fundamental problem
-----------------------
Nanopore cDNA sequencing has a background deletion rate of ~5–8 % (R9.4.1)
or ~2–4 % (R10.4.1) at any given position, arising from polymerase slippage,
homopolymer errors, and basecalling artefacts.  BID-seq induces deletion rates
of 30–70 % at genuine pseudouridine (Ψ) sites.  This module separates signal
from noise.

Statistical approach
--------------------
TWO MODES:

1. PAIRED (treated + control)
   Fisher's exact test per position (2×2 contingency table).
   H₀: deletion rate is the same in treated and control.
   H₁: treated deletion rate > control (one-sided).
   Additional filters: fold change ≥ 3.0, absolute rate ≥ 0.12.

2. SINGLE-SAMPLE (no control)
   Background deletion rate b is estimated from the treated sample itself
   by taking the *background_percentile* (default 25th) of per-position
   deletion rates.  A one-sided binomial test is then performed at each
   position.
   Additional filters: same fold change and absolute rate thresholds,
   plus a stricter absolute minimum.

FDR correction
--------------
Benjamini–Hochberg correction applied across all tested positions.

Confidence tiers
----------------
HIGH_CONFIDENCE  q < 0.01, fold_change ≥ 5,  del_rate ≥ 0.25
MODERATE         q < 0.05, fold_change ≥ 3,  del_rate ≥ 0.15
LOW              q < 0.10, fold_change ≥ 2,  del_rate ≥ 0.10
NOISE            anything else
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests
from loguru import logger


# ─── Data classes ────────────────────────────────────────────────────────────

@dataclass
class StatResult:
    """Per-position deletion statistics result."""
    position:         int
    ref_base:         str
    coverage_t:       int
    deletion_count_t: int
    deletion_rate_t:  float
    coverage_c:       Optional[int]         = None
    deletion_count_c: Optional[int]         = None
    deletion_rate_c:  Optional[float]       = None
    background_rate:  float                 = 0.06
    fold_change:      float                 = 0.0
    p_value:          float                 = 1.0
    q_value:          float                 = 1.0
    confidence:       str                   = "NOISE"
    is_uridine:       bool                  = False
    homopolymer:      bool                  = False
    in_renu_region:   bool                  = False
    is_known_psi:     bool                  = False
    test_type:        str                   = "none"
    flags:            List[str]             = field(default_factory=list)

    def to_dict(self) -> Dict:
        d = {k: v for k, v in self.__dict__.items() if k != "flags"}
        d["flags"] = "|".join(self.flags) if self.flags else ""
        return d


# ─── Background estimation ───────────────────────────────────────────────────

def estimate_background_rate(
    pileup_df: pd.DataFrame,
    percentile: float = 25.0,
    min_coverage: int = 20,
    nanopore_prior: float = 0.06,
) -> float:
    """
    Estimate background (unmodified) deletion rate from the treated pileup.

    Strategy: the lowest-quartile deletion rates across all well-covered
    positions represent unmodified sites.  Their median is used as the
    background rate.  A nanopore-specific prior (0.06) is used as a floor
    so the background is never unrealistically low.

    Parameters
    ----------
    pileup_df:   Per-position pileup DataFrame (from pileup.compute_pileup).
    percentile:  Take positions below this percentile deletion rate as background.
    min_coverage: Exclude low-coverage positions.
    nanopore_prior: Minimum background rate (prevents underestimation on sparse data).
    """
    df = pileup_df[pileup_df["coverage"] >= min_coverage].copy()
    if df.empty:
        logger.warning("No positions with sufficient coverage; using nanopore prior for background.")
        return nanopore_prior

    rates = df["deletion_rate"].dropna().values
    if len(rates) == 0:
        return nanopore_prior

    threshold = np.percentile(rates, percentile)
    background_positions = rates[rates <= threshold]
    estimated = float(np.median(background_positions))

    # Never go below the nanopore noise floor
    background = max(estimated, nanopore_prior)
    logger.info(
        f"Background deletion rate: {background:.4f} "
        f"(estimated from {len(background_positions)} positions at ≤{percentile}th percentile; "
        f"prior floor={nanopore_prior})"
    )
    return background


# ─── Per-position tests ───────────────────────────────────────────────────────

def fisher_test(
    del_t: int, cov_t: int,
    del_c: int, cov_c: int,
) -> float:
    """One-sided Fisher's exact test: treated deletion rate > control."""
    if cov_t == 0 or cov_c == 0:
        return 1.0
    table = np.array([
        [del_t,       cov_t - del_t],
        [del_c,       cov_c - del_c],
    ])
    _, p = scipy_stats.fisher_exact(table, alternative="greater")
    return float(p)


def binomial_test(
    del_count: int, coverage: int, background_rate: float
) -> float:
    """One-sided binomial test: deletion_count is significantly above background."""
    if coverage == 0:
        return 1.0
    # P(X >= del_count | n=coverage, p=background_rate)
    # scipy >= 1.7: binomtest replaces deprecated binom_test
    try:
        p = scipy_stats.binomtest(del_count, coverage, background_rate, alternative="greater").pvalue
    except AttributeError:
        p = scipy_stats.binom_test(del_count, coverage, background_rate, alternative="greater")
    return float(p)


def _fold_change(rate_t: float, rate_c_or_bg: float, min_bg: float = 0.01) -> float:
    denominator = max(rate_c_or_bg, min_bg)
    return rate_t / denominator


# ─── Main analysis function ────────────────────────────────────────────────────

def run_deletion_analysis(
    treated_pileup: pd.DataFrame,
    control_pileup: Optional[pd.DataFrame] = None,
    annotations: Optional[List[Dict]] = None,
    *,
    background_percentile: float = 25.0,
    nanopore_bg_prior: float = 0.06,
    min_coverage: int = 20,
    min_deletion_rate: float = 0.12,
    min_fold_change: float = 3.0,
    fdr_threshold: float = 0.05,
    high_conf_fdr: float = 0.01,
    high_conf_min_rate: float = 0.25,
    high_conf_min_fc: float = 5.0,
) -> pd.DataFrame:
    """
    Perform the full deletion analysis and return a DataFrame of StatResults.

    Parameters
    ----------
    treated_pileup:  Pileup from BID-seq treated sample.
    control_pileup:  Optional pileup from untreated/mock sample.
    annotations:     Per-position annotation dicts from reference.annotate_positions().
    background_percentile: Percentile used for background estimation (single-sample).
    nanopore_bg_prior: Minimum background rate.
    min_coverage:    Exclude positions below this depth.
    min_deletion_rate: Absolute minimum deletion rate to even consider a site.
    min_fold_change: Minimum fold-change (treated/background).
    fdr_threshold:   BH-corrected q-value cutoff.
    high_conf_*:     Stricter thresholds for HIGH_CONFIDENCE tier.

    Returns
    -------
    pd.DataFrame with one row per reference position (all positions, not just
    called ones).  Includes p-values, q-values, confidence tier, flags.
    """
    is_paired = control_pileup is not None
    mode_str  = "paired (treated + control)" if is_paired else "single-sample"
    logger.info(f"Running deletion analysis in {mode_str} mode …")

    # Estimate background from treated (for single-sample or as prior for paired)
    bg_rate = estimate_background_rate(
        treated_pileup,
        percentile=background_percentile,
        min_coverage=min_coverage,
        nanopore_prior=nanopore_bg_prior,
    )

    # Build annotation lookup {position -> dict}
    ann_lookup: Dict[int, Dict] = {}
    if annotations:
        ann_lookup = {a["position"]: a for a in annotations}

    results: List[StatResult] = []

    for _, row_t in treated_pileup.iterrows():
        pos = int(row_t["position"])
        ann = ann_lookup.get(pos, {})

        cov_t = int(row_t["coverage"])
        del_t = int(row_t["deletion_count"])
        rate_t = float(row_t["deletion_rate"]) if cov_t > 0 else float("nan")

        # Corresponding control row
        cov_c, del_c, rate_c = None, None, None
        if is_paired and control_pileup is not None:
            row_c = control_pileup[control_pileup["position"] == pos]
            if not row_c.empty:
                cov_c = int(row_c.iloc[0]["coverage"])
                del_c = int(row_c.iloc[0]["deletion_count"])
                rate_c = float(row_c.iloc[0]["deletion_rate"]) if cov_c > 0 else 0.0

        # Effective background for this position
        eff_bg = rate_c if (rate_c is not None and cov_c and cov_c >= min_coverage) else bg_rate

        # Fold change
        fc = _fold_change(rate_t if not np.isnan(rate_t) else 0.0, eff_bg)

        # Statistical test
        if cov_t < min_coverage:
            p_val = 1.0
            test_type = "insufficient_coverage"
        elif is_paired and cov_c is not None and cov_c >= min_coverage:
            p_val = fisher_test(del_t, cov_t, del_c, cov_c)
            test_type = "fisher_exact"
        else:
            p_val = binomial_test(del_t, cov_t, bg_rate)
            test_type = "binomial"

        # Flags
        flags: List[str] = []
        if ann.get("homopolymer"):
            flags.append("homopolymer")
        if not ann.get("is_uridine", True):
            flags.append("non_uridine_ref")
        if ann.get("in_renu_region"):
            flags.append("renu_region")

        sr = StatResult(
            position         = pos,
            ref_base         = str(row_t.get("ref_base", "N")),
            coverage_t       = cov_t,
            deletion_count_t = del_t,
            deletion_rate_t  = float(rate_t) if not np.isnan(rate_t) else 0.0,
            coverage_c       = cov_c,
            deletion_count_c = del_c,
            deletion_rate_c  = rate_c,
            background_rate  = eff_bg,
            fold_change      = fc,
            p_value          = p_val,
            q_value          = 1.0,       # filled in after BH correction
            confidence       = "NOISE",   # filled in after BH correction
            is_uridine       = bool(ann.get("is_uridine", False)),
            homopolymer      = bool(ann.get("homopolymer", False)),
            in_renu_region   = bool(ann.get("in_renu_region", False)),
            is_known_psi     = bool(ann.get("is_known_psi", False)),
            test_type        = test_type,
            flags            = flags,
        )
        results.append(sr)

    # ── BH FDR correction across all positions ────────────────────────────────
    p_values = np.array([r.p_value for r in results])
    valid    = np.isfinite(p_values) & (p_values <= 1.0)

    q_values = np.ones(len(p_values))
    if valid.sum() > 0:
        _, q_vals_valid, _, _ = multipletests(p_values[valid], method="fdr_bh")
        q_values[valid] = q_vals_valid

    for r, q in zip(results, q_values):
        r.q_value = float(q)

    # ── Assign confidence tiers ───────────────────────────────────────────────
    for r in results:
        r.confidence = _assign_confidence(
            r,
            fdr_threshold    = fdr_threshold,
            high_conf_fdr    = high_conf_fdr,
            high_conf_min_rate = high_conf_min_rate,
            high_conf_min_fc = high_conf_min_fc,
            min_deletion_rate = min_deletion_rate,
            min_fold_change  = min_fold_change,
        )

    result_df = pd.DataFrame([r.to_dict() for r in results])
    _log_summary(result_df)
    return result_df


def _assign_confidence(
    r: StatResult,
    fdr_threshold: float,
    high_conf_fdr: float,
    high_conf_min_rate: float,
    high_conf_min_fc: float,
    min_deletion_rate: float,
    min_fold_change: float,
) -> str:
    """Assign a confidence tier to a single position result."""
    # Hard exclusions
    if r.coverage_t < 20:
        return "NOISE"
    if not r.is_uridine and "non_uridine_ref" in r.flags:
        # BID-seq only acts on pseudouridine (T/U in reference).
        # A high deletion rate at a non-U position is artefactual.
        return "NOISE"
    if r.deletion_rate_t < min_deletion_rate:
        return "NOISE"

    # HIGH_CONFIDENCE
    if (
        r.q_value        <= high_conf_fdr
        and r.fold_change >= high_conf_min_fc
        and r.deletion_rate_t >= high_conf_min_rate
    ):
        return "HIGH_CONFIDENCE"

    # MODERATE
    if (
        r.q_value        <= fdr_threshold
        and r.fold_change >= min_fold_change
        and r.deletion_rate_t >= 0.15
    ):
        return "MODERATE"

    # LOW
    if (
        r.q_value        <= 0.10
        and r.fold_change >= 2.0
        and r.deletion_rate_t >= min_deletion_rate
    ):
        return "LOW"

    return "NOISE"


def _log_summary(df: pd.DataFrame) -> None:
    counts = df["confidence"].value_counts()
    logger.success(
        "Deletion analysis complete. "
        + "  |  ".join(
            f"{tier}: {counts.get(tier, 0)}"
            for tier in ("HIGH_CONFIDENCE", "MODERATE", "LOW", "NOISE")
        )
    )
