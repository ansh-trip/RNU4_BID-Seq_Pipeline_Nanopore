"""
test_deletion_stats.py — Tests for deletion_stats.py

Coverage:
  - Background rate estimation
  - Fisher's exact test
  - Binomial test
  - Confidence tier assignment
  - FDR correction (that it's applied)
  - Low-coverage exclusion
  - Non-uridine exclusion
  - Homopolymer flagging
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bidseq.deletion_stats import (
    estimate_background_rate,
    fisher_test,
    binomial_test,
    run_deletion_analysis,
)
from bidseq.reference import annotate_positions
from tests.conftest import MINI_REF_SEQ


# ─── Background estimation ────────────────────────────────────────────────────

class TestBackgroundEstimation:
    def test_uniform_background(self):
        """All positions at 6% — background should be ~6%."""
        n = 30
        cov  = np.full(n, 100)
        dels = np.full(n, 6, dtype=int)
        df = pd.DataFrame({
            "coverage":      cov,
            "deletion_rate": dels / cov,
        })
        bg = estimate_background_rate(df, percentile=25.0, nanopore_prior=0.0)
        assert 0.04 <= bg <= 0.08, f"Expected ~0.06, got {bg}"

    def test_prior_floor_applied(self):
        """Background should never go below nanopore_prior."""
        n = 30
        cov  = np.full(n, 100)
        dels = np.zeros(n, dtype=int)   # 0% deletion everywhere
        df = pd.DataFrame({
            "coverage":      cov,
            "deletion_rate": dels / cov,
        })
        bg = estimate_background_rate(df, percentile=25.0, nanopore_prior=0.06)
        assert bg >= 0.06, "Background should be at least the nanopore prior"

    def test_low_coverage_excluded(self):
        """Positions below min_coverage should be excluded from background estimation."""
        n = 30
        cov = np.array([2] * 15 + [100] * 15)
        dels = np.array([0] * 15 + [6] * 15, dtype=int)
        df = pd.DataFrame({
            "coverage":      cov,
            "deletion_rate": np.where(cov > 0, dels / cov, 0),
        })
        # Only the high-coverage positions should count
        bg = estimate_background_rate(df, percentile=25.0, min_coverage=20, nanopore_prior=0.0)
        assert 0.04 <= bg <= 0.08


# ─── Statistical tests ────────────────────────────────────────────────────────

class TestFisherTest:
    def test_high_signal_low_p(self):
        """60% treated deletion vs 6% control → very small p-value."""
        p = fisher_test(del_t=120, cov_t=200, del_c=12, cov_c=200)
        assert p < 1e-20, f"Expected p << 0.05, got {p}"

    def test_equal_rates_high_p(self):
        """Same deletion rates → high p-value (H0 true)."""
        p = fisher_test(del_t=12, cov_t=200, del_c=12, cov_c=200)
        assert p > 0.3, f"Expected p > 0.3 for equal rates, got {p}"

    def test_zero_coverage_returns_one(self):
        p = fisher_test(del_t=0, cov_t=0, del_c=0, cov_c=0)
        assert p == 1.0

    def test_moderate_signal(self):
        """30% treated, 6% control — should be significant."""
        p = fisher_test(del_t=60, cov_t=200, del_c=12, cov_c=200)
        assert p < 0.001


class TestBinomialTest:
    def test_high_signal(self):
        """60% deletion when background 6% → extremely significant."""
        p = binomial_test(120, 200, 0.06)
        assert p < 1e-30

    def test_background_level(self):
        """Deletion count at exact background rate → should not be significant."""
        p = binomial_test(12, 200, 0.06)
        assert p > 0.3

    def test_zero_coverage(self):
        p = binomial_test(0, 0, 0.06)
        assert p == 1.0


# ─── Full analysis pipeline ───────────────────────────────────────────────────

class TestRunDeletionAnalysis:

    def test_calls_true_psi_sites_paired(self, bidseq_pileup, control_pileup):
        """Positions with high deletion rate should be called in paired mode."""
        anns = annotate_positions(MINI_REF_SEQ)
        df = run_deletion_analysis(
            treated_pileup    = bidseq_pileup,
            control_pileup    = control_pileup,
            annotations       = anns,
            min_coverage      = 20,
            min_deletion_rate = 0.12,
        )
        # Position 4 (0-indexed 3) is T and should be called HIGH or MODERATE
        pos4 = df[df["position"] == 4].iloc[0]
        assert pos4["confidence"] in ("HIGH_CONFIDENCE", "MODERATE"), (
            f"Position 4 should be called, got {pos4['confidence']}"
        )

    def test_calls_true_psi_sites_single_sample(self, bidseq_pileup):
        """BID-seq signal recovered in single-sample mode."""
        anns = annotate_positions(MINI_REF_SEQ)
        df = run_deletion_analysis(
            treated_pileup    = bidseq_pileup,
            control_pileup    = None,
            annotations       = anns,
            min_coverage      = 20,
            min_deletion_rate = 0.12,
        )
        pos4 = df[df["position"] == 4].iloc[0]
        assert pos4["confidence"] in ("HIGH_CONFIDENCE", "MODERATE")

    def test_noise_positions_not_called(self, clean_pileup):
        """Uniform background pileup should produce no calls."""
        anns = annotate_positions(MINI_REF_SEQ)
        df = run_deletion_analysis(
            treated_pileup    = clean_pileup,
            control_pileup    = None,
            annotations       = anns,
            min_coverage      = 20,
            min_deletion_rate = 0.12,
        )
        non_noise = df[df["confidence"] != "NOISE"]
        # At most 1 false positive allowed in 30 positions
        assert len(non_noise) <= 1, (
            f"Expected no calls on clean pileup, got {len(non_noise)}"
        )

    def test_low_coverage_excluded(self, low_coverage_pileup):
        """All positions below min_coverage should be NOISE."""
        anns = annotate_positions(MINI_REF_SEQ)
        df = run_deletion_analysis(
            treated_pileup    = low_coverage_pileup,
            control_pileup    = None,
            annotations       = anns,
            min_coverage      = 20,
        )
        non_noise = df[df["confidence"] != "NOISE"]
        assert len(non_noise) == 0, "Low-coverage positions must all be NOISE"

    def test_non_uridine_positions_not_called(self, bidseq_pileup):
        """Non-T/U positions should not be called even with high deletion rate."""
        # Manually spike deletion at position 1 (A, not T)
        pileup = bidseq_pileup.copy()
        pileup.loc[pileup["position"] == 1, "deletion_count"] = 120
        pileup.loc[pileup["position"] == 1, "deletion_rate"]  = 0.60

        anns = annotate_positions(MINI_REF_SEQ)
        df = run_deletion_analysis(
            treated_pileup  = pileup,
            annotations     = anns,
            min_coverage    = 20,
        )
        pos1 = df[df["position"] == 1].iloc[0]
        assert pos1["confidence"] == "NOISE", (
            "Non-uridine position should not be called"
        )

    def test_fdr_correction_applied(self, bidseq_pileup, control_pileup):
        """q-values should differ from raw p-values (BH correction applied)."""
        anns = annotate_positions(MINI_REF_SEQ)
        df = run_deletion_analysis(
            treated_pileup = bidseq_pileup,
            control_pileup = control_pileup,
            annotations    = anns,
        )
        # p-value and q-value should not always be equal
        eq = (df["p_value"].round(6) == df["q_value"].round(6)).all()
        assert not eq, "FDR correction should change some p-values"

    def test_output_has_all_positions(self, clean_pileup):
        """Output should have one row per reference position."""
        anns = annotate_positions(MINI_REF_SEQ)
        df = run_deletion_analysis(treated_pileup=clean_pileup, annotations=anns)
        assert len(df) == len(MINI_REF_SEQ)

    def test_high_confidence_tier(self, bidseq_pileup, control_pileup):
        """Position 4 with 62% deletion vs 6% control should be HIGH_CONFIDENCE."""
        anns = annotate_positions(MINI_REF_SEQ)
        df = run_deletion_analysis(
            treated_pileup  = bidseq_pileup,
            control_pileup  = control_pileup,
            annotations     = anns,
        )
        high = df[df["confidence"] == "HIGH_CONFIDENCE"]
        assert len(high) >= 1, "Expected at least one HIGH_CONFIDENCE call"
        # Position 4 must be in there
        assert 4 in high["position"].values

    def test_fold_change_computed_correctly(self, bidseq_pileup, control_pileup):
        """Fold change at spiked position should be >> 1."""
        anns = annotate_positions(MINI_REF_SEQ)
        df = run_deletion_analysis(
            treated_pileup = bidseq_pileup,
            control_pileup = control_pileup,
            annotations    = anns,
        )
        pos4_fc = df[df["position"] == 4].iloc[0]["fold_change"]
        assert pos4_fc > 5.0, f"Expected fold_change > 5 at pos 4, got {pos4_fc}"
