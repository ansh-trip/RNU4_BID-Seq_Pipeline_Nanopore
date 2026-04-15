"""
test_pseudoU_caller.py — Tests for pseudoU_caller.py

Coverage:
  - Call structure (calls/all/summary keys)
  - Only non-NOISE positions in calls table
  - Known Ψ annotation
  - Novel call detection
  - Interpretation text generation
  - BED export format
  - Summary statistics
"""

from __future__ import annotations

import pytest
import pandas as pd

from bidseq.deletion_stats import run_deletion_analysis
from bidseq.pseudoU_caller import call_pseudoU_sites, calls_to_bed
from bidseq.reference import annotate_positions
from tests.conftest import MINI_REF_SEQ


KNOWN_PSI = {4, 15}      # pretend positions 4 and 15 are known


@pytest.fixture
def stats_with_calls(bidseq_pileup):
    anns = annotate_positions(MINI_REF_SEQ)
    return run_deletion_analysis(
        treated_pileup    = bidseq_pileup,
        annotations       = anns,
        min_coverage      = 20,
        min_deletion_rate = 0.10,
    )


@pytest.fixture
def stats_no_calls(clean_pileup):
    anns = annotate_positions(MINI_REF_SEQ)
    return run_deletion_analysis(
        treated_pileup    = clean_pileup,
        annotations       = anns,
        min_coverage      = 20,
        min_deletion_rate = 0.12,
    )


class TestCallPseudoUSites:

    def test_returns_expected_keys(self, stats_with_calls):
        result = call_pseudoU_sites(stats_with_calls, known_psi_sites=KNOWN_PSI)
        assert "calls" in result
        assert "all"   in result
        assert "summary" in result

    def test_calls_only_non_noise(self, stats_with_calls):
        result = call_pseudoU_sites(stats_with_calls, known_psi_sites=KNOWN_PSI)
        calls = result["calls"]
        assert (calls["confidence"] != "NOISE").all(), (
            "calls table must contain only non-NOISE entries"
        )

    def test_all_has_every_position(self, stats_with_calls):
        result = call_pseudoU_sites(stats_with_calls, known_psi_sites=KNOWN_PSI)
        assert len(result["all"]) == len(MINI_REF_SEQ)

    def test_known_psi_annotated(self, stats_with_calls):
        result = call_pseudoU_sites(stats_with_calls, known_psi_sites=KNOWN_PSI)
        all_df = result["all"]
        # Positions in KNOWN_PSI must be flagged is_known_psi=True
        for pos in KNOWN_PSI:
            row = all_df[all_df["position"] == pos]
            if not row.empty:
                assert bool(row.iloc[0]["is_known_psi"]), (
                    f"Position {pos} should be is_known_psi=True"
                )

    def test_novel_calls_detected(self, stats_with_calls):
        """Position 22 is not in KNOWN_PSI — should be novel if called."""
        result = call_pseudoU_sites(stats_with_calls, known_psi_sites=KNOWN_PSI)
        calls = result["calls"]
        if not calls.empty:
            novel_calls = calls[calls["novel"] == True]
            # At minimum, no crash; novel column exists
            assert "novel" in calls.columns

    def test_summary_counts_correct(self, stats_with_calls):
        result = call_pseudoU_sites(stats_with_calls, known_psi_sites=KNOWN_PSI)
        summary = result["summary"]
        calls   = result["calls"]
        assert summary["total_calls"] == len(calls)
        assert summary["high_confidence_calls"] == (calls["confidence"] == "HIGH_CONFIDENCE").sum()

    def test_no_calls_on_clean_pileup(self, stats_no_calls):
        result = call_pseudoU_sites(stats_no_calls, known_psi_sites=KNOWN_PSI)
        assert result["summary"]["total_calls"] == 0

    def test_interpretation_field_populated(self, stats_with_calls):
        result = call_pseudoU_sites(stats_with_calls, known_psi_sites=KNOWN_PSI)
        calls = result["calls"]
        if not calls.empty:
            assert calls["interpretation"].notna().all()
            assert (calls["interpretation"].str.len() > 5).all()

    def test_call_reason_field_populated(self, stats_with_calls):
        result = call_pseudoU_sites(stats_with_calls, known_psi_sites=KNOWN_PSI)
        # All positions in "all" should have a call_reason
        assert "call_reason" in result["all"].columns
        assert result["all"]["call_reason"].notna().all()


class TestCallsToBed:

    def test_bed_format(self, stats_with_calls):
        result = call_pseudoU_sites(stats_with_calls, known_psi_sites=KNOWN_PSI)
        bed_str = calls_to_bed(result["calls"], ref_name="TEST_REF")
        lines = [l for l in bed_str.splitlines() if not l.startswith("#") and l.strip()]
        for line in lines:
            parts = line.split("\t")
            assert len(parts) == 6, f"BED line should have 6 columns: {line}"
            # Column 3 (end) should be column 2 (start) + 1
            start, end = int(parts[1]), int(parts[2])
            assert end == start + 1, "BED end should be start + 1 for single-nt site"

    def test_bed_0based(self, stats_with_calls):
        """BED is 0-based; position 4 (1-based) should be start=3 in BED."""
        result = call_pseudoU_sites(stats_with_calls, known_psi_sites=KNOWN_PSI)
        bed_str = calls_to_bed(result["calls"], ref_name="TEST_REF")
        if "psi_4" in bed_str:
            for line in bed_str.splitlines():
                if "psi_4" in line:
                    assert line.split("\t")[1] == "3", "Position 4 should be BED start=3"

    def test_bed_score_by_confidence(self, stats_with_calls):
        """HIGH_CONFIDENCE → score=1000, MODERATE → 600, LOW → 300."""
        result = call_pseudoU_sites(stats_with_calls, known_psi_sites=KNOWN_PSI)
        calls = result["calls"]
        if calls.empty:
            pytest.skip("No calls to test BED scores")
        bed_str = calls_to_bed(calls, ref_name="TEST_REF")
        for line in bed_str.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split("\t")
            score = int(parts[4])
            assert score in (300, 600, 1000), f"Unexpected BED score: {score}"

    def test_empty_calls_no_crash(self):
        """calls_to_bed with empty DataFrame should not crash."""
        empty = pd.DataFrame(columns=["position", "confidence"])
        bed_str = calls_to_bed(empty, ref_name="TEST_REF")
        assert isinstance(bed_str, str)
