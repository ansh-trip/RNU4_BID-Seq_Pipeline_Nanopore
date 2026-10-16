"""
test_pileup.py — Tests for pileup.py utility functions.

BAM-level pileup is tested in the integration tests.  Here we test:
  - merge_pileups
  - save/load_pileup_tsv round-trip
  - pileup shape and column presence
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bidseq.pileup import merge_pileups, save_pileup_tsv, load_pileup_tsv


class TestMergePileups:

    def test_merge_preserves_positions(self, bidseq_pileup, control_pileup):
        merged = merge_pileups(bidseq_pileup, control_pileup)
        assert len(merged) == len(bidseq_pileup)
        assert "position" in merged.columns

    def test_merge_has_treated_and_control_columns(self, bidseq_pileup, control_pileup):
        merged = merge_pileups(bidseq_pileup, control_pileup)
        assert "deletion_rate_t" in merged.columns
        assert "deletion_rate_c" in merged.columns
        assert "coverage_t" in merged.columns
        assert "coverage_c" in merged.columns

    def test_merge_no_nans_in_coverage(self, bidseq_pileup, control_pileup):
        """After fillna(0), coverage columns should have no NaN."""
        merged = merge_pileups(bidseq_pileup, control_pileup)
        assert not merged["coverage_t"].isna().any()
        assert not merged["coverage_c"].isna().any()


class TestSaveLoadPileupTsv:

    def test_round_trip(self, bidseq_pileup):
        """Save then load should reproduce the original DataFrame."""
        with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as f:
            tmp = Path(f.name)

        save_pileup_tsv(bidseq_pileup, tmp)
        loaded = load_pileup_tsv(tmp)

        # Numeric columns should match
        for col in ("position", "coverage", "deletion_count", "deletion_rate"):
            np.testing.assert_allclose(
                bidseq_pileup[col].values,
                loaded[col].values,
                rtol=1e-4,
                err_msg=f"Column {col} mismatch after round-trip",
            )
        tmp.unlink()

    def test_ins_details_serialised(self, bidseq_pileup):
        """ins_details (dict column) should survive save/load as dict."""
        # Give position 5 some fake insertions
        pileup = bidseq_pileup.copy()
        pileup.at[4, "ins_details"] = {"AT": 5, "A": 2}

        with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as f:
            tmp = Path(f.name)

        save_pileup_tsv(pileup, tmp)
        loaded = load_pileup_tsv(tmp)

        recovered = loaded.iloc[4]["ins_details"]
        assert isinstance(recovered, dict)
        assert recovered.get("AT") == 5
        tmp.unlink()


class TestPileupShape:

    def test_required_columns_present(self, bidseq_pileup):
        required = [
            "position", "ref_base", "coverage", "deletion_count",
            "deletion_rate", "insertion_count", "insertion_rate",
        ]
        for col in required:
            assert col in bidseq_pileup.columns, f"Column {col} missing"

    def test_deletion_rate_bounded(self, bidseq_pileup):
        rates = bidseq_pileup["deletion_rate"].dropna()
        assert (rates >= 0).all() and (rates <= 1).all()

    def test_position_is_1based(self, bidseq_pileup):
        assert bidseq_pileup["position"].min() == 1
        assert bidseq_pileup["position"].max() == len(bidseq_pileup)
