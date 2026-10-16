"""
test_renu_tracker.py — Tests for renu_tracker.py

Because the tracker requires a real BAM file, most tests operate on the
helper functions that can be tested without BAM I/O.  The BAM-level
integration test is in test_integration.py (skipped if minimap2 absent).
"""

from __future__ import annotations

import pytest
from bidseq.renu_tracker import (
    _infer_zygosity,
    _build_interpretation,
    RenuSiteSummary,
    RENU_INS_POS_1BASED,
    VAF_HET_LOW,
    VAF_HET_HIGH,
    VAF_HOM_LOW,
)


# ─── Zygosity inference ───────────────────────────────────────────────────────

class TestInferZygosity:

    def test_wild_type_no_reads(self):
        z = _infer_zygosity(vaf=0.0, n_renu=0, min_reads=3, min_vaf=0.02,
                            het_low=0.30, het_high=0.70, hom_low=0.85)
        assert z == "WILD_TYPE"

    def test_wild_type_below_min_reads(self):
        z = _infer_zygosity(vaf=0.45, n_renu=2, min_reads=3, min_vaf=0.02,
                            het_low=0.30, het_high=0.70, hom_low=0.85)
        assert z == "WILD_TYPE"

    def test_wild_type_below_min_vaf(self):
        z = _infer_zygosity(vaf=0.01, n_renu=5, min_reads=3, min_vaf=0.02,
                            het_low=0.30, het_high=0.70, hom_low=0.85)
        assert z == "WILD_TYPE"

    def test_heterozygous(self):
        z = _infer_zygosity(vaf=0.48, n_renu=100, min_reads=3, min_vaf=0.02,
                            het_low=0.30, het_high=0.70, hom_low=0.85)
        assert z == "HETEROZYGOUS"

    def test_homozygous(self):
        z = _infer_zygosity(vaf=0.92, n_renu=200, min_reads=3, min_vaf=0.02,
                            het_low=0.30, het_high=0.70, hom_low=0.85)
        assert z == "HOMOZYGOUS"

    def test_mosaic(self):
        z = _infer_zygosity(vaf=0.08, n_renu=10, min_reads=3, min_vaf=0.02,
                            het_low=0.30, het_high=0.70, hom_low=0.85)
        assert z == "MOSAIC_OR_LOW_SIGNAL"

    def test_near_hom(self):
        z = _infer_zygosity(vaf=0.78, n_renu=150, min_reads=3, min_vaf=0.02,
                            het_low=0.30, het_high=0.70, hom_low=0.85)
        assert z == "NEAR_HOMOZYGOUS"


# ─── Interpretation text ──────────────────────────────────────────────────────

class TestBuildInterpretation:

    def _make_summary(self, **kwargs):
        s = RenuSiteSummary()
        s.total_reads     = kwargs.get("total_reads", 200)
        s.wild_type_reads = kwargs.get("wt", 100)
        s.renu_t_ins_reads = kwargs.get("renu", 100)
        s.vaf             = kwargs.get("vaf", 0.50)
        s.zygosity        = kwargs.get("zygosity", "HETEROZYGOUS")
        return s

    def test_wt_interpretation_contains_wild_type(self):
        s = self._make_summary(renu=0, vaf=0.0, zygosity="WILD_TYPE")
        text = _build_interpretation(s, min_reads=3, min_vaf=0.02)
        assert "WILD-TYPE" in text.upper() or "wild" in text.lower()

    def test_het_interpretation_contains_detected(self):
        s = self._make_summary(vaf=0.48, zygosity="HETEROZYGOUS")
        text = _build_interpretation(s, min_reads=3, min_vaf=0.02)
        assert "DETECTED" in text

    def test_low_coverage_warns(self):
        s = self._make_summary(total_reads=5, vaf=0.0, zygosity="WILD_TYPE")
        text = _build_interpretation(s, min_reads=3, min_vaf=0.02)
        assert "coverage" in text.lower() or "insufficient" in text.lower()

    def test_mosaic_interpretation(self):
        s = self._make_summary(renu=10, vaf=0.05, zygosity="MOSAIC_OR_LOW_SIGNAL")
        text = _build_interpretation(s, min_reads=3, min_vaf=0.02)
        assert "mosaic" in text.lower() or "low-level" in text.lower()


# ─── RenuSiteSummary.to_dict ──────────────────────────────────────────────────

class TestRenuSiteSummaryToDict:

    def test_to_dict_has_required_keys(self):
        s = RenuSiteSummary()
        s.total_reads = 100
        s.renu_t_ins_reads = 50
        s.wild_type_reads = 50
        s.vaf = 0.50
        s.zygosity = "HETEROZYGOUS"
        s.interpretation = "Test"
        d = s.to_dict()
        expected = [
            "renu_insertion_position_1based",
            "renu_nomenclature",
            "total_reads_assessed",
            "wild_type_reads",
            "renu_t_insertion_reads",
            "vaf",
            "zygosity",
            "interpretation",
            "coverage_at_site",
        ]
        for key in expected:
            assert key in d, f"Key '{key}' missing from to_dict()"

    def test_to_dict_nomenclature(self):
        d = RenuSiteSummary().to_dict()
        assert d["renu_nomenclature"] == "n.64_65insT"
        assert d["renu_insertion_position_1based"] == RENU_INS_POS_1BASED

    def test_vaf_rounded(self):
        s = RenuSiteSummary()
        s.vaf = 0.123456789
        d = s.to_dict()
        assert d["vaf"] == round(0.123456789, 4)


# ─── Constants ────────────────────────────────────────────────────────────────

def test_renu_insertion_position():
    """Sanity check: RENU insertion should be at position 64 (1-based)."""
    assert RENU_INS_POS_1BASED == 64, (
        f"Expected RENU_INS_POS_1BASED=64, got {RENU_INS_POS_1BASED}"
    )
