"""
test_reference.py — Tests for reference.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from bidseq.reference import (
    annotate_positions,
    read_reference_sequence,
    get_reference_name,
    _write_builtin,
    KNOWN_PSI_SITES,
    RENU_INSERTION_POS_1BASED,
)
RENU_INS_POS_1BASED = RENU_INSERTION_POS_1BASED  # alias
from tests.conftest import MINI_REF_SEQ, RNU4_2_MOCK


class TestAnnotatePositions:

    def test_length_matches_sequence(self):
        anns = annotate_positions(MINI_REF_SEQ)
        assert len(anns) == len(MINI_REF_SEQ)

    def test_positions_1based(self):
        anns = annotate_positions(MINI_REF_SEQ)
        assert anns[0]["position"] == 1
        assert anns[-1]["position"] == len(MINI_REF_SEQ)

    def test_uridine_flag(self):
        # T at position 4 (0-indexed 3) in MINI_REF_SEQ = "ACGTACGT..."
        anns = annotate_positions(MINI_REF_SEQ)
        assert anns[3]["is_uridine"] is True   # T at pos 4
        assert anns[0]["is_uridine"] is False  # A at pos 1

    def test_known_psi_flag(self):
        known = {4, 15}
        anns = annotate_positions(MINI_REF_SEQ, known_psi=known)
        assert anns[3]["is_known_psi"] is True   # pos 4
        assert anns[14]["is_known_psi"] is True  # pos 15
        assert anns[0]["is_known_psi"] is False  # pos 1

    def test_homopolymer_flag(self):
        # MINI_REF_SEQ has "TTTT" at positions 25-28 (0-indexed 24-27)
        anns = annotate_positions(MINI_REF_SEQ)
        hp_pos = [a["position"] for a in anns if a["homopolymer"]]
        # The TTTT run should produce homopolymer flags
        assert len(hp_pos) > 0

    def test_renu_region_flag(self):
        """Positions near RENU_INS_POS_1BASED should be flagged in_renu_region."""
        anns = annotate_positions(RNU4_2_MOCK, renu_pos=RENU_INS_POS_1BASED)
        renu_pos = [a["position"] for a in anns if a["in_renu_region"]]
        assert RENU_INS_POS_1BASED in renu_pos


class TestReadReferenceSequence:

    def test_reads_fasta_correctly(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fa", delete=False) as f:
            f.write(">TEST\nACGTACGT\nACGT\n")
            tmp = Path(f.name)
        seq = read_reference_sequence(tmp)
        assert seq == "ACGTACGTACGT"
        tmp.unlink()

    def test_multiline_fasta(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fa", delete=False) as f:
            f.write(">SEQ\nAAAA\nCCCC\nGGGG\n")
            tmp = Path(f.name)
        seq = read_reference_sequence(tmp)
        assert seq == "AAAACCCCGGGG"
        tmp.unlink()


class TestGetReferenceName:

    def test_returns_first_word(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fa", delete=False) as f:
            f.write(">RNU4-2 Homo sapiens\nACGT\n")
            tmp = Path(f.name)
        name = get_reference_name(tmp)
        assert name == "RNU4-2"
        tmp.unlink()


class TestBuiltinReference:

    def test_writes_valid_fasta(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "rnu4_2.fa"
            result = _write_builtin(out)
            assert result.exists()
            with open(result) as f:
                first_line = f.readline()
            assert first_line.startswith(">")

    def test_sequence_is_dna(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "rnu4_2.fa"
            _write_builtin(out)
            seq = read_reference_sequence(out)
            assert all(c in "ACGTN" for c in seq), "Built-in sequence should be uppercase DNA"


class TestConstants:

    def test_known_psi_sites_are_ints(self):
        assert all(isinstance(p, int) for p in KNOWN_PSI_SITES)

    def test_renu_pos_is_64(self):
        assert RENU_INS_POS_1BASED == 64
