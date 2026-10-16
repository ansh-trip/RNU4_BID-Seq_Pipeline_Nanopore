"""
test_utils.py — Tests for utils.py helper functions.
"""

from __future__ import annotations

import gzip
import json
import tempfile
from pathlib import Path

import pytest
from bidseq.utils import (
    find_homopolymers,
    homopolymer_mask,
    reverse_complement,
    write_json,
    read_json,
    is_gzipped,
)


class TestHomopolymers:

    def test_simple_run(self):
        runs = find_homopolymers("AAAACGT", min_len=4)
        assert len(runs) == 1
        assert runs[0] == (0, 4, "A")

    def test_no_run_below_min(self):
        runs = find_homopolymers("AAACGT", min_len=4)
        assert len(runs) == 0

    def test_multiple_runs(self):
        runs = find_homopolymers("AAAACCCCTTTTT", min_len=4)
        assert len(runs) == 3

    def test_mask_length(self):
        mask = homopolymer_mask("AAAACGT", min_len=4)
        assert len(mask) == 7

    def test_mask_flagged_positions(self):
        mask = homopolymer_mask("AAAACGT", min_len=4)
        # First 4 positions should be flagged
        assert all(mask[:4])
        assert not any(mask[4:])

    def test_empty_string(self):
        runs = find_homopolymers("", min_len=4)
        assert runs == []


class TestReverseComplement:

    def test_basic(self):
        assert reverse_complement("ACGT") == "ACGT"

    def test_all_a(self):
        assert reverse_complement("AAAA") == "TTTT"

    def test_mixed(self):
        assert reverse_complement("AATTGG") == "CCAATT"

    def test_empty(self):
        assert reverse_complement("") == ""


class TestJsonIO:

    def test_write_and_read_dict(self):
        data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp = Path(f.name)
        write_json(data, tmp)
        recovered = read_json(tmp)
        assert recovered == data
        tmp.unlink()

    def test_write_nested(self):
        data = {"nested": {"a": 1, "b": [2, 3]}}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp = Path(f.name)
        write_json(data, tmp)
        recovered = read_json(tmp)
        assert recovered["nested"]["b"] == [2, 3]
        tmp.unlink()


class TestIsGzipped:

    def test_plain_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"ACGT\nACGT\n")
            tmp = Path(f.name)
        assert not is_gzipped(tmp)
        tmp.unlink()

    def test_gzipped_file(self):
        with tempfile.NamedTemporaryFile(suffix=".gz", delete=False) as f:
            tmp = Path(f.name)
        with gzip.open(tmp, "wb") as gz:
            gz.write(b"ACGT\n")
        assert is_gzipped(tmp)
        tmp.unlink()
