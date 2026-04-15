"""
conftest.py — Shared pytest fixtures for the BID-seq pipeline tests.

All fixtures that require real BAM/FASTQ files use synthetic data generated
in-memory (no external files needed).  Fixtures that return DataFrames
represent the expected intermediate outputs of each module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ─── Reference sequence ───────────────────────────────────────────────────────

# Synthetic 30-nt mini reference (contains T at positions 4, 15, 22 — our "pseudoU")
# and a run of 4 T's at positions 25-28 (homopolymer trap)
MINI_REF_SEQ = "ACGTACGTACGATCGATCGATTTTACGTTTT"
#                1234567890123456789012345678901
#                         1111111111222222222233
# T at pos 4, 8, 15, 22 in this string (1-based)
MINI_REF_NAME = "TEST_REF"
MINI_REF_LEN  = len(MINI_REF_SEQ)

# Real-ish RNU4-2 mock sequence (141 nt synthetic approximation)
# T is FORCED at positions 4, 72, 79 (1-based) so the annotation tests work correctly.
def _make_mock_rnu4_2() -> str:
    base = (
        "ACGCACGCACGAACGAACGAACGAACGAACGAACGAACGAACGAACGAACGAACGAACGA"
        "CGAACGAACGAACGAACGAACGAACGAACGAACGAACGAACGAACGAACGAACGAACGAA"
        "CGAACGAACGAACGAACGAACGA"
    )[:141]
    seq = list((base + "A" * 141)[:141])
    for pos in (4, 72, 79):        # known Ψ positions — must be T
        seq[pos - 1] = "T"
    seq[63] = "A"                  # position 64 = A (RENU site base, non-T)
    return "".join(seq)

RNU4_2_MOCK = _make_mock_rnu4_2()


# ─── Pileup DataFrame fixtures ────────────────────────────────────────────────

@pytest.fixture
def clean_pileup():
    """Pileup with no significant deletions — all noise-level rates."""
    n = MINI_REF_LEN
    np.random.seed(42)
    cov  = np.full(n, 100)
    # Background ~6% deletion rate with noise
    dels = np.random.binomial(cov, 0.06)
    df = pd.DataFrame({
        "position":       np.arange(1, n + 1),
        "ref_base":       list(MINI_REF_SEQ),
        "coverage":       cov,
        "deletion_count": dels,
        "insertion_count":np.zeros(n, dtype=int),
        "base_A":         (cov - dels) // 4,
        "base_T":         (cov - dels) // 4,
        "base_C":         (cov - dels) // 4,
        "base_G":         (cov - dels) // 4,
        "base_N":         np.zeros(n, dtype=int),
        "mismatch_count": np.zeros(n, dtype=int),
        "deletion_rate":  dels / cov,
        "insertion_rate": np.zeros(n),
        "mismatch_rate":  np.zeros(n),
        "ins_details":    [{} for _ in range(n)],
    })
    return df


@pytest.fixture
def bidseq_pileup():
    """
    Pileup simulating BID-seq signal.

    Positions 4, 15 (1-based) have high deletion rates (Ψ sites).
    Position 22 has a moderate deletion rate.
    All other T positions have background deletion rates.
    Position 25 is in a homopolymer and has elevated but noisy deletion rate.
    """
    n = MINI_REF_LEN
    np.random.seed(0)
    cov  = np.full(n, 200)
    dels = np.random.binomial(cov, 0.06)  # background

    # Spike in true BID-seq signal at positions 4, 15 (0-indexed: 3, 14)
    dels[3]  = int(cov[3]  * 0.62)   # pos 4  — HIGH_CONFIDENCE
    dels[14] = int(cov[14] * 0.48)   # pos 15 — MODERATE
    dels[21] = int(cov[21] * 0.22)   # pos 22 — LOW

    # Homopolymer trap: position 25 (0-indexed 24) — elevated but should be flagged
    dels[24] = int(cov[24] * 0.18)

    del_rate = dels / cov
    df = pd.DataFrame({
        "position":       np.arange(1, n + 1),
        "ref_base":       list(MINI_REF_SEQ),
        "coverage":       cov,
        "deletion_count": dels,
        "insertion_count":np.zeros(n, dtype=int),
        "base_A":         (cov - dels) // 4,
        "base_T":         (cov - dels) // 4,
        "base_C":         (cov - dels) // 4,
        "base_G":         (cov - dels) // 4,
        "base_N":         np.zeros(n, dtype=int),
        "mismatch_count": np.zeros(n, dtype=int),
        "deletion_rate":  del_rate,
        "insertion_rate": np.zeros(n),
        "mismatch_rate":  np.zeros(n),
        "ins_details":    [{} for _ in range(n)],
    })
    return df


@pytest.fixture
def control_pileup():
    """Control (untreated) pileup with uniform background deletion rates."""
    n = MINI_REF_LEN
    np.random.seed(99)
    cov  = np.full(n, 200)
    dels = np.random.binomial(cov, 0.06)
    del_rate = dels / cov
    df = pd.DataFrame({
        "position":       np.arange(1, n + 1),
        "ref_base":       list(MINI_REF_SEQ),
        "coverage":       cov,
        "deletion_count": dels,
        "insertion_count":np.zeros(n, dtype=int),
        "base_A":         (cov - dels) // 4,
        "base_T":         (cov - dels) // 4,
        "base_C":         (cov - dels) // 4,
        "base_G":         (cov - dels) // 4,
        "base_N":         np.zeros(n, dtype=int),
        "mismatch_count": np.zeros(n, dtype=int),
        "deletion_rate":  del_rate,
        "insertion_rate": np.zeros(n),
        "mismatch_rate":  np.zeros(n),
        "ins_details":    [{} for _ in range(n)],
    })
    return df


@pytest.fixture
def rnu4_2_pileup():
    """
    141-position pileup for the RNU4-2 mock sequence.
    Positions 4, 72, 79 (1-based) are spiked with BID-seq signal.
    Position 64 has insertion data for RENU tracking tests.
    """
    n = 141
    np.random.seed(7)
    cov  = np.full(n, 500)
    dels = np.random.binomial(cov, 0.06)

    # Spike known Ψ sites
    dels[3]  = int(cov[3]  * 0.58)   # pos 4
    dels[71] = int(cov[71] * 0.51)   # pos 72
    dels[78] = int(cov[78] * 0.45)   # pos 79

    ins_details = [{} for _ in range(n)]
    # RENU site: position 64 (0-indexed 63) — heterozygous-like insertion
    ins_details[63] = {"T": 125}   # ~125/500 = 25% VAF (het-ish)
    ins_count = np.zeros(n, dtype=int)
    ins_count[63] = 125

    del_rate = dels / cov
    ins_rate = ins_count / cov

    seq = list(RNU4_2_MOCK)
    df = pd.DataFrame({
        "position":       np.arange(1, n + 1),
        "ref_base":       seq,
        "coverage":       cov,
        "deletion_count": dels,
        "insertion_count":ins_count,
        "base_A":         (cov - dels) // 4,
        "base_T":         (cov - dels) // 4,
        "base_C":         (cov - dels) // 4,
        "base_G":         (cov - dels) // 4,
        "base_N":         np.zeros(n, dtype=int),
        "mismatch_count": np.zeros(n, dtype=int),
        "deletion_rate":  del_rate,
        "insertion_rate": ins_rate,
        "mismatch_rate":  np.zeros(n),
        "ins_details":    ins_details,
    })
    return df


@pytest.fixture
def low_coverage_pileup():
    """Pileup with very low coverage — tests coverage filters."""
    n = MINI_REF_LEN
    cov  = np.full(n, 5)   # below min_coverage
    dels = np.zeros(n, dtype=int)
    df = pd.DataFrame({
        "position":       np.arange(1, n + 1),
        "ref_base":       list(MINI_REF_SEQ),
        "coverage":       cov,
        "deletion_count": dels,
        "insertion_count":np.zeros(n, dtype=int),
        "base_A": cov // 4, "base_T": cov // 4,
        "base_C": cov // 4, "base_G": cov // 4,
        "base_N": np.zeros(n, dtype=int),
        "mismatch_count": np.zeros(n, dtype=int),
        "deletion_rate":  dels / np.where(cov > 0, cov, 1),
        "insertion_rate": np.zeros(n),
        "mismatch_rate":  np.zeros(n),
        "ins_details":    [{} for _ in range(n)],
    })
    return df
