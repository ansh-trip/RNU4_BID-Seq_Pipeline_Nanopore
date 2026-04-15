"""
pileup.py — Position-resolved pileup statistics from a sorted BAM.

Core output
-----------
A pandas DataFrame with one row per reference position, columns:
  position          int   1-based reference position
  ref_base          str   nucleotide in reference (DNA uppercase)
  coverage          int   total reads spanning this position
  deletion_count    int   reads with a deletion AT this position
  insertion_count   int   reads with an insertion AFTER this position
  base_A/T/C/G/N    int   base call counts
  mismatch_count    int   reads with any base mismatch (vs ref)
  deletion_rate     float deletion_count / coverage
  insertion_rate    float insertion_count / coverage
  mismatch_rate     float mismatch_count / coverage
  ins_details       dict  {inserted_sequence: count, ...}

BID-seq deletion semantics
--------------------------
A BID-seq deletion means the reverse transcriptase skipped the pseudouridine
position.  In pysam pileup:
  - pileup_read.is_del == True  →  this read has a deletion at this column
  - pileup_read.indel > 0       →  this read has an INSERTION starting AFTER
                                   this column (used for RENU tracker)
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pysam
from loguru import logger


# ─── Main pileup function ────────────────────────────────────────────────────

def compute_pileup(
    bam_path: str | Path,
    ref_name: str,
    ref_sequence: str,
    *,
    min_mapq: int = 10,
    min_base_quality: int = 0,
    max_depth: int = 1_000_000,
    include_duplicates: bool = True,
    count_spanning_deletions: bool = True,
) -> pd.DataFrame:
    """
    Compute a full per-position pileup for *ref_name* in *bam_path*.

    Parameters
    ----------
    bam_path:      Sorted, indexed BAM file.
    ref_name:      Contig name in the BAM that corresponds to RNU4-2.
    ref_sequence:  Reference DNA string (uppercase, same length as contig).
    min_mapq:      Skip reads with MAPQ < this value.
    min_base_quality: Ignored bases below this Phred score (set 0 for nanopore).
    max_depth:     pysam stepper depth cap.
    include_duplicates: Include PCR-duplicate reads.
    count_spanning_deletions: If True, count long-range deletions spanning a
                  position (CIGAR D).  If False, only count D at column.

    Returns
    -------
    pd.DataFrame with columns described in module docstring.
    """
    bam_path = Path(bam_path)
    ref_len  = len(ref_sequence)

    # Pre-allocate result arrays (1 entry per reference position, 1-based index)
    cov     = np.zeros(ref_len, dtype=np.int64)
    n_del   = np.zeros(ref_len, dtype=np.int64)
    n_ins   = np.zeros(ref_len, dtype=np.int64)
    n_A     = np.zeros(ref_len, dtype=np.int64)
    n_T     = np.zeros(ref_len, dtype=np.int64)
    n_C     = np.zeros(ref_len, dtype=np.int64)
    n_G     = np.zeros(ref_len, dtype=np.int64)
    n_N     = np.zeros(ref_len, dtype=np.int64)
    # Per-position insertion detail: list of dicts
    ins_details: List[Dict] = [{} for _ in range(ref_len)]

    flag_filter = pysam.FSECONDARY | pysam.FSUPPLEMENTARY
    if not include_duplicates:
        flag_filter |= pysam.FDUP

    logger.info(f"Computing pileup for {ref_name} (length={ref_len}) …")

    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        for col in bam.pileup(
            ref_name,
            start=0,
            stop=ref_len,
            min_mapping_quality=min_mapq,
            min_base_quality=min_base_quality,
            max_depth=max_depth,
            stepper="nofilter",
            ignore_overlaps=False,
        ):
            i = col.reference_pos   # 0-based
            if i >= ref_len:
                continue

            for pr in col.pileups:
                # Skip low-MAPQ and flag-filtered reads
                if pr.alignment.mapping_quality < min_mapq:
                    continue
                if pr.alignment.flag & flag_filter:
                    continue

                cov[i] += 1

                if pr.is_del:
                    # Deletion at this position (RT skipped this base)
                    n_del[i] += 1

                elif pr.is_refskip:
                    # Reference skip (intron in spliced alignment) — not a true deletion
                    pass

                else:
                    # Base present
                    try:
                        qpos = pr.query_position
                        base = pr.alignment.query_sequence[qpos].upper()
                    except (TypeError, IndexError):
                        base = "N"

                    if   base == "A": n_A[i] += 1
                    elif base == "T": n_T[i] += 1
                    elif base == "C": n_C[i] += 1
                    elif base == "G": n_G[i] += 1
                    else:             n_N[i] += 1

                    # Insertion AFTER this position
                    if pr.indel > 0:
                        n_ins[i] += 1
                        ins_seq = _get_insertion_seq(pr)
                        if ins_seq:
                            ins_details[i][ins_seq] = ins_details[i].get(ins_seq, 0) + 1

    # Build DataFrame
    positions  = np.arange(1, ref_len + 1, dtype=np.int64)
    ref_bases  = np.array(list(ref_sequence.upper()), dtype=object)
    del_rate   = np.where(cov > 0, n_del / cov, np.nan)
    ins_rate   = np.where(cov > 0, n_ins / cov, np.nan)
    mismatch   = _compute_mismatch(ref_bases, n_A, n_T, n_C, n_G, cov, n_del)
    mm_rate    = np.where(cov > 0, mismatch / cov, np.nan)

    df = pd.DataFrame({
        "position":       positions,
        "ref_base":       ref_bases,
        "coverage":       cov,
        "deletion_count": n_del,
        "insertion_count": n_ins,
        "base_A":         n_A,
        "base_T":         n_T,
        "base_C":         n_C,
        "base_G":         n_G,
        "base_N":         n_N,
        "mismatch_count": mismatch,
        "deletion_rate":  del_rate,
        "insertion_rate": ins_rate,
        "mismatch_rate":  mm_rate,
        "ins_details":    ins_details,
    })

    total_cov = int(np.median(cov[cov > 0])) if np.any(cov > 0) else 0
    logger.success(
        f"Pileup complete. Positions: {ref_len}. "
        f"Median coverage: {total_cov}×. "
        f"Mean deletion rate: {np.nanmean(del_rate):.3f}"
    )
    return df


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _get_insertion_seq(pr: pysam.PileupRead) -> Optional[str]:
    """Extract the inserted sequence from a pileup read with indel > 0."""
    try:
        qstart = pr.query_position + 1       # base AFTER current column
        qend   = qstart + pr.indel
        return pr.alignment.query_sequence[qstart:qend].upper()
    except (TypeError, IndexError):
        return None


def _compute_mismatch(
    ref_bases: np.ndarray,
    n_A, n_T, n_C, n_G, cov, n_del
) -> np.ndarray:
    """
    Count reads with a base mismatch relative to reference.

    A mismatch is a read base that differs from the reference.  Deletions are
    excluded (they are handled separately as BID-seq signal).
    """
    mismatch = np.zeros(len(ref_bases), dtype=np.int64)
    for i, rbase in enumerate(ref_bases):
        total_bases = n_A[i] + n_T[i] + n_C[i] + n_G[i]
        if rbase == "A": mismatch[i] = total_bases - n_A[i]
        elif rbase == "T": mismatch[i] = total_bases - n_T[i]
        elif rbase == "C": mismatch[i] = total_bases - n_C[i]
        elif rbase == "G": mismatch[i] = total_bases - n_G[i]
        else: mismatch[i] = 0
    return np.maximum(mismatch, 0)


# ─── Merge pileups (treated + control) ───────────────────────────────────────

def merge_pileups(treated_df: pd.DataFrame, control_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join treated and control pileup DataFrames on position.

    All treated columns get suffix '_t', control columns '_c'.
    """
    cols_to_merge = [
        "position",
        "coverage", "deletion_count", "deletion_rate",
        "insertion_count", "insertion_rate",
        "mismatch_count", "mismatch_rate",
        "ins_details",
    ]
    t = treated_df[cols_to_merge].copy()
    c = control_df[cols_to_merge].copy()

    merged = t.merge(c, on="position", suffixes=("_t", "_c"), how="outer")

    # Carry forward ref_base and position
    if "ref_base" in treated_df.columns:
        ref_map = dict(zip(treated_df["position"], treated_df["ref_base"]))
        merged["ref_base"] = merged["position"].map(ref_map)

    return merged.fillna(0)


# ─── Pileup serialisation ─────────────────────────────────────────────────────

def save_pileup_tsv(df: pd.DataFrame, path: str | Path) -> None:
    """Save pileup to a TSV, serialising ins_details as JSON strings."""
    import json
    out = df.copy()
    if "ins_details" in out.columns:
        out["ins_details"] = out["ins_details"].apply(
            lambda x: json.dumps(x) if isinstance(x, dict) else "{}"
        )
    out.to_csv(path, sep="\t", index=False, float_format="%.6f")
    logger.debug(f"Pileup TSV written: {path}")


def load_pileup_tsv(path: str | Path) -> pd.DataFrame:
    """Load a pileup TSV written by save_pileup_tsv."""
    import json
    df = pd.read_csv(path, sep="\t")
    if "ins_details" in df.columns:
        df["ins_details"] = df["ins_details"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else {}
        )
    return df
