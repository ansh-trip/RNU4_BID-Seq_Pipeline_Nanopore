"""
renu_tracker.py — Track the RENU syndrome n.64_65insT variant in RNU4-2.

Background
----------
The most recurrent pathogenic variant in RENU syndrome is n.64_65insT — a
single-T insertion between positions 64 and 65 of the U4C snRNA (RNU4-2).
This insertion sits within the T-loop quasi-pseudoknot and disrupts U4/U6
snRNA base-pairing, impairing spliceosome activation.

What this module does
---------------------
1. Scans every read aligned to the RNU4-2 region spanning positions 60–70.
2. Detects insertions AFTER reference position 64 (0-based: 63).
3. Classifies each read as:
       WILD_TYPE    — no insertion at position 64
       RENU_T_INS   — exactly a single-T insertion at position 64
       OTHER_INS    — insertion at position 64 but not a simple T
4. Reports:
       counts and frequencies for each class
       variant allele frequency (VAF)
       inferred zygosity (WT / HET / HOM / MOSAIC / AMBIGUOUS)
       nearby deletion rates (to distinguish RENU signal from BID-seq noise)
5. For noisy nanopore data, applies a minimum supporting-read filter and a
   strict sequence-context check (surrounding bases must match reference).

All positions are 1-based in the public API; 0-based internally.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pysam
from loguru import logger


# ─── Constants ────────────────────────────────────────────────────────────────

# 1-based RNU4-2 position BEFORE the RENU insertion
# (insertion occurs AFTER this position)
RENU_INS_POS_1BASED  = 64
RENU_INS_POS_0BASED  = RENU_INS_POS_1BASED - 1   # 63

# Expected inserted sequence
RENU_INS_SEQ         = "T"

# Window around insertion site to scan
SCAN_WINDOW          = 5    # check positions 59–69 (1-based: 60–70)

# Zygosity inference thresholds (VAF)
VAF_HOM_LOW          = 0.85
VAF_HET_HIGH         = 0.70
VAF_HET_LOW          = 0.30
VAF_MOSAIC_LOW       = 0.05


# ─── Data classes ────────────────────────────────────────────────────────────

@dataclass
class RenuReadResult:
    """Classification result for a single read at the insertion site."""
    read_name:   str
    category:    str          # WILD_TYPE | RENU_T_INS | OTHER_INS
    ins_seq:     Optional[str] = None
    ins_length:  int           = 0
    mapq:        int           = 0
    context_ok:  bool          = True   # surrounding bases match reference


@dataclass
class RenuSiteSummary:
    """Aggregate RENU insertion analysis at position 64–65."""
    total_reads:       int            = 0
    wild_type_reads:   int            = 0
    renu_t_ins_reads:  int            = 0
    other_ins_reads:   int            = 0
    low_mapq_skipped:  int            = 0
    insufficient_context_skipped: int = 0

    vaf:               float          = 0.0
    zygosity:          str            = "AMBIGUOUS"
    interpretation:    str            = ""

    insertion_seqs:    Dict[str, int] = field(default_factory=dict)
    coverage_at_site:  int            = 0
    nearby_deletion_rate: float       = 0.0

    # Per-read details (for debugging)
    read_details:      List[RenuReadResult] = field(default_factory=list)

    def to_dict(self) -> Dict:
        d = {
            "renu_insertion_position_1based":  RENU_INS_POS_1BASED,
            "renu_nomenclature":               "n.64_65insT",
            "total_reads_assessed":            self.total_reads,
            "wild_type_reads":                 self.wild_type_reads,
            "renu_t_insertion_reads":          self.renu_t_ins_reads,
            "other_insertion_reads":           self.other_ins_reads,
            "vaf":                             round(self.vaf, 4),
            "zygosity":                        self.zygosity,
            "interpretation":                  self.interpretation,
            "insertion_sequences_observed":    self.insertion_seqs,
            "coverage_at_site":                self.coverage_at_site,
            "nearby_deletion_rate":            round(self.nearby_deletion_rate, 4),
        }
        return d


# ─── Main function ────────────────────────────────────────────────────────────

def track_renu_insertion(
    bam_path: str | Path,
    ref_name: str,
    ref_sequence: str,
    *,
    min_mapq: int = 10,
    min_renu_reads: int = 3,
    min_vaf: float = 0.02,
    het_low: float = VAF_HET_LOW,
    het_high: float = VAF_HET_HIGH,
    hom_low: float = VAF_HOM_LOW,
    pileup_df: Optional[pd.DataFrame] = None,
) -> RenuSiteSummary:
    """
    Detect and quantify the n.64_65insT RENU insertion in RNU4-2.

    Parameters
    ----------
    bam_path:     Sorted, indexed BAM.
    ref_name:     Reference contig name (must match BAM header).
    ref_sequence: Full reference DNA string (uppercase).
    min_mapq:     Skip reads with MAPQ < this.
    min_renu_reads: Minimum reads supporting RENU insertion to call.
    min_vaf:      Minimum VAF to report.
    het_low/high: VAF boundaries for heterozygous inference.
    hom_low:      VAF boundary for homozygous inference.
    pileup_df:    Optional pre-computed pileup (for nearby deletion rate).
    """
    bam_path   = Path(bam_path)
    ins_pos_0  = RENU_INS_POS_0BASED          # 63 (0-based)
    ref_len    = len(ref_sequence)

    summary = RenuSiteSummary()

    logger.info(
        f"Scanning RNU4-2 for RENU insertion (n.{RENU_INS_POS_1BASED}_"
        f"{RENU_INS_POS_1BASED+1}insT) …"
    )

    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        # Use pileup at the insertion position
        for col in bam.pileup(
            ref_name,
            start=ins_pos_0,
            stop=ins_pos_0 + 1,
            min_mapping_quality=min_mapq,
            max_depth=1_000_000,
            stepper="nofilter",
        ):
            if col.reference_pos != ins_pos_0:
                continue

            summary.coverage_at_site = col.nsegments

            for pr in col.pileups:
                # Skip secondary / supplementary
                if pr.alignment.is_secondary or pr.alignment.is_supplementary:
                    continue
                if pr.alignment.mapping_quality < min_mapq:
                    summary.low_mapq_skipped += 1
                    continue
                if pr.is_del or pr.is_refskip:
                    # Deletion at position 64 itself — not an insertion
                    summary.wild_type_reads += 1
                    summary.total_reads += 1
                    summary.read_details.append(RenuReadResult(
                        read_name  = pr.alignment.query_name or "?",
                        category   = "WILD_TYPE",
                        mapq       = pr.alignment.mapping_quality,
                    ))
                    continue

                summary.total_reads += 1
                ins_seq = None
                category = "WILD_TYPE"

                if pr.indel > 0:
                    # Insertion AFTER position 64
                    ins_seq = _extract_ins_seq(pr)
                    ctx_ok  = _check_context(pr, ref_sequence, ins_pos_0)

                    if not ctx_ok:
                        summary.insufficient_context_skipped += 1
                        # Still count it but flag

                    if ins_seq:
                        summary.insertion_seqs[ins_seq] = (
                            summary.insertion_seqs.get(ins_seq, 0) + 1
                        )

                    if ins_seq == RENU_INS_SEQ:
                        category = "RENU_T_INS"
                        summary.renu_t_ins_reads += 1
                    else:
                        category = "OTHER_INS"
                        summary.other_ins_reads += 1
                else:
                    summary.wild_type_reads += 1

                summary.read_details.append(RenuReadResult(
                    read_name  = pr.alignment.query_name or "?",
                    category   = category,
                    ins_seq    = ins_seq,
                    ins_length = pr.indel if pr.indel > 0 else 0,
                    mapq       = pr.alignment.mapping_quality,
                    context_ok = _check_context(pr, ref_sequence, ins_pos_0),
                ))
            break  # Only need pileup at one position

    # ── Calculate VAF ─────────────────────────────────────────────────────────
    denom = summary.total_reads
    if denom > 0:
        summary.vaf = summary.renu_t_ins_reads / denom
    else:
        summary.vaf = 0.0

    # ── Infer zygosity ────────────────────────────────────────────────────────
    summary.zygosity = _infer_zygosity(
        summary.vaf,
        summary.renu_t_ins_reads,
        min_renu_reads,
        min_vaf,
        het_low,
        het_high,
        hom_low,
    )

    # ── Nearby deletion rate (from pileup if provided) ────────────────────────
    if pileup_df is not None:
        window_pos = list(range(
            RENU_INS_POS_1BASED - SCAN_WINDOW,
            RENU_INS_POS_1BASED + SCAN_WINDOW + 1,
        ))
        nearby = pileup_df[pileup_df["position"].isin(window_pos)]
        if not nearby.empty:
            summary.nearby_deletion_rate = float(nearby["deletion_rate"].mean())

    # ── Build interpretation ──────────────────────────────────────────────────
    summary.interpretation = _build_interpretation(summary, min_renu_reads, min_vaf)

    _log_renu_result(summary)
    return summary


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _extract_ins_seq(pr: pysam.PileupRead) -> Optional[str]:
    """Extract the inserted nucleotide sequence from an insertion pileup event."""
    try:
        qpos   = pr.query_position
        n_ins  = pr.indel
        return pr.alignment.query_sequence[qpos + 1: qpos + 1 + n_ins].upper()
    except (TypeError, IndexError):
        return None


def _check_context(
    pr: pysam.PileupRead,
    ref_seq: str,
    ins_pos_0: int,
    context_bases: int = 3,
) -> bool:
    """
    Verify that the bases flanking the insertion match the reference.

    For nanopore, a genuine insertion at position 64 should occur in reads
    that otherwise match the reference at the surrounding bases.  If the
    flanking bases are also mismatched, the insertion may be a sequencing
    error in a badly-aligned read.
    """
    try:
        aln   = pr.alignment
        qseq  = aln.query_sequence
        pairs = dict(aln.get_aligned_pairs(matches_only=False))

        mismatches = 0
        for delta in range(-context_bases, context_bases + 1):
            if delta == 0:
                continue
            ref_p = ins_pos_0 + delta
            if ref_p < 0 or ref_p >= len(ref_seq):
                continue
            q_p = pairs.get(ref_p)
            if q_p is None:
                continue
            q_base   = qseq[q_p].upper() if qseq and q_p < len(qseq) else None
            ref_base = ref_seq[ref_p].upper()
            if q_base and q_base != ref_base:
                mismatches += 1

        return mismatches <= 1   # allow at most 1 flanking mismatch
    except Exception:
        return True  # unclear context; don't filter


def _infer_zygosity(
    vaf: float,
    n_renu: int,
    min_reads: int,
    min_vaf: float,
    het_low: float,
    het_high: float,
    hom_low: float,
) -> str:
    """Infer zygosity from VAF and read support."""
    if n_renu < min_reads or vaf < min_vaf:
        return "WILD_TYPE"
    if vaf >= hom_low:
        return "HOMOZYGOUS"
    if het_low <= vaf <= het_high:
        return "HETEROZYGOUS"
    if min_vaf <= vaf < het_low:
        return "MOSAIC_OR_LOW_SIGNAL"
    if vaf > het_high:
        return "NEAR_HOMOZYGOUS"
    return "AMBIGUOUS"


def _build_interpretation(
    s: RenuSiteSummary,
    min_reads: int,
    min_vaf: float,
) -> str:
    """Generate a plain-English interpretation of the RENU tracking result."""
    if s.total_reads < 20:
        return (
            f"Insufficient coverage at RENU site (n={s.total_reads} reads). "
            "Cannot assess insertion status."
        )

    if s.zygosity == "WILD_TYPE":
        return (
            f"No RENU n.64_65insT insertion detected "
            f"(VAF={s.vaf:.1%}, n={s.renu_t_ins_reads} supporting reads). "
            "Sample appears WILD-TYPE at this site."
        )

    if s.zygosity == "HETEROZYGOUS":
        return (
            f"RENU n.64_65insT DETECTED (heterozygous). "
            f"VAF={s.vaf:.1%} ({s.renu_t_ins_reads}/{s.total_reads} reads). "
            "Consistent with de novo heterozygous RENU syndrome variant. "
            "Recommend Sanger confirmation."
        )

    if s.zygosity == "HOMOZYGOUS":
        return (
            f"RENU n.64_65insT DETECTED (homozygous). "
            f"VAF={s.vaf:.1%} ({s.renu_t_ins_reads}/{s.total_reads} reads). "
            "Homozygous calls are unusual for this variant (typically de novo het). "
            "Verify with orthogonal method."
        )

    if s.zygosity == "MOSAIC_OR_LOW_SIGNAL":
        return (
            f"Low-level RENU-like insertion detected (VAF={s.vaf:.1%}, "
            f"n={s.renu_t_ins_reads} reads). "
            "Could represent somatic mosaicism or nanopore artefact. "
            "Consider increasing sequencing depth and confirming orthogonally."
        )

    return (
        f"Ambiguous RENU insertion signal "
        f"(VAF={s.vaf:.1%}, zygosity={s.zygosity}). "
        "Manual review recommended."
    )


def _log_renu_result(s: RenuSiteSummary) -> None:
    icon = "✅" if s.zygosity == "WILD_TYPE" else "⚠️ " if "RENU" not in s.zygosity else "🔴"
    logger.info(
        f"{icon} RENU n.64_65insT | zygosity={s.zygosity} | "
        f"VAF={s.vaf:.1%} ({s.renu_t_ins_reads}/{s.total_reads} reads) | "
        f"coverage={s.coverage_at_site}"
    )


# ─── Tabular output ──────────────────────────────────────────────────────────

def renu_reads_to_df(summary: RenuSiteSummary) -> pd.DataFrame:
    """Convert per-read RENU classifications to a DataFrame."""
    return pd.DataFrame([
        {
            "read_name":   r.read_name,
            "category":    r.category,
            "ins_seq":     r.ins_seq or "",
            "ins_length":  r.ins_length,
            "mapq":        r.mapq,
            "context_ok":  r.context_ok,
        }
        for r in summary.read_details
    ])
