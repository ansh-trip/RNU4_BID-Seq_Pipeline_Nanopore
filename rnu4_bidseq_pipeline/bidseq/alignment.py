"""
alignment.py — Align nanopore reads to the RNU4-2 reference using minimap2.

Design notes
------------
* Uses minimap2 with the `map-ont` preset (cDNA after BID-seq RT, default) or
  `splice -k14` for direct-RNA mode.
* The `--cs` flag records the per-base difference string in the CS tag, which
  enables precise identification of deletions vs mismatches vs insertions in
  downstream analysis.
* `-Y` uses soft-clipping for supplementary alignments (keeps full read seq).
* Output BAM is coordinate-sorted and indexed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

from loguru import logger

from bidseq.utils import run_command, ensure_dir


# ─── Preset table ────────────────────────────────────────────────────────────

MINIMAP2_PRESETS = {
    "cdna":       "map-ont",
    "direct_rna": "splice -k14",
    "map-ont":    "map-ont",
    "splice":     "splice -k14",
}


# ─── Index reference ─────────────────────────────────────────────────────────

def index_reference(ref_fasta: str | Path) -> Path:
    """Build minimap2 MMI index if it doesn't already exist."""
    ref_fasta = Path(ref_fasta)
    mmi = ref_fasta.with_suffix(".mmi")
    if mmi.exists():
        logger.debug(f"Minimap2 index already exists: {mmi}")
        return mmi

    logger.info(f"Building minimap2 index for {ref_fasta.name} …")
    run_command(
        ["minimap2", "-d", str(mmi), str(ref_fasta)],
        label="minimap2-index",
    )
    return mmi


def samtools_faidx(ref_fasta: str | Path) -> None:
    """Create samtools .fai index for the reference FASTA."""
    fai = Path(str(ref_fasta) + ".fai")
    if not fai.exists():
        run_command(["samtools", "faidx", str(ref_fasta)], label="samtools faidx")


# ─── Align ───────────────────────────────────────────────────────────────────

def align_reads(
    fastq: str | Path,
    ref_fasta: str | Path,
    output_bam: str | Path,
    *,
    mode: str = "cdna",
    threads: int = 4,
    min_mapq: int = 10,
    extra_flags: str = "",
    sample_name: Optional[str] = None,
) -> Dict:
    """
    Align nanopore reads to the RNU4-2 reference.

    Steps:
      1. minimap2 → SAM (piped)
      2. samtools view  (filter MAPQ, convert to BAM)
      3. samtools sort  (coordinate sort)
      4. samtools index (BAI index)

    Returns a dict with alignment stats from `samtools flagstat`.
    """
    fastq      = Path(fastq)
    ref_fasta  = Path(ref_fasta)
    output_bam = Path(output_bam)
    ensure_dir(output_bam.parent)

    preset = MINIMAP2_PRESETS.get(mode, mode)
    sm_tag = sample_name or fastq.stem

    # Build minimap2 command
    mm2_cmd = (
        f"minimap2 -ax {preset} --cs -Y "
        f"-t {threads} "
        f"-R '@RG\\tID:{sm_tag}\\tSM:{sm_tag}\\tPL:ONT' "
        f"{extra_flags} "
        f"{ref_fasta} {fastq}"
    )

    # Build samtools pipeline
    unsorted_bam = output_bam.with_suffix(".unsorted.bam")
    view_cmd = f"samtools view -bS -q {min_mapq} -@ {threads} -o {unsorted_bam}"
    sort_cmd = f"samtools sort -@ {threads} -o {output_bam} {unsorted_bam}"
    index_cmd = ["samtools", "index", str(output_bam)]

    logger.info(f"Aligning {fastq.name} → {output_bam.name} [mode={mode}, threads={threads}]")

    # Run alignment pipeline
    run_command(f"{mm2_cmd} | {view_cmd}", label="minimap2|samtools view")
    run_command(sort_cmd, label="samtools sort")
    os.remove(unsorted_bam)
    run_command(index_cmd, label="samtools index")

    # Collect stats
    stats = get_flagstat(output_bam)
    _log_flagstat(stats, sm_tag)
    return stats


# ─── Flagstat ────────────────────────────────────────────────────────────────

def get_flagstat(bam_path: str | Path) -> Dict:
    """Parse `samtools flagstat` into a dict."""
    result = run_command(
        ["samtools", "flagstat", str(bam_path)],
        capture=True,
        label="samtools flagstat",
    )
    stats: Dict = {}
    for line in result.stdout.splitlines():
        m_total   = _re_match(r"^(\d+) \+ \d+ in total", line)
        m_mapped  = _re_match(r"^(\d+) \+ \d+ mapped \(", line)
        m_primary = _re_match(r"^(\d+) \+ \d+ primary mapped", line)
        if m_total:
            stats["total_reads"] = int(m_total)
        if m_mapped:
            stats["mapped_reads"] = int(m_mapped)
        if m_primary:
            stats["primary_mapped_reads"] = int(m_primary)

    total = stats.get("total_reads", 0)
    mapped = stats.get("mapped_reads", 0)
    stats["mapping_rate"] = mapped / total if total > 0 else 0.0
    return stats


def _re_match(pattern: str, line: str):
    import re
    m = re.match(pattern, line)
    return m.group(1) if m else None


def _log_flagstat(stats: Dict, label: str) -> None:
    rate = stats.get("mapping_rate", 0)
    mapped = stats.get("mapped_reads", 0)
    total = stats.get("total_reads", 0)
    logger.info(
        f"[{label}] Alignment: {mapped:,}/{total:,} reads mapped ({rate:.1%})"
    )
    if rate < 0.50:
        logger.warning(
            f"[{label}] Low mapping rate ({rate:.1%}). "
            "Check that the correct reference and sequencing mode are being used."
        )


# ─── BAM utilities ───────────────────────────────────────────────────────────

def get_bam_header(bam_path: str | Path) -> Dict:
    """Return dict with reference names and lengths from BAM header."""
    import pysam
    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        refs = {
            name: bam.get_reference_length(name)
            for name in bam.references
        }
    return refs


def count_mapped_reads(bam_path: str | Path, ref_name: Optional[str] = None) -> int:
    """Count primary mapped reads (optionally restricted to ref_name)."""
    import pysam
    with pysam.AlignmentFile(str(bam_path), "rb") as bam:
        if ref_name:
            return bam.count(ref_name, read_callback="all")
        return sum(1 for r in bam.fetch() if not r.is_unmapped and not r.is_secondary)
