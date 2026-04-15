"""
qc.py — Read-level quality control for nanopore FASTQ files.

Runs NanoStat for aggregate statistics, then applies configurable filters
to produce a clean FASTQ for downstream alignment.
"""

from __future__ import annotations

import subprocess
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from bidseq.utils import run_command, ensure_dir, open_maybe_gz


# ─── NanoStat wrapper ─────────────────────────────────────────────────────────

def run_nanostat(
    fastq_path: str | Path,
    output_dir: str | Path,
    threads: int = 4,
) -> Dict:
    """Run NanoStat and return parsed summary as a dict."""
    ensure_dir(output_dir)
    stat_txt = Path(output_dir) / "nanostat.txt"

    run_command(
        [
            "NanoStat",
            "--fastq", str(fastq_path),
            "--outdir", str(output_dir),
            "--name", "nanostat.txt",
            "-t", str(threads),
        ],
        label="NanoStat",
        capture=False,
    )

    return _parse_nanostat(stat_txt) if stat_txt.exists() else {}


def _parse_nanostat(stat_path: Path) -> Dict:
    """Parse NanoStat plain-text output into a Python dict."""
    metrics = {}
    with open(stat_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("General summary"):
                continue
            # Lines look like:  "Mean read length:            1,234.5"
            m = re.match(r"^(.+?):\s+([\d,\.]+)$", line)
            if m:
                key = m.group(1).strip().lower().replace(" ", "_")
                val_str = m.group(2).replace(",", "")
                try:
                    metrics[key] = float(val_str)
                except ValueError:
                    metrics[key] = val_str
    return metrics


# ─── Per-read filtering ───────────────────────────────────────────────────────

def filter_reads(
    fastq_in: str | Path,
    fastq_out: str | Path,
    min_length: int = 50,
    max_length: int = 10_000,
    min_mean_quality: float = 7.0,
) -> Dict:
    """Filter FASTQ by read length and mean quality.

    Uses nanoq when available (fast Rust binary) and falls back to a pure-
    Python implementation.

    Returns a summary dict with counts.
    """
    fastq_in = Path(fastq_in)
    fastq_out = Path(fastq_out)
    ensure_dir(fastq_out.parent)

    # Try nanoq first (very fast)
    try:
        result = _filter_with_nanoq(
            fastq_in, fastq_out, min_length, max_length, min_mean_quality
        )
        logger.info(
            f"nanoq filtered: {result['total_in']} reads in → "
            f"{result['total_out']} reads out "
            f"({result['pass_rate']:.1%} pass)"
        )
        return result
    except FileNotFoundError:
        logger.warning("nanoq not found; using built-in Python filter (slower).")
        return _filter_python(
            fastq_in, fastq_out, min_length, max_length, min_mean_quality
        )


def _filter_with_nanoq(
    fastq_in: Path,
    fastq_out: Path,
    min_length: int,
    max_length: int,
    min_quality: float,
) -> Dict:
    """nanoq-based filtering with JSON stats output."""
    stats_json = fastq_out.with_suffix(".nanoq_stats.json")
    cmd = [
        "nanoq",
        "--input", str(fastq_in),
        "--output", str(fastq_out),
        "--min-len", str(min_length),
        "--max-len", str(max_length),
        "--min-qual", str(min_quality),
        "--json", str(stats_json),
    ]
    run_command(cmd, label="nanoq", capture=True)

    summary = {}
    if stats_json.exists():
        with open(stats_json) as f:
            d = json.load(f)
        summary = {
            "total_in":  d.get("reads", {}).get("input", 0),
            "total_out": d.get("reads", {}).get("output", 0),
        }
    if summary.get("total_in", 0) > 0:
        summary["pass_rate"] = summary["total_out"] / summary["total_in"]
    else:
        summary["pass_rate"] = 0.0
    return summary


def _filter_python(
    fastq_in: Path,
    fastq_out: Path,
    min_length: int,
    max_length: int,
    min_quality: float,
) -> Dict:
    """Pure-Python FASTQ filter (fallback when nanoq absent)."""
    total_in = 0
    total_out = 0

    with open_maybe_gz(fastq_in) as fh_in, open(fastq_out, "w") as fh_out:
        while True:
            name  = fh_in.readline()
            seq   = fh_in.readline()
            plus  = fh_in.readline()
            quals = fh_in.readline()
            if not name:
                break
            total_in += 1
            seq_str   = seq.strip()
            qual_str  = quals.strip()
            read_len  = len(seq_str)
            mean_q    = _mean_phred(qual_str)

            if min_length <= read_len <= max_length and mean_q >= min_quality:
                fh_out.write(name)
                fh_out.write(seq)
                fh_out.write(plus)
                fh_out.write(quals)
                total_out += 1

    return {
        "total_in":  total_in,
        "total_out": total_out,
        "pass_rate": total_out / total_in if total_in > 0 else 0.0,
    }


def _mean_phred(qual_str: str) -> float:
    if not qual_str:
        return 0.0
    return np.mean([ord(c) - 33 for c in qual_str])


# ─── Coverage check ───────────────────────────────────────────────────────────

def check_coverage(
    bam_path: str | Path,
    ref_name: str,
    ref_length: int,
    min_coverage: int = 20,
    min_fraction: float = 0.80,
) -> Dict:
    """Check that sufficient coverage exists across RNU4-2.

    Uses `samtools depth` for speed.

    Returns dict with:
        mean_coverage      float
        median_coverage    float
        fraction_covered   float   (fraction of positions ≥ min_coverage)
        low_coverage_pos   list    (1-based positions below min_coverage)
        pass               bool
    """
    result = run_command(
        ["samtools", "depth", "-a", "-r", ref_name, str(bam_path)],
        capture=True,
        label="samtools depth",
    )

    depths = []
    low_cov_pos = []
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        pos   = int(parts[1])
        depth = int(parts[2])
        depths.append(depth)
        if depth < min_coverage:
            low_cov_pos.append(pos)

    if not depths:
        return {
            "mean_coverage":    0.0,
            "median_coverage":  0.0,
            "fraction_covered": 0.0,
            "low_coverage_pos": list(range(1, ref_length + 1)),
            "pass":             False,
        }

    arr = np.array(depths)
    frac_cov = np.mean(arr >= min_coverage)

    summary = {
        "mean_coverage":    float(np.mean(arr)),
        "median_coverage":  float(np.median(arr)),
        "fraction_covered": float(frac_cov),
        "low_coverage_pos": low_cov_pos,
        "pass":             frac_cov >= min_fraction,
    }

    if not summary["pass"]:
        logger.warning(
            f"Coverage check FAILED: {frac_cov:.1%} of {ref_name} covered at "
            f"≥{min_coverage}×  (need ≥{min_fraction:.0%}). "
            f"Low-coverage positions: {low_cov_pos[:20]}{'...' if len(low_cov_pos) > 20 else ''}"
        )
    else:
        logger.success(
            f"Coverage check passed: {frac_cov:.1%} of {ref_name} covered at "
            f"≥{min_coverage}× (mean {summary['mean_coverage']:.0f}×)"
        )

    return summary


# ─── QC report summary ────────────────────────────────────────────────────────

def build_qc_summary(
    nanostat_metrics: Dict,
    filter_summary: Dict,
    coverage_summary: Dict,
    sample_name: str = "sample",
) -> Dict:
    """Consolidate all QC outputs into a single summary dict."""
    return {
        "sample": sample_name,
        "nanostat": nanostat_metrics,
        "read_filtering": filter_summary,
        "coverage": coverage_summary,
        "overall_pass": filter_summary.get("pass_rate", 0) > 0 and coverage_summary.get("pass", False),
    }
