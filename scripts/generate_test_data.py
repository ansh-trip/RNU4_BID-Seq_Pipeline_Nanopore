#!/usr/bin/env python3
"""
generate_test_data.py — Synthetic test data generator for the BID-seq pipeline.

Generates:
  1. A FASTA file containing a mock RNU4-2 reference (~141 nt)
  2. A treated FASTQ: reads with BID-seq-like deletions at positions 4, 72, 79
  3. A control FASTQ: reads with only background nanopore error rates
  4. A RENU variant FASTQ: treated reads plus a T insertion at position 64

All reads are synthetic; no real genomic data is used.

Usage
-----
  python scripts/generate_test_data.py --output tests/data/

The generated FASTQ files can be used for:
  - Verifying the pipeline runs end-to-end
  - Benchmarking calling accuracy
  - CI testing (requires minimap2 + samtools)
"""

from __future__ import annotations

import argparse
import gzip
import random
from pathlib import Path

import numpy as np


# ─── Configuration ────────────────────────────────────────────────────────────

# RNU4-2 mock: 141-nt sequence (realistic base composition)
# T at positions (1-based): 4,8,12,15,19,22,26,30,35,40,45,50,55,60,64,72,79,85,90,95,100,105,110,115,120,125,130,135,140
RNU4_2_MOCK_SEQ = (
    "ATGCTGTGTCCAGAAGATCGCAGCCAAGCGGTGCATGTGCTGCAGATGCAGCAAACAGAT"
    "GCCAGAGCAAAATGCAAGGAATGTCATGGGAAAAGAGCACTGTCTGGTTGCTCTTGGAAAG"
    "CCCAAGTCAAGCCCAGAGTCGCTGCG"
)
# Pad/trim to exactly 141
RNU4_2_MOCK_SEQ = (RNU4_2_MOCK_SEQ + "A" * 141)[:141]
REF_NAME = "RNU4-2"

# BID-seq deletion rates at each position
PSI_SITES = {4: 0.60, 72: 0.50, 79: 0.45}    # true Ψ sites — high deletion
BACKGROUND_DEL_RATE = 0.06                      # nanopore noise floor
RENU_INS_POS = 64                               # T insertion position (1-based)
RENU_VAF     = 0.45                             # ~het; fraction of reads with T-ins

N_READS = 3000    # reads per FASTQ
READ_LEN_MEAN = 141
READ_LEN_SD   = 5
PHRED_QUAL = 15   # a realistic nanopore quality score


def phred_char(q: int) -> str:
    return chr(q + 33)


def random_base(exclude: str = "") -> str:
    bases = [b for b in "ACGT" if b != exclude.upper()]
    return random.choice(bases)


def simulate_read(
    ref_seq: str,
    del_rates: dict,           # {0-based_pos: rate}
    bg_del_rate: float = BACKGROUND_DEL_RATE,
    insertion_pos: int = -1,   # 0-based; -1 = no insertion
    insertion_seq: str = "T",
) -> tuple[str, str]:
    """
    Simulate a nanopore read from ref_seq with:
      - BID-seq deletions at del_rates positions
      - Background deletions at all other positions
      - Optional insertion at insertion_pos (for RENU)
      - Realistic substitution errors

    Returns (read_sequence, quality_string).
    """
    bases = []
    quals = []
    ref_len = len(ref_seq)

    i = 0
    while i < ref_len:
        ref_base = ref_seq[i].upper()

        # Insertion before this position (RENU)
        if i == insertion_pos and random.random() < RENU_VAF:
            for ins_b in insertion_seq:
                bases.append(ins_b)
                quals.append(phred_char(PHRED_QUAL))

        # Deletion at this position?
        del_rate = del_rates.get(i, bg_del_rate)
        if random.random() < del_rate:
            i += 1
            continue   # skip (deletion)

        # Substitution error (nanopore ~2% sub rate)
        if random.random() < 0.02:
            bases.append(random_base(exclude=ref_base))
            quals.append(phred_char(PHRED_QUAL - 5))
        else:
            bases.append(ref_base)
            quals.append(phred_char(PHRED_QUAL))

        i += 1

    return "".join(bases), "".join(quals)


def write_fastq(reads: list, path: Path, gzipped: bool = False) -> None:
    opener = gzip.open if gzipped else open
    mode   = "wt" if gzipped else "w"
    with opener(path, mode) as f:
        for i, (seq, qual) in enumerate(reads):
            f.write(f"@read_{i:06d}\n{seq}\n+\n{qual}\n")


def generate(output_dir: Path, seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_seq  = RNU4_2_MOCK_SEQ
    del_rates_treated = {pos - 1: rate for pos, rate in PSI_SITES.items()}

    # ── Reference FASTA ───────────────────────────────────────────────────────
    ref_path = output_dir / "rnu4_2_test.fa"
    with open(ref_path, "w") as f:
        f.write(f">{REF_NAME}\n")
        for i in range(0, len(ref_seq), 60):
            f.write(ref_seq[i:i+60] + "\n")
    print(f"Reference written: {ref_path}")

    # ── Treated FASTQ (BID-seq signal at Ψ sites) ─────────────────────────────
    treated_reads = [
        simulate_read(ref_seq, del_rates_treated)
        for _ in range(N_READS)
    ]
    treated_path = output_dir / "treated_test.fastq.gz"
    write_fastq(treated_reads, treated_path, gzipped=True)
    print(f"Treated FASTQ written: {treated_path}  ({N_READS} reads)")

    # ── Control FASTQ (background only) ───────────────────────────────────────
    control_reads = [
        simulate_read(ref_seq, {})   # no spike — pure background
        for _ in range(N_READS)
    ]
    control_path = output_dir / "control_test.fastq.gz"
    write_fastq(control_reads, control_path, gzipped=True)
    print(f"Control FASTQ written: {control_path}  ({N_READS} reads)")

    # ── RENU variant FASTQ (treated + RENU insertion) ─────────────────────────
    renu_reads = [
        simulate_read(
            ref_seq,
            del_rates_treated,
            insertion_pos = RENU_INS_POS - 1,   # 0-based
            insertion_seq = "T",
        )
        for _ in range(N_READS)
    ]
    renu_path = output_dir / "renu_het_test.fastq.gz"
    write_fastq(renu_reads, renu_path, gzipped=True)
    print(f"RENU variant FASTQ written: {renu_path}  ({N_READS} reads, ~{RENU_VAF:.0%} T-ins)")

    # ── README for test data ───────────────────────────────────────────────────
    readme = output_dir / "README.txt"
    with open(readme, "w") as f:
        f.write(f"""Synthetic BID-seq Test Data
===========================
Reference: {ref_path.name}   ({len(ref_seq)} nt, mock RNU4-2)

Treated FASTQ:  {treated_path.name}
  - {N_READS} reads
  - BID-seq deletion spikes at positions {sorted(PSI_SITES.keys())} (1-based)
  - Background deletion rate: {BACKGROUND_DEL_RATE:.0%}

Control FASTQ:  {control_path.name}
  - {N_READS} reads
  - Background deletion rate only (no Ψ spikes)

RENU FASTQ:     {renu_path.name}
  - {N_READS} reads
  - Same Ψ spikes as treated
  - T insertion at position {RENU_INS_POS} (n.64_65insT) with VAF ~{RENU_VAF:.0%}

Expected calls
--------------
  Treated (single-sample): positions {sorted(PSI_SITES.keys())} should be called
  Treated (paired):        same, with higher specificity
  RENU:                    zygosity HETEROZYGOUS or MOSAIC_OR_LOW_SIGNAL
""")
    print(f"README written: {readme}")
    print("\nDone!  Run the pipeline with:")
    print(f"  bidseq run --treated {treated_path} --control {control_path} --output results/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "-o", default="tests/data", help="Output directory")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()
    generate(Path(args.output), seed=args.seed)


if __name__ == "__main__":
    main()
