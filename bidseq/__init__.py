"""
rnu4-bidseq-pipeline
====================

Nanopore-adapted BID-seq pipeline for:
  - Quantitative pseudouridine (Ψ) mapping in RNU4-2 / U4 snRNA
  - Discrimination of real BID-seq signal from nanopore sequencing noise
  - Detection and quantification of the RENU-syndrome n.64_65insT variant

BID-seq principle
-----------------
Bisulfite treatment of RNA selectively reacts with pseudouridine (Ψ) to form
a stable Ψ-bisulfite adduct.  When reverse transcriptase encounters this
adduct it stalls and skips the modified nucleotide, producing a characteristic
deletion in the cDNA.  BID-seq exploits this: high deletion rates at a given
reference position indicate Ψ occupancy.

Nanopore adaptation
-------------------
Nanopore cDNA sequencing has a higher background deletion rate (~5–8 % for
R9.4.1 chemistry, ~2–4 % for R10.4.1) compared with Illumina (~0.1 %).
This pipeline corrects for this using:

  1. Paired-sample Fisher's exact test (treated vs untreated control)
  2. Single-sample binomial test against a position-matched background model
  3. Absolute and fold-change filters calibrated to the nanopore noise floor
  4. Homopolymer flagging (elevated nanopore indel error near poly-T/poly-A)

Usage (command line)
--------------------
  bidseq run --treated sample.fastq --output results/
  bidseq run --treated treated.fastq --control ctrl.fastq --output results/
  bidseq fetch-reference --output ref/rnu4_2.fa
  bidseq qc sample.fastq
"""

__version__ = "1.0.0"
__all__ = [
    "qc",
    "alignment",
    "pileup",
    "deletion_stats",
    "pseudoU_caller",
    "renu_tracker",
    "report",
    "reference",
    "utils",
]
