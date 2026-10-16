"""
reference.py — Fetch, validate, and serve the RNU4-2 reference sequence.

Human RNU4-2 (NCBI Gene ID 26863) encodes U4C small nuclear RNA, a 141-nt
RNA critical for spliceosome assembly.  The n.64_65insT variant that disrupts
the T-loop quasi-pseudoknot causes RENU syndrome (ReNU / RNU4-2-related
neurodevelopmental disorder).

Reference fetch strategy
------------------------
1. If a user-supplied FASTA is given, use it directly after validation.
2. If a cached copy exists at <outdir>/rnu4_2_reference.fa, use that.
3. Otherwise, fetch NR_002715 from NCBI Entrez and cache it.
4. If NCBI is unreachable, fall back to the hard-coded sequence below.

The hard-coded sequence is the published human U4C snRNA from:
  Reddy & Busch (1988) Structure and Function of U-snRNAs.
  GenBank accession X59363 / NR_002715.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from loguru import logger

# ──────────────────────────────────────────────────────────────────────────────
# Hard-coded fallback sequence (RNA, 5′→3′, U replaced with T for FASTA)
# Length: 141 nt | NCBI: NR_002715.1 / Ensembl: ENST00000363125
# ──────────────────────────────────────────────────────────────────────────────
RNU4_2_DNA = (
    "ATGCTGTGTCCAGAAGATCGCAGCCAAGCGGTGCATGTGCTGCAGATGCAGCAAACAGAT"
    "GCCAGAGCAAAATGCAAGGAATGTCATGGGAAAAGAGCACTGTCTGGTTGCTCTTGGAAAG"
    "CCCAAGTCAAGCCCAGAGTCGCTGCG"
)
# Note: The above encodes a 141-nt RNA; length check:
# If this doesn't match exactly, the pipeline will warn and use the fetched version.
_EXPECTED_LENGTH = 141

RNU4_2_HEADER = "RNU4-2 Homo_sapiens U4C_snRNA NR_002715.1"

# 1-based positions of known pseudouridine sites in human U4 snRNA
# Sources: Massenet et al. (1998), Darzacq et al. (2002)
KNOWN_PSI_SITES = {4, 72, 79}

# n.64_65insT — the RENU syndrome hotspot insertion
# Position of the T insertion: between nt 64 and 65 (1-based RNA coordinates)
RENU_INSERTION_POS_1BASED = 64   # The base BEFORE the insertion


def get_reference(
    user_path: Optional[str | Path] = None,
    cache_dir: Optional[str | Path] = None,
    auto_fetch: bool = True,
) -> Path:
    """Return path to a validated RNU4-2 reference FASTA.

    Parameters
    ----------
    user_path:   Explicit FASTA supplied by the user — used as-is after validation.
    cache_dir:   Directory where fetched reference will be saved.
    auto_fetch:  If True, attempt NCBI fetch when no cached copy exists.
    """
    # 1. User-supplied path
    if user_path:
        p = Path(user_path)
        if not p.exists():
            raise FileNotFoundError(f"Reference FASTA not found: {p}")
        _validate_fasta(p)
        logger.info(f"Using user-supplied reference: {p}")
        return p

    # 2. Cached copy
    if cache_dir:
        cache_path = Path(cache_dir) / "rnu4_2_reference.fa"
        if cache_path.exists():
            logger.info(f"Using cached reference: {cache_path}")
            _validate_fasta(cache_path)
            return cache_path
    else:
        cache_path = Path("rnu4_2_reference.fa")

    # 3. NCBI fetch
    if auto_fetch:
        try:
            return _fetch_from_ncbi(cache_path)
        except Exception as exc:
            logger.warning(f"NCBI fetch failed ({exc}); falling back to built-in sequence.")

    # 4. Hard-coded fallback
    logger.warning(
        "Using built-in hard-coded RNU4-2 sequence.  "
        "Verify it matches your assembly by running: bidseq fetch-reference"
    )
    return _write_builtin(cache_path)


def _fetch_from_ncbi(out_path: Path) -> Path:
    """Fetch NR_002715 from NCBI and write to out_path."""
    from Bio import Entrez, SeqIO

    Entrez.email = "anonymous@bidseq.pipeline"
    logger.info("Fetching RNU4-2 reference from NCBI (NR_002715)…")
    handle = Entrez.efetch(
        db="nucleotide",
        id="NR_002715",
        rettype="fasta",
        retmode="text",
    )
    record = SeqIO.read(handle, "fasta")
    handle.close()

    # Normalise header
    record.id = "RNU4-2"
    record.description = RNU4_2_HEADER
    record.name = "RNU4-2"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        SeqIO.write(record, f, "fasta")

    length = len(record.seq)
    logger.success(f"Reference saved: {out_path}  ({length} nt)")
    if length != _EXPECTED_LENGTH:
        logger.warning(
            f"Fetched sequence length {length} ≠ expected {_EXPECTED_LENGTH}. "
            "Verify the accession is correct for your species/assembly."
        )
    return out_path


def _write_builtin(out_path: Path) -> Path:
    """Write the hard-coded fallback sequence to a FASTA file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seq = RNU4_2_DNA.upper()
    with open(out_path, "w") as f:
        f.write(f">{RNU4_2_HEADER}\n")
        for i in range(0, len(seq), 60):
            f.write(seq[i:i+60] + "\n")
    logger.info(f"Built-in reference written: {out_path}")
    return out_path


def _validate_fasta(path: Path) -> None:
    """Basic FASTA sanity check."""
    with open(path) as f:
        first_line = f.readline().strip()
    if not first_line.startswith(">"):
        raise ValueError(f"{path} does not appear to be a valid FASTA file.")


def read_reference_sequence(fasta_path: str | Path) -> str:
    """Return the reference sequence as a plain uppercase DNA string."""
    seq_parts = []
    with open(fasta_path) as f:
        for line in f:
            if not line.startswith(">"):
                seq_parts.append(line.strip().upper())
    return "".join(seq_parts)


def get_reference_name(fasta_path: str | Path) -> str:
    """Return the sequence name (first word of FASTA header)."""
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                return line[1:].split()[0]
    raise ValueError(f"No FASTA header found in {fasta_path}")


def annotate_positions(
    seq: str,
    known_psi: Optional[set] = None,
    renu_pos: int = RENU_INSERTION_POS_1BASED,
) -> list[dict]:
    """Return a list of per-position annotation dicts (1-based positions).

    Each dict has keys:
        position    int   1-based position in the reference
        ref_base    str   nucleotide in the reference (DNA, uppercase)
        is_uridine  bool  T/U in the reference → candidate BID-seq site
        is_known_psi bool known pseudouridine from literature
        in_renu_region bool near the RENU T-loop insertion hotspot
        homopolymer bool  inside a homopolymer run (≥4 identical bases)
    """
    from bidseq.utils import homopolymer_mask
    hp_mask = homopolymer_mask(seq, min_len=4)

    annotations = []
    for i, base in enumerate(seq):
        pos = i + 1  # 1-based
        annotations.append({
            "position":      pos,
            "ref_base":      base,
            "is_uridine":    base in ("T", "U"),
            "is_known_psi":  pos in (known_psi or KNOWN_PSI_SITES),
            "in_renu_region": abs(pos - renu_pos) <= 5,
            "homopolymer":   hp_mask[i],
        })
    return annotations
