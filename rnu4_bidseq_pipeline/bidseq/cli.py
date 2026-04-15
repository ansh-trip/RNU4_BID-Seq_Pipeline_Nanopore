"""
cli.py — Command-line interface for the RNU4-2 BID-seq nanopore pipeline.

Commands
--------
  bidseq run            Full end-to-end pipeline (FASTQ → HTML report)
  bidseq call           Run calling only from a pre-aligned BAM
  bidseq qc             Run QC only on a FASTQ file
  bidseq align          Align a FASTQ to RNU4-2
  bidseq fetch-reference  Download RNU4-2 reference from NCBI
  bidseq version        Print version

Examples
--------
  # Single-sample BID-seq run
  bidseq run --treated sample.fastq.gz --output results/

  # Paired (treated + control)
  bidseq run --treated treated.fastq.gz --control ctrl.fastq.gz --output results/

  # Use a custom reference
  bidseq run --treated sample.fastq.gz --reference my_rnu4_2.fa --output results/

  # Call from a pre-aligned BAM (skip QC + alignment)
  bidseq call --treated sample.bam --output results/

  # Just fetch the reference
  bidseq fetch-reference --output ref/rnu4_2.fa
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click
from loguru import logger

from bidseq import __version__
from bidseq.utils import load_config, setup_logging, ensure_dir, check_tools, write_json


# ─── CLI group ────────────────────────────────────────────────────────────────

@click.group()
@click.version_option(__version__, prog_name="bidseq")
def main():
    """
    RNU4-2 BID-seq Nanopore Pipeline.

    Maps pseudouridine sites and detects the RENU syndrome n.64_65insT
    variant from nanopore sequencing data.
    """


# ─── bidseq run ──────────────────────────────────────────────────────────────

@main.command("run")
@click.option("--treated",   "-t", required=True, type=click.Path(exists=True),
              help="BID-seq treated FASTQ (or FASTQ.gz).")
@click.option("--control",   "-c", default=None, type=click.Path(exists=True),
              help="Untreated control FASTQ (optional but strongly recommended).")
@click.option("--reference", "-r", default=None, type=click.Path(),
              help="Reference FASTA.  If omitted, RNU4-2 is fetched from NCBI.")
@click.option("--output",    "-o", required=True, type=click.Path(),
              help="Output directory.")
@click.option("--sample",    "-s", default=None,
              help="Sample name for report.  Defaults to FASTQ filename stem.")
@click.option("--mode",      "-m", default="cdna",
              type=click.Choice(["cdna", "direct_rna"]),
              help="Sequencing mode.  'cdna' = cDNA after BID-seq RT (default).")
@click.option("--threads",   "-p", default=4, type=int,
              help="CPU threads for minimap2/samtools (default 4).")
@click.option("--config",    default=None, type=click.Path(exists=True),
              help="Optional YAML config override.")
@click.option("--min-coverage", default=20, type=int,
              help="Minimum read depth to analyse a position (default 20).")
@click.option("--min-del-rate", default=0.12, type=float,
              help="Minimum deletion rate to call a site (default 0.12).")
@click.option("--fdr",         default=0.05, type=float,
              help="FDR threshold after BH correction (default 0.05).")
@click.option("--background",  default=0.06, type=float,
              help="Nanopore background deletion rate prior (default 0.06).")
@click.option("--verbose",   "-v", is_flag=True, default=False)
def run_cmd(
    treated, control, reference, output, sample, mode, threads,
    config, min_coverage, min_del_rate, fdr, background, verbose,
):
    """Full end-to-end BID-seq pipeline: QC → align → pileup → call → report."""
    _run_pipeline(
        treated_fastq   = treated,
        control_fastq   = control,
        reference_path  = reference,
        output_dir      = Path(output),
        sample_name     = sample or Path(treated).stem,
        mode            = mode,
        threads         = threads,
        config_path     = config,
        min_coverage    = min_coverage,
        min_del_rate    = min_del_rate,
        fdr             = fdr,
        background_rate = background,
        verbose         = verbose,
        bam_mode        = False,
    )


# ─── bidseq call ─────────────────────────────────────────────────────────────

@main.command("call")
@click.option("--treated",   "-t", required=True, type=click.Path(exists=True),
              help="Pre-aligned treated BAM (sorted + indexed).")
@click.option("--control",   "-c", default=None, type=click.Path(exists=True),
              help="Pre-aligned control BAM (optional).")
@click.option("--reference", "-r", default=None, type=click.Path(),
              help="Reference FASTA.")
@click.option("--output",    "-o", required=True, type=click.Path())
@click.option("--sample",    "-s", default=None)
@click.option("--config",    default=None, type=click.Path(exists=True))
@click.option("--min-coverage", default=20, type=int)
@click.option("--min-del-rate", default=0.12, type=float)
@click.option("--fdr",         default=0.05, type=float)
@click.option("--background",  default=0.06, type=float)
@click.option("--verbose",   "-v", is_flag=True, default=False)
def call_cmd(treated, control, reference, output, sample, config,
             min_coverage, min_del_rate, fdr, background, verbose):
    """Run deletion calling + reporting from a pre-aligned BAM."""
    _run_pipeline(
        treated_fastq   = treated,
        control_fastq   = control,
        reference_path  = reference,
        output_dir      = Path(output),
        sample_name     = sample or Path(treated).stem,
        mode            = "cdna",
        threads         = 1,
        config_path     = config,
        min_coverage    = min_coverage,
        min_del_rate    = min_del_rate,
        fdr             = fdr,
        background_rate = background,
        verbose         = verbose,
        bam_mode        = True,
    )


# ─── bidseq qc ───────────────────────────────────────────────────────────────

@main.command("qc")
@click.argument("fastq", type=click.Path(exists=True))
@click.option("--output", "-o", default=".", type=click.Path())
@click.option("--threads", "-p", default=4, type=int)
def qc_cmd(fastq, output, threads):
    """Run QC on a FASTQ file and print summary."""
    from bidseq.qc import run_nanostat
    setup_logging()
    out = ensure_dir(output)
    metrics = run_nanostat(fastq, out, threads=threads)
    click.echo("\n── NanoStat Summary ──")
    for k, v in sorted(metrics.items()):
        click.echo(f"  {k:<40} {v}")


# ─── bidseq align ─────────────────────────────────────────────────────────────

@main.command("align")
@click.argument("fastq", type=click.Path(exists=True))
@click.option("--reference", "-r", required=True, type=click.Path())
@click.option("--output",    "-o", required=True, type=click.Path())
@click.option("--threads",   "-p", default=4, type=int)
@click.option("--mode",      "-m", default="cdna",
              type=click.Choice(["cdna", "direct_rna"]))
def align_cmd(fastq, reference, output, threads, mode):
    """Align a FASTQ to the reference and output a sorted BAM."""
    from bidseq.alignment import align_reads
    setup_logging()
    check_tools(["minimap2", "samtools"])
    output_bam = Path(output)
    stats = align_reads(fastq, reference, output_bam, mode=mode, threads=threads)
    click.echo(f"\nMapping rate: {stats.get('mapping_rate', 0):.1%}")


# ─── bidseq fetch-reference ───────────────────────────────────────────────────

@main.command("fetch-reference")
@click.option("--output", "-o", default="rnu4_2_reference.fa", type=click.Path())
def fetch_ref_cmd(output):
    """Download RNU4-2 reference sequence from NCBI."""
    from bidseq.reference import _fetch_from_ncbi
    setup_logging()
    path = _fetch_from_ncbi(Path(output))
    click.echo(f"Reference saved to: {path}")


# ─── bidseq version ──────────────────────────────────────────────────────────

@main.command("version")
def version_cmd():
    """Print version and exit."""
    click.echo(f"rnu4-bidseq-pipeline  v{__version__}")


# ─── Core pipeline function ───────────────────────────────────────────────────

def _run_pipeline(
    *,
    treated_fastq: str,
    control_fastq: Optional[str],
    reference_path: Optional[str],
    output_dir: Path,
    sample_name: str,
    mode: str,
    threads: int,
    config_path: Optional[str],
    min_coverage: int,
    min_del_rate: float,
    fdr: float,
    background_rate: float,
    verbose: bool,
    bam_mode: bool,
) -> None:
    """Orchestrate the full pipeline."""

    # ── Setup ─────────────────────────────────────────────────────────────────
    ensure_dir(output_dir)
    log_file = output_dir / "pipeline.log"
    setup_logging(log_file, level="DEBUG" if verbose else "INFO")

    logger.info(f"{'='*60}")
    logger.info(f" RNU4-2 BID-seq Pipeline  v{__version__}")
    logger.info(f" Sample : {sample_name}")
    logger.info(f" Output : {output_dir}")
    logger.info(f" Mode   : {mode}  |  Paired: {control_fastq is not None}")
    logger.info(f"{'='*60}")

    # ── Load config ───────────────────────────────────────────────────────────
    cfg = load_config(config_path)

    # ── Check tools (skip in bam_mode if no alignment needed) ─────────────────
    if not bam_mode:
        check_tools(["minimap2", "samtools"])

    # ── Reference ─────────────────────────────────────────────────────────────
    from bidseq.reference import get_reference, read_reference_sequence, get_reference_name, annotate_positions, KNOWN_PSI_SITES
    ref_dir  = output_dir / "reference"
    ref_fasta = get_reference(
        user_path  = reference_path,
        cache_dir  = ref_dir,
        auto_fetch = cfg["reference"]["auto_fetch"],
    )
    from bidseq.alignment import samtools_faidx
    samtools_faidx(ref_fasta)

    ref_seq  = read_reference_sequence(ref_fasta)
    ref_name = get_reference_name(ref_fasta)
    annotations = annotate_positions(
        ref_seq,
        known_psi = set(cfg["reference"]["known_pseudoU_sites"]),
    )

    # ── QC + Alignment (FASTQ mode) ───────────────────────────────────────────
    treated_bam  = None
    control_bam  = None
    qc_summary_t = {}
    align_stats  = {}

    if bam_mode:
        treated_bam = Path(treated_fastq)
        control_bam = Path(control_fastq) if control_fastq else None
        logger.info(f"BAM mode: skipping QC and alignment.")
    else:
        treated_bam, qc_summary_t, align_stats = _qc_and_align(
            fastq        = treated_fastq,
            ref_fasta    = ref_fasta,
            ref_name     = ref_name,
            ref_len      = len(ref_seq),
            output_dir   = output_dir / "treated",
            sample_name  = sample_name,
            mode         = mode,
            threads      = threads,
            cfg          = cfg,
        )

        if control_fastq:
            control_bam, _, _ = _qc_and_align(
                fastq       = control_fastq,
                ref_fasta   = ref_fasta,
                ref_name    = ref_name,
                ref_len     = len(ref_seq),
                output_dir  = output_dir / "control",
                sample_name = sample_name + "_ctrl",
                mode        = mode,
                threads     = threads,
                cfg         = cfg,
            )

    # ── Pileup ────────────────────────────────────────────────────────────────
    from bidseq.pileup import compute_pileup, save_pileup_tsv
    treated_pileup = compute_pileup(
        treated_bam, ref_name, ref_seq,
        min_mapq         = cfg["pileup"]["min_mapq"],
        min_base_quality = cfg["pileup"]["min_base_quality"],
        max_depth        = cfg["pileup"]["max_depth"],
    )
    control_pileup = None
    if control_bam:
        control_pileup = compute_pileup(
            control_bam, ref_name, ref_seq,
            min_mapq         = cfg["pileup"]["min_mapq"],
            min_base_quality = cfg["pileup"]["min_base_quality"],
            max_depth        = cfg["pileup"]["max_depth"],
        )

    pileup_out = output_dir / f"{sample_name}_pileup.tsv"
    save_pileup_tsv(treated_pileup, pileup_out)

    # ── Coverage QC ───────────────────────────────────────────────────────────
    from bidseq.qc import check_coverage
    cov_summary = check_coverage(
        treated_bam, ref_name, len(ref_seq),
        min_coverage = min_coverage,
        min_fraction = cfg["qc"]["min_transcript_coverage_fraction"],
    )
    if qc_summary_t:
        qc_summary_t["coverage"] = cov_summary
    else:
        qc_summary_t = {"coverage": cov_summary, "read_filtering": {}, "nanostat": {}}

    # ── Deletion statistics ───────────────────────────────────────────────────
    from bidseq.deletion_stats import run_deletion_analysis
    stats_df = run_deletion_analysis(
        treated_pileup   = treated_pileup,
        control_pileup   = control_pileup,
        annotations      = annotations,
        nanopore_bg_prior = background_rate,
        min_coverage     = min_coverage,
        min_deletion_rate = min_del_rate,
        fdr_threshold    = fdr,
    )
    stats_out = output_dir / f"{sample_name}_deletion_stats.tsv"
    stats_df.to_csv(stats_out, sep="\t", index=False, float_format="%.6f")

    # ── Pseudouridine calls ───────────────────────────────────────────────────
    from bidseq.pseudoU_caller import call_pseudoU_sites, calls_to_bed
    psi_result = call_pseudoU_sites(
        stats_df,
        known_psi_sites = set(cfg["reference"]["known_pseudoU_sites"]),
    )

    calls_out = output_dir / f"{sample_name}_pseudoU_calls.tsv"
    psi_result["calls"].to_csv(calls_out, sep="\t", index=False, float_format="%.6f")

    bed_out = output_dir / f"{sample_name}_pseudoU_calls.bed"
    calls_to_bed(psi_result["calls"], ref_name=ref_name, output_path=str(bed_out))

    # ── RENU tracking ─────────────────────────────────────────────────────────
    from bidseq.renu_tracker import track_renu_insertion
    renu_summary = track_renu_insertion(
        treated_bam,
        ref_name    = ref_name,
        ref_sequence= ref_seq,
        min_mapq    = cfg["pileup"]["min_mapq"],
        pileup_df   = treated_pileup,
    )
    renu_json_out = output_dir / f"{sample_name}_renu_summary.json"
    write_json(renu_summary.to_dict(), renu_json_out)

    # ── JSON summary ──────────────────────────────────────────────────────────
    summary = {
        "sample_name":      sample_name,
        "pipeline_version": __version__,
        "mode":             mode,
        "paired":           control_fastq is not None,
        "qc":               qc_summary_t,
        "alignment":        align_stats,
        "pseudoU_calls":    psi_result["summary"],
        "renu":             renu_summary.to_dict(),
    }
    write_json(summary, output_dir / f"{sample_name}_summary.json")

    # ── HTML report ───────────────────────────────────────────────────────────
    from bidseq.report import generate_report
    report_path = output_dir / f"{sample_name}_report.html"
    generate_report(
        sample_name     = sample_name,
        pileup_df       = treated_pileup,
        stats_df        = stats_df,
        psi_result      = psi_result,
        renu_summary    = renu_summary,
        qc_summary      = qc_summary_t,
        alignment_stats = align_stats,
        output_path     = report_path,
        mode            = mode,
        paired          = control_fastq is not None,
        background_rate = background_rate,
        known_psi_sites = set(cfg["reference"]["known_pseudoU_sites"]),
    )

    # ── Done ──────────────────────────────────────────────────────────────────
    logger.success(f"\n{'='*60}")
    logger.success(f" Pipeline complete!  Outputs in: {output_dir}")
    logger.success(f" HTML report       : {report_path}")
    logger.success(f" Ψ calls           : {calls_out}")
    logger.success(f" RENU summary      : {renu_json_out}")
    logger.success(f" Full stats        : {stats_out}")
    logger.success(f"{'='*60}\n")


# ─── QC + Align helper ────────────────────────────────────────────────────────

def _qc_and_align(
    fastq, ref_fasta, ref_name, ref_len,
    output_dir, sample_name, mode, threads, cfg,
):
    """Run QC filtering + alignment; return (bam_path, qc_summary, align_stats)."""
    from bidseq.qc import run_nanostat, filter_reads, build_qc_summary
    from bidseq.alignment import align_reads

    ensure_dir(output_dir)

    # NanoStat
    nanostat_metrics = run_nanostat(fastq, output_dir / "qc", threads=threads)

    # Read filtering
    filtered_fq = output_dir / "filtered.fastq"
    filter_summary = filter_reads(
        fastq,
        filtered_fq,
        min_length      = cfg["alignment"]["min_read_length"],
        max_length      = cfg["alignment"]["max_read_length"],
        min_mean_quality= cfg["qc"]["min_mean_quality"],
    )

    # Alignment
    bam_path = output_dir / f"{sample_name}.bam"
    align_stats = align_reads(
        filtered_fq, ref_fasta, bam_path,
        mode       = mode,
        threads    = threads,
        min_mapq   = cfg["alignment"]["min_mapq"],
        extra_flags= cfg["alignment"]["extra_flags"],
        sample_name= sample_name,
    )

    # Build QC summary
    from bidseq.qc import build_qc_summary, check_coverage
    qc_summary = build_qc_summary(nanostat_metrics, filter_summary, {}, sample_name)

    return bam_path, qc_summary, align_stats


if __name__ == "__main__":
    main()
