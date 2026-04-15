#!/usr/bin/env bash
# ============================================================
# install.sh — One-command installer for rnu4-bidseq-pipeline
#
# Usage:
#   bash scripts/install.sh
#
# What this does:
#   1. Creates a conda environment 'bidseq_nanopore'
#   2. Installs all Python and bioinformatics dependencies
#   3. Installs the pipeline as an editable package
#   4. Verifies key tools are accessible
#   5. Fetches the RNU4-2 reference from NCBI
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
ENV_NAME="bidseq_nanopore"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   RNU4-2 BID-seq Nanopore Pipeline — Installer          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── Check conda ───────────────────────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found. Install Miniconda or Anaconda first."
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
echo "✓ conda found: $(conda --version)"

# ── Create or update conda environment ───────────────────────────────────────
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists — updating …"
    conda env update -n "${ENV_NAME}" -f "${REPO_DIR}/environment.yml" --prune
else
    echo "Creating conda environment '${ENV_NAME}' …"
    conda env create -n "${ENV_NAME}" -f "${REPO_DIR}/environment.yml"
fi
echo "✓ Conda environment ready"

# ── Install pipeline package ──────────────────────────────────────────────────
echo ""
echo "Installing rnu4-bidseq-pipeline in editable mode …"
conda run -n "${ENV_NAME}" pip install -e "${REPO_DIR}" --quiet
echo "✓ Pipeline package installed"

# ── Verify tools ──────────────────────────────────────────────────────────────
echo ""
echo "Verifying required tools:"
for tool in minimap2 samtools; do
    if conda run -n "${ENV_NAME}" command -v "$tool" &>/dev/null; then
        ver=$(conda run -n "${ENV_NAME}" "$tool" --version 2>&1 | head -1)
        echo "  ✓ $tool   ($ver)"
    else
        echo "  ✗ $tool NOT FOUND — install with: conda install -c bioconda $tool"
    fi
done

# ── Fetch reference ───────────────────────────────────────────────────────────
echo ""
echo "Fetching RNU4-2 reference from NCBI …"
REFDIR="${REPO_DIR}/reference"
mkdir -p "${REFDIR}"
if [ -f "${REFDIR}/rnu4_2_reference.fa" ]; then
    echo "  ✓ Reference already cached: ${REFDIR}/rnu4_2_reference.fa"
else
    conda run -n "${ENV_NAME}" bidseq fetch-reference \
        --output "${REFDIR}/rnu4_2_reference.fa" || \
    echo "  ⚠ NCBI fetch failed — will use built-in sequence on first run"
fi

# ── Run tests ─────────────────────────────────────────────────────────────────
echo ""
echo "Running test suite …"
conda run -n "${ENV_NAME}" pytest "${REPO_DIR}/tests/" -v \
    --tb=short --ignore="${REPO_DIR}/tests/test_integration.py" \
    2>&1 | tail -30

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Installation complete!                                  ║"
echo "║                                                          ║"
echo "║  Activate environment:                                   ║"
echo "║    conda activate ${ENV_NAME}                            ║"
echo "║                                                          ║"
echo "║  Run the pipeline:                                       ║"
echo "║    bidseq run --treated sample.fastq --output results/   ║"
echo "║                                                          ║"
echo "║  Full documentation:  cat README.md                      ║"
echo "╚══════════════════════════════════════════════════════════╝"
