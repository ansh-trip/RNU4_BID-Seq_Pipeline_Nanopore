"""
utils.py — Shared helpers for the BID-seq nanopore pipeline.
"""

from __future__ import annotations

import os
import sys
import shutil
import subprocess
import hashlib
import json
import gzip
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from loguru import logger


# ─── Config ──────────────────────────────────────────────────────────────────

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load YAML config, merging with defaults."""
    default_path = Path(__file__).parent.parent / "config" / "default_config.yaml"
    with open(default_path) as f:
        config = yaml.safe_load(f)

    if config_path:
        with open(config_path) as f:
            user_cfg = yaml.safe_load(f)
        config = _deep_merge(config, user_cfg)

    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


# ─── Tool availability ────────────────────────────────────────────────────────

REQUIRED_TOOLS = {
    "minimap2": "conda install -c bioconda minimap2",
    "samtools": "conda install -c bioconda samtools",
}

def check_tools(tools: Optional[List[str]] = None) -> None:
    """Verify required external tools are on PATH.  Raises RuntimeError if missing."""
    targets = tools or list(REQUIRED_TOOLS.keys())
    missing = [t for t in targets if shutil.which(t) is None]
    if missing:
        msgs = [f"  {t}  →  {REQUIRED_TOOLS.get(t, 'install via bioconda')}" for t in missing]
        raise RuntimeError(
            "Required tools not found on PATH:\n" + "\n".join(msgs) +
            "\nRun:  conda install -c bioconda " + " ".join(missing)
        )


# ─── File helpers ─────────────────────────────────────────────────────────────

def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_gzipped(path: str | Path) -> bool:
    with open(path, "rb") as f:
        return f.read(2) == b"\x1f\x8b"


def open_maybe_gz(path: str | Path, mode: str = "r"):
    path = Path(path)
    if is_gzipped(path):
        return gzip.open(path, mode + "t")
    return open(path, mode)


def file_md5(path: str | Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ─── Shell runner ─────────────────────────────────────────────────────────────

def run_command(
    cmd: List[str] | str,
    *,
    capture: bool = False,
    check: bool = True,
    cwd: Optional[str | Path] = None,
    label: Optional[str] = None,
) -> subprocess.CompletedProcess:
    """Run a shell command with logging.

    Parameters
    ----------
    cmd:     Command as list (preferred) or shell string.
    capture: If True, capture stdout/stderr; otherwise stream to terminal.
    check:   Raise CalledProcessError on non-zero exit.
    cwd:     Working directory.
    label:   Human-readable label for log messages.
    """
    tag = label or (cmd[0] if isinstance(cmd, list) else cmd.split()[0])
    logger.info(f"[{tag}] {' '.join(cmd) if isinstance(cmd, list) else cmd}")

    kwargs: Dict[str, Any] = dict(
        check=check,
        cwd=str(cwd) if cwd else None,
        shell=isinstance(cmd, str),
    )
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
        kwargs["text"] = True

    result = subprocess.run(cmd, **kwargs)

    if capture and result.returncode != 0 and not check:
        logger.warning(f"[{tag}] Exit {result.returncode}: {result.stderr or ''}")

    return result


# ─── Sequence utilities ───────────────────────────────────────────────────────

COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")

def reverse_complement(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def find_homopolymers(seq: str, min_len: int = 4) -> List[Tuple[int, int, str]]:
    """Return list of (start, end, base) for homopolymer runs (0-based, exclusive end)."""
    runs: List[Tuple[int, int, str]] = []
    i = 0
    while i < len(seq):
        j = i + 1
        while j < len(seq) and seq[j] == seq[i]:
            j += 1
        if j - i >= min_len:
            runs.append((i, j, seq[i]))
        i = j
    return runs


def homopolymer_mask(seq: str, min_len: int = 4) -> List[bool]:
    """Return boolean list; True = position is inside a homopolymer run."""
    mask = [False] * len(seq)
    for start, end, _ in find_homopolymers(seq, min_len):
        for k in range(start, end):
            mask[k] = True
    return mask


# ─── FASTQ helpers ────────────────────────────────────────────────────────────

def count_reads(fastq_path: str | Path) -> int:
    """Fast read count via line count / 4."""
    fastq_path = Path(fastq_path)
    if is_gzipped(fastq_path):
        with gzip.open(fastq_path, "rb") as f:
            return sum(1 for _ in f) // 4
    with open(fastq_path, "rb") as f:
        return sum(1 for _ in f) // 4


# ─── JSON I/O ─────────────────────────────────────────────────────────────────

def write_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, default=str)


def read_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


# ─── Logging setup ───────────────────────────────────────────────────────────

def setup_logging(log_file: Optional[str | Path] = None, level: str = "INFO") -> None:
    """Configure loguru: console (coloured) + optional file sink."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        level=level,
        colorize=True,
    )
    if log_file:
        logger.add(
            str(log_file),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
            level="DEBUG",
            rotation="10 MB",
        )
