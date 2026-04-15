"""
report.py — Generate a self-contained HTML report for the BID-seq run.

The report embeds:
  - Run metadata (sample name, date, software versions)
  - QC summary table
  - Coverage plot
  - Deletion rate landscape plot
  - Volcano plot
  - Heatmap
  - Known Ψ recovery bar chart
  - Full Ψ calls table (sortable)
  - RENU insertion summary + pie chart
  - Per-position statistics table (collapsible)
  - Interpretation text for each call
"""

from __future__ import annotations

import json
import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from jinja2 import Environment, BaseLoader
from loguru import logger

from bidseq import __version__
from bidseq import visualizer


# ─── HTML template (inline Jinja2) ───────────────────────────────────────────

_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>BID-seq Report — {{ sample_name }}</title>
  <style>
    :root {
      --high: #c0392b; --mod: #e67e22; --low: #3498db;
      --noise: #bdc3c7; --renu-red: #e74c3c; --psi: #8e44ad;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', Arial, sans-serif; background: #f4f6f8; color: #2c3e50; }
    header { background: linear-gradient(135deg,#1a252f,#2c3e50); color:white; padding:28px 40px; }
    header h1 { font-size:1.8em; font-weight:700; }
    header p  { color:#bdc3c7; font-size:0.9em; margin-top:4px; }
    .container { max-width:1200px; margin:30px auto; padding:0 20px; }
    .card { background:white; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,.08);
            padding:24px; margin-bottom:28px; }
    .card h2 { font-size:1.2em; border-bottom:2px solid #ecf0f1; padding-bottom:10px;
               margin-bottom:16px; color:#2c3e50; }
    .metrics-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(160px,1fr));
                    gap:12px; margin-bottom:16px; }
    .metric { background:#f8f9fa; border-radius:6px; padding:14px; text-align:center; }
    .metric .value { font-size:1.6em; font-weight:700; color:#2c3e50; }
    .metric .label { font-size:0.78em; color:#7f8c8d; margin-top:4px; }
    .badge { display:inline-block; padding:2px 9px; border-radius:12px; font-size:0.78em;
             font-weight:600; color:white; }
    .badge-high     { background:var(--high); }
    .badge-moderate { background:var(--mod); }
    .badge-low      { background:var(--low); }
    .badge-noise    { background:var(--noise); color:#555; }
    .badge-wt       { background:#27ae60; }
    .badge-renu     { background:var(--renu-red); }
    .badge-mosaic   { background:#f39c12; }
    .badge-ambig    { background:#95a5a6; }
    table { width:100%; border-collapse:collapse; font-size:0.86em; }
    th { background:#f0f2f5; font-weight:600; padding:9px 12px; text-align:left;
         border-bottom:2px solid #dee2e6; position:sticky; top:0; }
    td { padding:8px 12px; border-bottom:1px solid #f0f2f5; vertical-align:top; }
    tr:hover td { background:#f8f9fa; }
    .tbl-scroll { overflow-x:auto; max-height:500px; overflow-y:auto; }
    .img-block { text-align:center; margin:10px 0; }
    .img-block img { max-width:100%; border-radius:4px; }
    .interpretation { font-size:0.88em; color:#555; background:#fdfefe;
                      border-left:3px solid #3498db; padding:8px 12px;
                      border-radius:0 4px 4px 0; margin:4px 0; }
    .renu-alert { background:#fdebd0; border:1px solid #e67e22;
                  border-radius:6px; padding:14px 18px; margin-bottom:16px; }
    .renu-ok    { background:#d5f5e3; border:1px solid #27ae60;
                  border-radius:6px; padding:14px 18px; margin-bottom:16px; }
    details { margin-bottom:8px; }
    summary { cursor:pointer; font-weight:600; padding:8px;
              background:#f0f2f5; border-radius:4px; user-select:none; }
    summary:hover { background:#e8eaed; }
    footer { text-align:center; color:#aaa; font-size:0.82em; padding:20px; }
    .warn  { color:#e67e22; font-weight:600; }
    .ok    { color:#27ae60; font-weight:600; }
    .pass-badge { display:inline-block; padding:3px 10px; border-radius:4px;
                  font-size:0.85em; font-weight:700; }
    .pass  { background:#d5f5e3; color:#1e8449; }
    .fail  { background:#fadbd8; color:#c0392b; }
  </style>
</head>
<body>
<header>
  <h1>🔬 BID-seq Pseudouridine Report — {{ sample_name }}</h1>
  <p>Pipeline: rnu4-bidseq-pipeline v{{ version }} &nbsp;|&nbsp;
     Run date: {{ run_date }} &nbsp;|&nbsp;
     Mode: {{ mode }} &nbsp;|&nbsp;
     Paired: {{ "Yes (+ control)" if paired else "No (single-sample)" }}</p>
</header>

<div class="container">

  <!-- ── QC Summary ─────────────────────────────────────────────────────── -->
  <div class="card">
    <h2>📊 Quality Control Summary</h2>
    <div class="metrics-grid">
      <div class="metric">
        <div class="value">{{ qc.total_reads | default('—') }}</div>
        <div class="label">Total reads (input)</div>
      </div>
      <div class="metric">
        <div class="value">{{ qc.filtered_reads | default('—') }}</div>
        <div class="label">Reads after QC filter</div>
      </div>
      <div class="metric">
        <div class="value">{{ qc.mapped_reads | default('—') }}</div>
        <div class="label">Mapped reads</div>
      </div>
      <div class="metric">
        <div class="value">{{ "%.1f%%" | format(qc.mapping_rate * 100) if qc.mapping_rate is not none else '—' }}</div>
        <div class="label">Mapping rate</div>
      </div>
      <div class="metric">
        <div class="value">{{ "%.0f×" | format(qc.mean_coverage) if qc.mean_coverage is not none else '—' }}</div>
        <div class="label">Mean coverage</div>
      </div>
      <div class="metric">
        <div class="value">{{ "%.1f%%" | format(qc.fraction_covered * 100) if qc.fraction_covered is not none else '—' }}</div>
        <div class="label">Transcript covered</div>
      </div>
      <div class="metric">
        <div class="value">
          <span class="pass-badge {{ 'pass' if qc.coverage_pass else 'fail' }}">
            {{ "PASS" if qc.coverage_pass else "FAIL" }}
          </span>
        </div>
        <div class="label">Coverage QC</div>
      </div>
    </div>
    {% if qc.warnings %}
    <p class="warn">⚠ Warnings: {{ qc.warnings | join("; ") }}</p>
    {% endif %}
  </div>

  <!-- ── Coverage plot ──────────────────────────────────────────────────── -->
  <div class="card">
    <h2>📈 Read Coverage</h2>
    <div class="img-block"><img src="data:image/png;base64,{{ plots.coverage }}" alt="Coverage plot"></div>
  </div>

  <!-- ── Deletion rate landscape ────────────────────────────────────────── -->
  <div class="card">
    <h2>🗺 Deletion Rate Landscape (BID-seq Signal)</h2>
    <p style="color:#7f8c8d;font-size:0.88em;margin-bottom:12px;">
      Each position's deletion rate in the BID-seq treated sample.
      Coloured markers indicate called Ψ sites.  Dashed line = nanopore background.
    </p>
    <div class="img-block"><img src="data:image/png;base64,{{ plots.deletion_rate }}" alt="Deletion rate plot"></div>
  </div>

  <!-- ── Pseudouridine calls ────────────────────────────────────────────── -->
  <div class="card">
    <h2>🔵 Pseudouridine Calls</h2>

    <div class="metrics-grid">
      <div class="metric">
        <div class="value" style="color:var(--high)">{{ psi_summary.high_confidence_calls }}</div>
        <div class="label">High confidence</div>
      </div>
      <div class="metric">
        <div class="value" style="color:var(--mod)">{{ psi_summary.moderate_calls }}</div>
        <div class="label">Moderate</div>
      </div>
      <div class="metric">
        <div class="value" style="color:var(--low)">{{ psi_summary.low_calls }}</div>
        <div class="label">Low confidence</div>
      </div>
      <div class="metric">
        <div class="value" style="color:var(--psi)">{{ psi_summary.known_psi_recovered }} / {{ psi_summary.known_psi_total }}</div>
        <div class="label">Known Ψ recovered</div>
      </div>
      <div class="metric">
        <div class="value">{{ psi_summary.novel_calls }}</div>
        <div class="label">Novel calls</div>
      </div>
    </div>

    {% if calls_rows %}
    <div class="tbl-scroll">
      <table>
        <thead>
          <tr>
            <th>Position</th><th>Ref</th><th>Del rate</th><th>Fold change</th>
            <th>q-value</th><th>Confidence</th><th>Known Ψ</th><th>Flags</th><th>Interpretation</th>
          </tr>
        </thead>
        <tbody>
          {% for row in calls_rows %}
          <tr>
            <td><b>{{ row.position }}</b></td>
            <td>{{ row.ref_base }}</td>
            <td>{{ "%.1f%%" | format(row.deletion_rate_t * 100) }}</td>
            <td>{{ "%.1f×" | format(row.fold_change) }}</td>
            <td>{{ "%.4f" | format(row.q_value) }}</td>
            <td>
              {% if row.confidence == "HIGH_CONFIDENCE" %}
                <span class="badge badge-high">HIGH</span>
              {% elif row.confidence == "MODERATE" %}
                <span class="badge badge-moderate">MOD</span>
              {% else %}
                <span class="badge badge-low">LOW</span>
              {% endif %}
            </td>
            <td>{{ "✓" if row.is_known_psi else "—" }}</td>
            <td style="color:#e67e22;font-size:0.82em">{{ row.flags or "—" }}</td>
            <td class="interpretation">{{ row.interpretation }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
    {% else %}
    <p style="color:#7f8c8d;">No pseudouridine sites called above threshold.</p>
    {% endif %}

    <div class="img-block" style="margin-top:20px">
      <img src="data:image/png;base64,{{ plots.volcano }}" alt="Volcano plot">
    </div>
  </div>

  <!-- ── Known Ψ recovery ───────────────────────────────────────────────── -->
  <div class="card">
    <h2>🟣 Known Ψ Site Recovery</h2>
    <div class="img-block"><img src="data:image/png;base64,{{ plots.known_psi }}" alt="Known Ψ recovery"></div>
  </div>

  <!-- ── Deletion heatmap ───────────────────────────────────────────────── -->
  <div class="card">
    <h2>🌡 Deletion Rate Heatmap</h2>
    <div class="img-block"><img src="data:image/png;base64,{{ plots.heatmap }}" alt="Heatmap"></div>
  </div>

  <!-- ── RENU tracking ──────────────────────────────────────────────────── -->
  <div class="card">
    <h2>🧬 RENU Syndrome Variant Tracking (n.64_65insT)</h2>

    {% set zyg = renu.zygosity %}
    {% if zyg == "WILD_TYPE" %}
      <div class="renu-ok"><strong>✅ WILD-TYPE</strong> — No RENU insertion detected at position 64.</div>
    {% elif zyg in ("HETEROZYGOUS", "NEAR_HOMOZYGOUS") %}
      <div class="renu-alert"><strong>🔴 RENU INSERTION DETECTED — {{ zyg }}</strong></div>
    {% elif zyg == "HOMOZYGOUS" %}
      <div class="renu-alert"><strong>🔴🔴 RENU INSERTION DETECTED — HOMOZYGOUS (unusual; verify)</strong></div>
    {% elif zyg == "MOSAIC_OR_LOW_SIGNAL" %}
      <div class="renu-alert"><strong>⚠ Low-level insertion signal — possible mosaicism or artefact</strong></div>
    {% else %}
      <div class="renu-alert"><strong>❓ Ambiguous</strong></div>
    {% endif %}

    <div class="metrics-grid">
      <div class="metric">
        <div class="value" style="color:var(--renu-red)">{{ "%.1f%%" | format(renu.vaf * 100) }}</div>
        <div class="label">VAF</div>
      </div>
      <div class="metric">
        <div class="value">{{ renu.renu_t_insertion_reads }}</div>
        <div class="label">RENU reads</div>
      </div>
      <div class="metric">
        <div class="value">{{ renu.wild_type_reads }}</div>
        <div class="label">Wild-type reads</div>
      </div>
      <div class="metric">
        <div class="value">{{ renu.total_reads_assessed }}</div>
        <div class="label">Total assessed</div>
      </div>
      <div class="metric">
        <div class="value">{{ renu.coverage_at_site }}</div>
        <div class="label">Coverage at pos 64</div>
      </div>
      <div class="metric">
        <div class="value">{{ "%.1f%%" | format(renu.nearby_deletion_rate * 100) }}</div>
        <div class="label">Nearby del rate</div>
      </div>
    </div>

    <div class="img-block"><img src="data:image/png;base64,{{ plots.renu }}" alt="RENU chart"></div>

    <p style="margin-top:12px;font-size:0.9em;background:#f8f9fa;border-radius:4px;padding:10px;">
      <strong>Interpretation:</strong> {{ renu.interpretation }}
    </p>
  </div>

  <!-- ── Full position table (collapsible) ──────────────────────────────── -->
  <div class="card">
    <h2>📋 Full Per-Position Statistics</h2>
    <details>
      <summary>Show all {{ all_rows | length }} positions</summary>
      <div class="tbl-scroll" style="margin-top:12px;">
        <table>
          <thead>
            <tr>
              <th>Pos</th><th>Ref</th><th>Cov</th><th>Del count</th>
              <th>Del rate</th><th>Fold change</th><th>q-value</th>
              <th>Confidence</th><th>Flags</th>
            </tr>
          </thead>
          <tbody>
            {% for row in all_rows %}
            <tr>
              <td>{{ row.position }}</td>
              <td>{{ row.ref_base }}</td>
              <td>{{ row.coverage_t }}</td>
              <td>{{ row.deletion_count_t }}</td>
              <td>{{ "%.1f%%" | format(row.deletion_rate_t * 100) }}</td>
              <td>{{ "%.1f×" | format(row.fold_change) }}</td>
              <td>{{ "%.4f" | format(row.q_value) }}</td>
              <td>
                {% if row.confidence == "HIGH_CONFIDENCE" %}
                  <span class="badge badge-high">HIGH</span>
                {% elif row.confidence == "MODERATE" %}
                  <span class="badge badge-moderate">MOD</span>
                {% elif row.confidence == "LOW" %}
                  <span class="badge badge-low">LOW</span>
                {% else %}
                  <span class="badge badge-noise">NOISE</span>
                {% endif %}
              </td>
              <td style="font-size:0.82em;color:#e67e22">{{ row.flags or "—" }}</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
    </details>
  </div>

  <!-- ── Methods note ───────────────────────────────────────────────────── -->
  <div class="card">
    <h2>📝 Methods Summary</h2>
    <p style="font-size:0.9em;line-height:1.7;">
      Nanopore reads were quality-filtered (min length={{ params.min_read_length }},
      min quality={{ params.min_mean_quality }}) and aligned to the RNU4-2 reference
      (NCBI NR_002715) using minimap2 (preset={{ params.minimap2_preset }}) with
      minimum MAPQ {{ params.min_mapq }}.  Per-position pileup statistics were computed
      using pysam.  Deletion rates were tested for enrichment above the estimated nanopore
      background rate ({{ "%.1f%%" | format(params.background_rate * 100) }}) using a
      {% if paired %}Fisher's exact test (treated vs control){% else %}one-sided binomial test{% endif %},
      with Benjamini–Hochberg FDR correction.  Sites with q &lt; {{ params.fdr_threshold }},
      fold-change ≥ {{ params.min_fold_change }}× and deletion rate ≥ {{ "%.0f%%" | format(params.min_deletion_rate * 100) }}
      were called as pseudouridine candidates.  The RENU n.64_65insT insertion was detected
      by scanning insertions in the pileup at reference position 64 with sequence-context
      validation of flanking bases.
    </p>
  </div>

</div>
<footer>
  Generated by rnu4-bidseq-pipeline v{{ version }} on {{ run_date }}.
  For research use only.  Not validated for clinical diagnostics.
</footer>
</body>
</html>
"""


# ─── Report builder ───────────────────────────────────────────────────────────

def generate_report(
    sample_name: str,
    pileup_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    psi_result: Dict,
    renu_summary,
    qc_summary: Dict,
    alignment_stats: Dict,
    output_path: str | Path,
    *,
    mode: str = "cdna",
    paired: bool = False,
    background_rate: float = 0.06,
    params: Optional[Dict] = None,
    known_psi_sites: Optional[set] = None,
) -> Path:
    """
    Render the complete HTML report and write to output_path.
    """
    from bidseq.reference import KNOWN_PSI_SITES
    known = known_psi_sites or KNOWN_PSI_SITES

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Generating HTML report …")

    # ── Generate plots (base64 embedded) ─────────────────────────────────────
    plots = {}
    try:
        plots["coverage"]     = visualizer.plot_coverage(pileup_df, title=f"Coverage — {sample_name}")
        plots["deletion_rate"]= visualizer.plot_deletion_rate(
            stats_df, background_rate=background_rate, title=f"Deletion Rate — {sample_name}"
        )
        plots["volcano"]      = visualizer.plot_volcano(stats_df)
        plots["heatmap"]      = visualizer.plot_deletion_heatmap(stats_df)
        plots["known_psi"]    = visualizer.plot_known_psi_recovery(stats_df, known)
        plots["renu"]         = visualizer.plot_renu_summary(renu_summary.to_dict())
    except Exception as e:
        logger.warning(f"Plot generation error: {e}")
        for k in ("coverage", "deletion_rate", "volcano", "heatmap", "known_psi", "renu"):
            plots.setdefault(k, "")

    # ── Prepare template data ─────────────────────────────────────────────────
    calls_df = psi_result.get("calls", pd.DataFrame())
    all_df   = psi_result.get("all", stats_df)
    psi_summary = psi_result.get("summary", {})

    calls_rows = calls_df.to_dict(orient="records") if not calls_df.empty else []
    all_rows   = all_df.to_dict(orient="records")

    default_params = {
        "min_read_length":  50,
        "min_mean_quality": 7.0,
        "minimap2_preset":  "map-ont",
        "min_mapq":         10,
        "background_rate":  background_rate,
        "fdr_threshold":    0.05,
        "min_fold_change":  3.0,
        "min_deletion_rate": 0.12,
    }
    if params:
        default_params.update(params)

    qc_data = {
        "total_reads":       qc_summary.get("read_filtering", {}).get("total_in"),
        "filtered_reads":    qc_summary.get("read_filtering", {}).get("total_out"),
        "mapped_reads":      alignment_stats.get("mapped_reads"),
        "mapping_rate":      alignment_stats.get("mapping_rate"),
        "mean_coverage":     qc_summary.get("coverage", {}).get("mean_coverage"),
        "fraction_covered":  qc_summary.get("coverage", {}).get("fraction_covered"),
        "coverage_pass":     qc_summary.get("coverage", {}).get("pass", False),
        "warnings":          [],
    }
    if qc_data.get("mapping_rate", 1.0) is not None and (qc_data["mapping_rate"] or 1.0) < 0.5:
        qc_data["warnings"].append(f"Low mapping rate ({(qc_data['mapping_rate'] or 0):.1%})")
    if not qc_data.get("coverage_pass"):
        qc_data["warnings"].append("Coverage below threshold")

    # ── Render ────────────────────────────────────────────────────────────────
    env = Environment(loader=BaseLoader(), autoescape=False)
    tmpl = env.from_string(_TEMPLATE)

    html = tmpl.render(
        sample_name = sample_name,
        version     = __version__,
        run_date    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        mode        = mode,
        paired      = paired,
        qc          = qc_data,
        plots       = plots,
        psi_summary = psi_summary,
        calls_rows  = calls_rows,
        all_rows    = all_rows,
        renu        = renu_summary.to_dict(),
        params      = default_params,
    )

    output_path.write_text(html, encoding="utf-8")
    logger.success(f"HTML report written: {output_path}")
    return output_path
