"""Plot benchmark results from the metrics folder.

Each JSON file in the folder becomes one series; the legend label is the
filename stem (e.g. baseline.json → "baseline").

Usage:
    python -m ocr_llm.plot                        # reads ./metrics/
    python -m ocr_llm.plot --metrics_dir=metrics  # explicit path
    python -m ocr_llm.plot --out=comparison.png   # save instead of show
"""

from __future__ import annotations

import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

DEFAULT_METRICS_DIR = "metrics"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_runs(metrics_dir: str) -> dict[str, dict]:
    """Return {legend: data} for every JSON file in metrics_dir, sorted by name."""
    folder = Path(metrics_dir)
    files = sorted(folder.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {metrics_dir!r}")
    runs = {}
    for f in files:
        with open(f) as fh:
            runs[f.stem] = json.load(fh)
    return runs


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _grouped_bars(
    ax: plt.Axes,
    labels: list[str],
    groups: dict[str, list[float]],
    title: str,
    ylabel: str,
    colors: list[str],
) -> None:
    """Draw a grouped bar chart.

    groups: {group_label: [value_per_run, ...]}
    """
    n_runs = len(labels)
    n_groups = len(groups)
    group_keys = list(groups.keys())
    x = np.arange(n_groups)
    bar_width = min(0.7 / n_runs, 0.25)
    offsets = np.linspace(-(n_runs - 1) / 2, (n_runs - 1) / 2, n_runs) * bar_width

    for i, (run_label, offset) in enumerate(zip(labels, offsets)):
        values = [groups[gk][i] for gk in group_keys]
        bars = ax.bar(
            x + offset,
            values,
            bar_width,
            label=run_label,
            color=colors[i % len(colors)],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f"{val:.0f}" if val >= 10 else f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="black",
                )

    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(group_keys, fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)


def _bar_chart(
    ax: plt.Axes,
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    colors: list[str],
) -> None:
    """Simple single-group bar chart — one bar per run."""
    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        values,
        color=[colors[i % len(colors)] for i in range(len(labels))],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        width=0.5,
    )
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.1f}"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)


# ---------------------------------------------------------------------------
# Main plot builder
# ---------------------------------------------------------------------------

PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
]


def plot(metrics_dir: str = DEFAULT_METRICS_DIR, out: str | None = None) -> None:
    """Load all JSONs in metrics_dir and produce a comparison figure.

    Args:
        metrics_dir: Folder containing benchmark JSON files.
        out:         If set, save the figure to this path instead of displaying it.
    """
    runs = load_runs(metrics_dir)
    legends = list(runs.keys())
    n = len(legends)

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#F8F8F8")
    title_parts = [f"OCR Benchmark Comparison  ({n} run{'s' if n != 1 else ''})"]
    fig.suptitle(title_parts[0], fontsize=14, fontweight="bold", y=0.98)

    gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.38,
                          left=0.07, right=0.97, top=0.93, bottom=0.07)

    colors = PALETTE[:n] if n <= len(PALETTE) else PALETTE * (n // len(PALETTE) + 1)

    # ── Row 0: latency metrics ───────────────────────────────────────────────
    latency_specs = [
        ("ttft",  "TTFT (ms)",              "Time to First Token"),
        ("itl",   "ITL (ms)",               "Inter-Token Latency"),
        ("tpot",  "TPOT (ms)",              "Time Per Output Token"),
    ]
    for col, (key, ylabel, title) in enumerate(latency_specs):
        ax = fig.add_subplot(gs[0, col])
        groups: dict[str, list[float]] = {
            "mean":   [runs[l]["latency_ms"][key]["mean"]   for l in legends],
            "median": [runs[l]["latency_ms"][key]["median"] for l in legends],
            "p90":    [runs[l]["latency_ms"][key]["p90"]    for l in legends],
            "p99":    [runs[l]["latency_ms"][key]["p99"]    for l in legends],
        }
        _grouped_bars(ax, legends, groups, title, ylabel, colors)
        if col == 0:
            ax.legend(legends, fontsize=7, loc="upper right",
                      framealpha=0.7, handlelength=1.2)

    # ── Row 1, col 0-1: E2EL + throughput ────────────────────────────────────
    ax_e2el = fig.add_subplot(gs[1, 0])
    groups_e2el: dict[str, list[float]] = {
        "mean":   [runs[l]["latency_ms"]["e2el"]["mean"]   for l in legends],
        "median": [runs[l]["latency_ms"]["e2el"]["median"] for l in legends],
        "p90":    [runs[l]["latency_ms"]["e2el"]["p90"]    for l in legends],
        "p99":    [runs[l]["latency_ms"]["e2el"]["p99"]    for l in legends],
    }
    _grouped_bars(ax_e2el, legends, groups_e2el, "End-to-End Latency", "E2EL (ms)", colors)

    ax_tput = fig.add_subplot(gs[1, 1])
    groups_tput: dict[str, list[float]] = {
        "output\ntokens/s":  [runs[l]["throughput"]["output_tokens_per_s"]  for l in legends],
        "total\ntokens/s":   [runs[l]["throughput"]["total_tokens_per_s"]   for l in legends],
        "peak\ntokens/s":    [runs[l]["throughput"]["peak_output_tokens_per_s"] for l in legends],
    }
    _grouped_bars(ax_tput, legends, groups_tput, "Throughput", "tokens / s", colors)

    ax_rps = fig.add_subplot(gs[1, 2])
    groups_rps: dict[str, list[float]] = {
        "requests/s":  [runs[l]["throughput"]["requests_per_s"]  for l in legends],
        "completed":   [runs[l]["throughput"]["completed"]        for l in legends],
        "failed":      [runs[l]["throughput"]["failed"]           for l in legends],
    }
    _grouped_bars(ax_rps, legends, groups_rps, "Request Stats", "count / rate", colors)

    # ── Row 2: PDF-level + normalised page time ───────────────────────────────
    ax_render = fig.add_subplot(gs[2, 0])
    _bar_chart(
        ax_render, legends,
        [runs[l]["pdf_metrics"]["mean_render_time_s"] for l in legends],
        "Mean PDF Render Time\n(PDF → vLLM ready)", "seconds", colors,
    )

    ax_total = fig.add_subplot(gs[2, 1])
    _bar_chart(
        ax_total, legends,
        [runs[l]["pdf_metrics"]["mean_total_time_s"] for l in legends],
        "Mean Total Time per PDF", "seconds", colors,
    )

    ax_norm = fig.add_subplot(gs[2, 2])
    groups_norm: dict[str, list[float]] = {
        "mean":   [runs[l]["normalized_page_time_ms_per_token"]["mean"]   for l in legends],
        "median": [runs[l]["normalized_page_time_ms_per_token"]["median"] for l in legends],
        "p90":    [runs[l]["normalized_page_time_ms_per_token"]["p90"]    for l in legends],
        "p99":    [runs[l]["normalized_page_time_ms_per_token"]["p99"]    for l in legends],
    }
    _grouped_bars(ax_norm, legends, groups_norm,
                  "Normalised Page Time\n(E2EL ms / output token)", "ms / token", colors)

    # ── Metadata table beneath the title ─────────────────────────────────────
    meta_lines = []
    for leg in legends:
        m = runs[leg].get("metadata", {})
        pages = m.get("num_pages", "?")
        pdfs  = m.get("num_pdfs",  "?")
        notes = m.get("notes", "") or ""
        notes_str = f"  [{notes}]" if notes else ""
        meta_lines.append(f"{leg}: {pdfs} PDFs, {pages} pages{notes_str}")
    fig.text(0.5, 0.955, "   |   ".join(meta_lines),
             ha="center", va="top", fontsize=8, color="#444444",
             style="italic")

    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[plot] Saved → {out}")
    else:
        plt.show()


if __name__ == "__main__":
    fire.Fire(plot)
