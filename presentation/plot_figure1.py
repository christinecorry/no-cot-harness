"""Figure 1 — the headline comparison: baseline vs. peak (whichever of repeat/filler scored
higher) for every model, side by side on Gen-Arithmetic and 2-Hop. Two panels in one image,
each styled like plot_baseline_vs_peak.py's per-dataset figure but with a single peak bar per
model instead of separate peak-repeat/peak-filler bars — reuses that module's compute_peaks
rather than recomputing anything.

    python -m presentation.plot_figure1
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt

from harness import config, stats as hstats
from presentation.figures import apply_style
from presentation.plot_baseline_vs_peak import (
    DEFAULT_RUN_NAME, Panel, _baseline_condition, _cond_short_label, _draw_ci,
    _readable_text_color, _value_label_y, compute_peaks,
)

DATASETS = ["gen_arithmetic_500", "nhop_2"]


def _best_peak(baseline_acc: float, peak_repeat: Dict[str, Any] | None,
               peak_filler: Dict[str, Any] | None) -> Dict[str, Any] | None:
    """Whichever of peak-repeat/peak-filler scored the higher accuracy for this (model, dataset)."""
    candidates = [r for r in (peak_repeat, peak_filler) if r is not None]
    return max(candidates, key=lambda r: r["acc_cond"]) if candidates else None


def plot_figure1(peaks: Dict[Tuple[str, str], Panel], models: List[str], datasets: List[str],
                 out_path: Path,
                 cis: Dict[Tuple[str, str, str], Tuple[float, float]] | None = None) -> Path:
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.5 * len(datasets), 5.5))
    if len(datasets) == 1:
        axes = [axes]
    colors = config.model_colors(models)
    width = 0.32
    group_gap = 0.5

    for ax, dataset in zip(axes, datasets):
        y_max = 50 if dataset.startswith("nhop") else 100
        label_gap = y_max * 0.01
        x0 = 0.0
        xticks, xticklabels = [], []
        for model in models:
            baseline_acc, peak_repeat, peak_filler = peaks.get((model, dataset), (None, None, None))
            if baseline_acc is None:
                continue
            peak = _best_peak(baseline_acc, peak_repeat, peak_filler)
            vals = [baseline_acc * 100, peak["acc_cond"] * 100 if peak else float("nan")]
            rows = [None, peak]
            conds = [_baseline_condition(peak_repeat, peak_filler),
                    peak["condition"] if peak else None]
            color = f"#{colors[model]}"
            for bi, (val, row, cond) in enumerate(zip(vals, rows, conds)):
                if val != val:
                    continue
                xpos = x0 + bi * width
                bar_alpha = 1.0 if bi == 0 else 0.75
                ax.bar(xpos, val, width=width * 0.92, color=color,
                       alpha=bar_alpha, edgecolor="#111111", linewidth=0.6, zorder=2)
                ci = cis.get((model, dataset, cond)) if (cis is not None and cond) else None
                _draw_ci(ax, xpos, val, ci, colors[model], bar_alpha)
                sig_marker = " *" if (row is not None and row["sig_holm"]) else ""
                ax.text(xpos, _value_label_y(val, y_max, label_gap, ci),
                       f"{val:.1f}%{sig_marker}", ha="center",
                       va="bottom", fontsize=8, fontweight="bold", zorder=5)
                if row is not None:
                    cond_txt = _cond_short_label(row["condition"])
                    ax.text(xpos, label_gap, cond_txt, ha="center", va="bottom", fontsize=7,
                           color=_readable_text_color(colors[model], 0.75), zorder=5)
            group_center = x0 + width / 2
            xticks.append(group_center)
            xticklabels.append(config.short_model(model))
            x0 += 2 * width + group_gap

        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_ylabel("accuracy (%)")
        ax.set_ylim(0, y_max)
        ax.set_title(config.short_dataset(dataset))

    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor="#888888", alpha=1.0, label="baseline"),
                      Patch(facecolor="#888888", alpha=0.75, label="peak (repeat or filler)")]
    axes[0].legend(handles=legend_handles, loc="upper left", fontsize=8)

    fig.suptitle("Baseline vs. Peak No-CoT Augmentation", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=DEFAULT_RUN_NAME, choices=list(config.NAMED_RUNS))
    ap.add_argument("--models", help="comma-separated model ids; default: the chosen --run's own list")
    ap.add_argument("--dataset", action="append", help="dataset id (repeatable); default: gen_arithmetic_500,nhop_2")
    ap.add_argument("--out", default=str(config.FIGURES_DIR / "figure1_baseline_vs_peak.png"))
    ap.add_argument("--alpha", type=float, default=0.001,
                    help="Holm-corrected significance cutoff for the '*' marker (default 0.001)")
    ap.add_argument("--no-error-bars", action="store_true",
                    help="omit the 95%% paired-bootstrap CI whisker on each bar (drawn by default)")
    args = ap.parse_args(argv)

    apply_style()
    spec: Dict[str, Any] = config.NAMED_RUNS[args.run]
    if args.models:
        spec = {**spec, "models": args.models.split(",")}
    models = spec["models"]
    datasets = args.dataset or DATASETS

    peaks = compute_peaks(spec, alpha=args.alpha)
    cis = None if args.no_error_bars else hstats.paired_bootstrap_cis(spec, None)
    out = Path(args.out)
    plot_figure1(peaks, models, datasets, out, cis)
    print(f"wrote {out}")
    for dataset in datasets:
        for model in models:
            baseline_acc, peak_repeat, peak_filler = peaks.get((model, dataset), (None, None, None))
            if baseline_acc is None:
                print(f"  {dataset} {model}: no data")
                continue
            peak = _best_peak(baseline_acc, peak_repeat, peak_filler)
            peak_desc = (f"{_cond_short_label(peak['condition'])}={peak['acc_cond']:.3f} "
                        f"(p_holm={peak['p_holm']:.4f})") if peak else "n/a"
            print(f"  {dataset} {config.short_model(model)}: baseline={baseline_acc:.3f}  peak={peak_desc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
