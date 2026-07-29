"""Baseline vs peak-repeat vs peak-filler accuracy — one figure per MODEL, bars grouped by
dataset (the transpose of plot_baseline_vs_peak.py, which is one figure per dataset, bars grouped
by model). Same underlying numbers (reuses compute_peaks/_cond_short_label/_readable_text_color
from that module rather than recomputing anything), useful for comparing one model's own
repeat/filler response across datasets side by side.

Datasets span very different accuracy ranges (Gen-Arithmetic/Comp-Math sit at 40-90%, the n-hop
tasks at 2-45%), so each figure splits into two panels with their own y-scale — a single 0-100
axis would flatten the n-hop bars to invisible slivers.

    python -m presentation.plot_baseline_vs_peak_by_model
    python -m presentation.plot_baseline_vs_peak_by_model --run condition_matched_500 --models anthropic/claude-fable-5,anthropic/claude-opus-5
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


def _dataset_groups(datasets: List[str], uniform_scale: bool = False) -> List[Tuple[str, List[str], int]]:
    """Split datasets into (panel title, dataset ids, y-axis max) groups by accuracy range —
    same 0-50-for-nhop / 0-100-otherwise split plot_baseline_vs_peak.py uses per figure, unless
    `uniform_scale` forces every panel to 0-100 (a separate, deliberately less-legible-for-nhop
    variant some readers may still want for direct cross-panel comparability)."""
    big = [d for d in datasets if not d.startswith("nhop")]
    hop = [d for d in datasets if d.startswith("nhop")]
    groups = []
    if big:
        groups.append(("Gen-Arithmetic / Comp-Math", big, 100))
    if hop:
        groups.append(("N-Hop", hop, 100 if uniform_scale else 50))
    return groups


def plot_one_model(model: str, peaks: Dict[Tuple[str, str], Panel], datasets: List[str],
                   out_path: Path, color_hex: str, uniform_scale: bool = False,
                   cis: Dict[Tuple[str, str, str], Tuple[float, float]] | None = None) -> Path:
    groups = _dataset_groups(datasets, uniform_scale)
    fig, axes = plt.subplots(1, len(groups),
                             figsize=(max(7, 2.6 * sum(len(g[1]) for g in groups)), 5.5))
    if len(groups) == 1:
        axes = [axes]
    color = f"#{color_hex}"
    width = 0.25
    group_gap = 1.0

    for ax, (panel_title, panel_datasets, y_max) in zip(axes, groups):
        x0 = 0.0
        xticks, xticklabels = [], []
        label_gap = y_max * 0.01
        for dataset in panel_datasets:
            baseline_acc, peak_repeat, peak_filler = peaks.get((model, dataset), (None, None, None))
            if baseline_acc is not None:
                vals = [baseline_acc * 100,
                        peak_repeat["acc_cond"] * 100 if peak_repeat else float("nan"),
                        peak_filler["acc_cond"] * 100 if peak_filler else float("nan")]
                rows = [None, peak_repeat, peak_filler]
                conds = [_baseline_condition(peak_repeat, peak_filler),
                        peak_repeat["condition"] if peak_repeat else None,
                        peak_filler["condition"] if peak_filler else None]
                for bi, (val, row, cond) in enumerate(zip(vals, rows, conds)):
                    if val != val:  # NaN -> no such condition for this dataset
                        continue
                    xpos = x0 + bi * width
                    bar_alpha = 1.0 if bi == 0 else 0.75
                    ax.bar(xpos, val, width=width * 0.92, color=color,
                           alpha=bar_alpha, edgecolor="#111111", linewidth=0.6, zorder=2)
                    ci = cis.get((model, dataset, cond)) if (cis is not None and cond) else None
                    _draw_ci(ax, xpos, val, ci, color_hex, bar_alpha)
                    sig_marker = " *" if (row is not None and row["sig_holm"]) else ""
                    ax.text(xpos, _value_label_y(val, y_max, label_gap, ci),
                           f"{val:.1f}%{sig_marker}", ha="center",
                           va="bottom", fontsize=7.5, fontweight="bold", zorder=5)
                    if row is not None:
                        cond_txt = _cond_short_label(row["condition"])
                        ax.text(xpos, label_gap, cond_txt, ha="center", va="bottom", fontsize=6.5,
                               color=_readable_text_color(color_hex, 0.75), zorder=5)
            group_center = x0 + width
            xticks.append(group_center)
            xticklabels.append(config.short_dataset(dataset))
            x0 += 3 * width + group_gap * 0.4

        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_ylabel("accuracy (%)")
        ax.set_ylim(0, y_max)
        ax.set_title(panel_title)

    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor="#888888", alpha=1.0, label="baseline"),
                      Patch(facecolor="#888888", alpha=0.75, label="peak repeat / peak filler")]
    axes[0].legend(handles=legend_handles, loc="upper right", fontsize=8)

    fig.suptitle(f"Baseline vs peak — {config.short_model(model)}", fontsize=13, fontweight="bold")
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
    ap.add_argument("--dataset", action="append", help="dataset id (repeatable); default: all in --run")
    ap.add_argument("--out-dir", default=str(config.FIGURES_DIR))
    ap.add_argument("--alpha", type=float, default=0.001,
                    help="Holm-corrected significance cutoff for the '*' marker (default 0.001, "
                         "matching the source LessWrong post's own reporting convention)")
    ap.add_argument("--uniform-scale", action="store_true",
                    help="force every panel to a 0-100 y-axis instead of 0-50 for n-hop; "
                         "saved under a separate filename (_uniform100 suffix), not overwriting "
                         "the default dual-scale figures")
    ap.add_argument("--no-error-bars", action="store_true",
                    help="omit the 95%% paired-bootstrap CI whisker on each bar (drawn by default)")
    args = ap.parse_args(argv)

    apply_style()
    spec: Dict[str, Any] = config.NAMED_RUNS[args.run]
    if args.models:
        spec = {**spec, "models": args.models.split(",")}
    models = spec["models"]
    datasets = args.dataset or list(spec["axes"].keys())

    peaks = compute_peaks(spec, alpha=args.alpha)
    cis = None if args.no_error_bars else hstats.paired_bootstrap_cis(spec, None)
    colors = config.model_colors(models)

    out_dir = Path(args.out_dir)
    suffix = "_uniform100" if args.uniform_scale else ""
    for model in models:
        out = out_dir / f"baseline_vs_peak_by_model_{config.short_model(model)}{suffix}.png"
        plot_one_model(model, peaks, datasets, out, colors[model], args.uniform_scale, cis)
        print(f"wrote {out}")
        for dataset in datasets:
            baseline_acc, peak_repeat, peak_filler = peaks.get((model, dataset), (None, None, None))
            if baseline_acc is None:
                print(f"  {dataset}: no data")
                continue
            pr = (f"{_cond_short_label(peak_repeat['condition'])}={peak_repeat['acc_cond']:.3f} "
                  f"(p_holm={peak_repeat['p_holm']:.4f})") if peak_repeat else "n/a"
            pf = (f"{_cond_short_label(peak_filler['condition'])}={peak_filler['acc_cond']:.3f} "
                  f"(p_holm={peak_filler['p_holm']:.4f})") if peak_filler else "n/a"
            print(f"  {dataset}: baseline={baseline_acc:.3f}  peak_repeat={pr}  peak_filler={pf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
