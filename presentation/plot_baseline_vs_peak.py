"""Baseline vs peak-repeat vs peak-filler accuracy — one figure per dataset, bars grouped by
model, in the style of the source paper's headline summary figure. "Peak" is whichever repeat/
filler condition scored highest for that (model, dataset); significance is the paired t-test of
that specific condition against baseline (Holm-corrected within the panel), reusing
harness/stats.py's existing paired_ttests rather than reimplementing significance testing here.

    python -m presentation.plot_baseline_vs_peak
    python -m presentation.plot_baseline_vs_peak --run condition_matched_500 --dataset nhop_3
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from harness import config, stats as hstats
from presentation.figures import apply_style

DEFAULT_RUN_NAME = "condition_matched_500"

Panel = Tuple[Optional[float], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]  # baseline, peak_repeat_row, peak_filler_row


def compute_peaks(spec: Dict[str, Any], alpha: float = 0.001) -> Dict[Tuple[str, str], Panel]:
    """(model, dataset) -> (baseline_acc, peak_repeat_row, peak_filler_row). A row is None if
    that dataset's axes don't include that condition kind (e.g. no repeats swept).

    `alpha` (default 0.001, matching the source LessWrong post's own reporting convention,
    stricter than the usual 0.05) is the Holm-corrected significance cutoff for the '*' marker."""
    rows = hstats.paired_ttests(spec, None, alpha=alpha)
    by_panel: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_panel[(r["model"], r["dataset"])].append(r)

    result: Dict[Tuple[str, str], Panel] = {}
    for key, panel_rows in by_panel.items():
        baseline_acc = panel_rows[0]["acc_baseline"] if panel_rows else None
        repeats = [r for r in panel_rows if r["condition"].startswith("repeat_")]
        fillers = [r for r in panel_rows if r["condition"].startswith("filler_")]
        peak_repeat = max(repeats, key=lambda r: r["acc_cond"]) if repeats else None
        peak_filler = max(fillers, key=lambda r: r["acc_cond"]) if fillers else None
        result[key] = (baseline_acc, peak_repeat, peak_filler)
    return result


def _readable_text_color(hex_color: str, alpha: float) -> str:
    """White or near-black, whichever reads against `hex_color` rendered at `alpha` over the
    figure's white background — a fixed white broke down on lighter bars (e.g. opus-4.5's
    light-red peak bars), where it was nearly invisible."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (alpha * c + (1 - alpha) * 255 for c in (r, g, b))
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 150 else "#222222"


def _cond_short_label(cond: str) -> str:
    """'repeat_r10+md' -> 'r=10'; 'filler_f300+md' -> 'f=300'."""
    base = cond.split("+")[0]
    kind, _, num = base.partition("_")
    value = num[1:] if len(num) > 1 else num
    return f"{'r' if kind == 'repeat' else 'f'}={value}"


def _baseline_condition(peak_repeat: Dict[str, Any] | None, peak_filler: Dict[str, Any] | None) -> str:
    """The baseline condition label matching this panel's match-demos suffix ('repeat_r10+md'
    pairs with 'baseline+md', not bare 'baseline')."""
    row = peak_repeat or peak_filler
    if row is None:
        return "baseline"
    return "baseline+md" if row["condition"].endswith("+md") else "baseline"


def _ci_shade(hex_color: str, bar_alpha: float) -> str:
    """A shade of `hex_color` for its own CI whisker — lighter for a dark bar, darker for a light
    one (blending toward white/black respectively) — so the whisker visibly belongs to that
    model's own color rather than a neutral gray that can wash out against a dark fill, and
    reads clearly either way."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    er, eg, eb = (bar_alpha * c + (1 - bar_alpha) * 255 for c in (r, g, b))
    luminance = 0.299 * er + 0.587 * eg + 0.114 * eb
    mix = 0.55
    if luminance < 150:
        rr, gg, bb = (c * (1 - mix) + 255 * mix for c in (r, g, b))  # lighten
    else:
        rr, gg, bb = (c * (1 - mix) for c in (r, g, b))  # darken
    return f"#{int(rr):02x}{int(gg):02x}{int(bb):02x}"


def _draw_ci(ax: Any, xpos: float, val: float, ci: Tuple[float, float] | None,
            color_hex: str, bar_alpha: float) -> None:
    """A 95% paired-bootstrap CI whisker on one bar, shaded from that bar's own color (see
    `_ci_shade`) so it reads as belonging to that model rather than a neutral overlay.
    Note: since these are PAIRED comparisons, two bars' CIs visually overlapping (or not) doesn't
    itself determine paired significance the way the '*' marker does — this is descriptive, not
    the test."""
    if ci is None:
        return
    lo, hi = ci[0] * 100, ci[1] * 100
    ecolor = _ci_shade(color_hex, bar_alpha)
    ax.errorbar(xpos, val, yerr=[[val - lo], [hi - val]], fmt="none", ecolor=ecolor,
               elinewidth=1.0, capsize=3, capthick=1.0, alpha=0.85, zorder=4)


def _value_label_y(val: float, y_max: float, label_gap: float,
                   ci: Tuple[float, float] | None) -> float:
    """Where the accuracy-value label sits: a FIXED, consistent gap above the bar top in the
    common case — only nudged up to clear the CI's upper whisker if that whisker's cap would
    otherwise land right on top of the number (rather than always floating the label above the
    whisker regardless of how tall it is, which looks inconsistent bar-to-bar since CI width
    varies a lot with sample size and base rate)."""
    default_y = val + label_gap
    if ci is None:
        return default_y
    whisker_top = ci[1] * 100
    text_height = y_max * 0.045  # rough estimate of the label's own vertical extent
    if default_y <= whisker_top <= default_y + text_height:
        return whisker_top + label_gap
    return default_y


def plot_one_dataset(dataset: str, peaks: Dict[Tuple[str, str], Panel], models: List[str],
                     out_path: Path, cis: Dict[Tuple[str, str, str], Tuple[float, float]] | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(max(7, 2.2 * len(models)), 5.5))
    width = 0.25
    group_gap = 1.0
    x0 = 0.0
    xticks, xticklabels = [], []
    y_max = 50 if dataset.startswith("nhop") else 100
    # A flat data-unit offset (e.g. "1.0") would sit at a different physical height on a 0-50
    # axis than on a 0-100 one, since the same figure height maps to twice the pixels-per-unit —
    # scale the gap by y_max so every dataset's labels sit the same visual distance from the bar.
    label_gap = y_max * 0.01
    colors = config.model_colors(models)

    for model in models:
        baseline_acc, peak_repeat, peak_filler = peaks.get((model, dataset), (None, None, None))
        if baseline_acc is None:
            continue
        vals = [baseline_acc * 100,
                peak_repeat["acc_cond"] * 100 if peak_repeat else float("nan"),
                peak_filler["acc_cond"] * 100 if peak_filler else float("nan")]
        rows = [None, peak_repeat, peak_filler]
        conds = [_baseline_condition(peak_repeat, peak_filler),
                peak_repeat["condition"] if peak_repeat else None,
                peak_filler["condition"] if peak_filler else None]
        color = f"#{colors[model]}"
        centers = []
        for bi, (val, row, cond) in enumerate(zip(vals, rows, conds)):
            if val != val:  # NaN -> no such condition for this dataset
                continue
            xpos = x0 + bi * width
            centers.append(xpos)
            ax.bar(xpos, val, width=width * 0.92, color=color,
                   alpha=1.0 if bi == 0 else 0.75, edgecolor="#111111", linewidth=0.6, zorder=2)
            bar_alpha = 1.0 if bi == 0 else 0.75
            ci = cis.get((model, dataset, cond)) if (cis is not None and cond) else None
            _draw_ci(ax, xpos, val, ci, colors[model], bar_alpha)
            # The accuracy value + significance marker sit at a consistent gap above the bar top
            # (nudged above the CI whisker only if its cap would otherwise land on the number);
            # the condition label (peak bars only) sits at the bar's base, in a color chosen to
            # read against that bar's own color, regardless of the bar's height.
            sig_marker = " *" if (row is not None and row["sig_holm"]) else ""
            ax.text(xpos, _value_label_y(val, y_max, label_gap, ci), f"{val:.1f}%{sig_marker}",
                   ha="center", va="bottom", fontsize=7.5, fontweight="bold", zorder=5)
            if row is not None:
                cond_txt = _cond_short_label(row["condition"])
                ax.text(xpos, label_gap, cond_txt, ha="center", va="bottom", fontsize=6.5,
                       color=_readable_text_color(colors[model], 0.75), zorder=5)
        group_center = x0 + width
        xticks.append(group_center)
        xticklabels.append(config.short_model(model))
        x0 += 3 * width + group_gap * 0.4

    # One shared legend explaining the bar triplet + significance marker, not per-model.
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor="#888888", alpha=1.0, label="baseline"),
                      Patch(facecolor="#888888", alpha=0.75, label="peak repeat / peak filler")]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, y_max)
    ax.set_title(f"Baseline vs peak — {config.short_dataset(dataset)}")
    fig.tight_layout()
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
    ap.add_argument("--no-error-bars", action="store_true",
                    help="omit the subtle 95%% paired-bootstrap CI whisker on each bar (drawn by default)")
    args = ap.parse_args(argv)

    apply_style()
    spec = config.NAMED_RUNS[args.run]
    if args.models:
        spec = {**spec, "models": args.models.split(",")}
    models = spec["models"]
    datasets = args.dataset or list(spec["axes"].keys())

    peaks = compute_peaks(spec, alpha=args.alpha)
    cis = None if args.no_error_bars else hstats.paired_bootstrap_cis(spec, None)

    out_dir = Path(args.out_dir)
    for dataset in datasets:
        out = out_dir / f"baseline_vs_peak_{dataset}.png"
        plot_one_dataset(dataset, peaks, models, out, cis)
        print(f"wrote {out}")
        for model in models:
            baseline_acc, peak_repeat, peak_filler = peaks.get((model, dataset), (None, None, None))
            if baseline_acc is None:
                print(f"  {model}: no data")
                continue
            pr = (f"{_cond_short_label(peak_repeat['condition'])}={peak_repeat['acc_cond']:.3f} "
                  f"(p_holm={peak_repeat['p_holm']:.4f})") if peak_repeat else "n/a"
            pf = (f"{_cond_short_label(peak_filler['condition'])}={peak_filler['acc_cond']:.3f} "
                  f"(p_holm={peak_filler['p_holm']:.4f})") if peak_filler else "n/a"
            print(f"  {model}: baseline={baseline_acc:.3f}  peak_repeat={pr}  peak_filler={pf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
