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
_LINE_COLORS = ["5e0000", "d45d00", "1a1a1a", "c9a227", "666666", "800000"]

Panel = Tuple[Optional[float], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]  # baseline, peak_repeat_row, peak_filler_row


def compute_peaks(spec: Dict[str, Any]) -> Dict[Tuple[str, str], Panel]:
    """(model, dataset) -> (baseline_acc, peak_repeat_row, peak_filler_row). A row is None if
    that dataset's axes don't include that condition kind (e.g. no repeats swept)."""
    rows = hstats.paired_ttests(spec, None)
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


def _cond_short_label(cond: str) -> str:
    """'repeat_r10+md' -> 'r=10'; 'filler_f300+md' -> 'f=300'."""
    base = cond.split("+")[0]
    kind, _, num = base.partition("_")
    value = num[1:] if len(num) > 1 else num
    return f"{'r' if kind == 'repeat' else 'f'}={value}"


def plot_one_dataset(dataset: str, peaks: Dict[Tuple[str, str], Panel], models: List[str],
                     out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(max(7, 2.2 * len(models)), 5.5))
    width = 0.25
    group_gap = 1.0
    x0 = 0.0
    xticks, xticklabels = [], []

    for mi, model in enumerate(models):
        baseline_acc, peak_repeat, peak_filler = peaks.get((model, dataset), (None, None, None))
        if baseline_acc is None:
            continue
        vals = [baseline_acc * 100,
                peak_repeat["acc_cond"] * 100 if peak_repeat else float("nan"),
                peak_filler["acc_cond"] * 100 if peak_filler else float("nan")]
        rows = [None, peak_repeat, peak_filler]
        color = f"#{_LINE_COLORS[mi % len(_LINE_COLORS)]}"
        centers = []
        for bi, (val, row) in enumerate(zip(vals, rows)):
            if val != val:  # NaN -> no such condition for this dataset
                continue
            xpos = x0 + bi * width
            centers.append(xpos)
            ax.bar(xpos, val, width=width * 0.92, color=color,
                   alpha=1.0 if bi == 0 else 0.75, edgecolor="#111111", linewidth=0.6)
            # The accuracy value + significance marker sit at the bar top; the condition label
            # (peak bars only) is printed vertically at the bar's base, inside the bar, so it
            # never competes for space with the value/marker above.
            sig_marker = f" {'*' if row['sig_holm'] else 'ns'}" if row is not None else ""
            ax.text(xpos, val + 1.0, f"{val:.1f}%{sig_marker}", ha="center", va="bottom",
                   fontsize=7.5, fontweight="bold")
            if row is not None:
                ax.text(xpos, 1.0, _cond_short_label(row["condition"]), ha="center", va="bottom",
                       fontsize=7, color="white", rotation=90)
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
    ax.set_ylim(0, 50 if dataset.startswith("nhop") else 100)
    ax.set_title(f"Baseline vs peak — {config.short_dataset(dataset)}")
    fig.text(0.5, 0.01,
             "Peak = highest-scoring repeat/filler condition per model. Label = condition, "
             "then '*' (Holm-corrected paired t-test p<0.05 vs baseline) or 'ns'.",
             ha="center", fontsize=8, color="#333333")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
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
    args = ap.parse_args(argv)

    apply_style()
    spec = config.NAMED_RUNS[args.run]
    if args.models:
        spec = {**spec, "models": args.models.split(",")}
    models = spec["models"]
    datasets = args.dataset or list(spec["axes"].keys())

    peaks = compute_peaks(spec)

    out_dir = Path(args.out_dir)
    for dataset in datasets:
        out = out_dir / f"{args.run}_baseline_vs_peak_{dataset}.png"
        plot_one_dataset(dataset, peaks, models, out)
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
